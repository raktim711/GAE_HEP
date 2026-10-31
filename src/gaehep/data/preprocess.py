"""Cluster LHC Olympics R&D events into anti-kT R=1.0 jets and store the two leading jets.

Input: a pandas HDF5 file whose rows are events and whose columns are 700 x (pT, eta, phi) zero-padded
particles (+ a truth-label column in the 2-prong R&D file).  The black-box files (samples ``bb1``-``bb3``) have
no label column; their labels come from the master key (one label per line, in file order; BB2 has none and is
background only).  The official LHCO examples cluster with
anti-kT, R = 1.0 and keep inclusive jets with pT > 20 GeV; we do the same with scikit-hep ``fastjet``.

Outputs, written to ``processed/<sample>/`` (``.npy`` files are memory-mappable):

``constituents.npy``  float32 (n, 2, N_MAX, 3)  (pT, eta, phi) of each leading jet's constituents,
                                                  sorted by pT (descending), truncated / zero padded
``n_const.npy``       int32   (n, 2)            untruncated constituent multiplicity (0 if no jet)
``jets_p4.npy``       float64 (n, 2, 4)         (px, py, pz, E) of the two leading jets (NaN if missing)
``n_jets.npy``        int16   (n,)              number of inclusive jets with pT > ptmin
``labels.npy``        int8    (n,)              truth label (1 = signal); 1 for the 3-prong-only file;
                                                  black boxes: 0 / 1 (/ 2, the second signal of BB3)
``meta.json``                                   settings, timings, provenance

Processing is chunked (the raw file is 17 GB in memory) and parallelised over chunks; finished chunks
are recorded so that an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from gaehep.config import load_json, processed_dir, raw_dir, run_metadata, save_json

N_PARTICLES = 700
N_MAX = 100
JET_R = 1.0
JET_PTMIN = 20.0

RAW_FILES = {
    "rnd": "events_anomalydetection_v2.h5",
    "qqq": "events_anomalydetection_Z_XY_qqq.h5",
    "bb1": "events_LHCO2020_BlackBox1.h5",
    "bb2": "events_LHCO2020_BlackBox2.h5",
    "bb3": "events_LHCO2020_BlackBox3.h5",
}
# black-box master keys (BB2 has none: all its events are background)
MASTER_KEYS = {
    "bb1": "events_LHCO2020_BlackBox1.masterkey",
    "bb3": "events_LHCO2020_BlackBox3.masterkey",
}
MASTER_KEY_LABELS = (0, 1, 2)  # 0 background, 1 signal, 2 the second signal of BB3


# ----------------------------------------------------------------------------------------------- reading
# The LHCO files are pandas "fixed"-format HDF5 frames with a single float64 block stored row-major
# (``/df/block0_values``, shape (n_events, 2101), blosc-compressed).  The main R&D file was written by an
# old pandas whose string attributes are bytes, which pandas >= 3 can no longer parse, so we read the block
# directly with PyTables (also faster: no DataFrame construction).
def _values_node(h):  # h: tables.File
    df = h.root.df
    nblocks = int(df._v_attrs["nblocks"]) if "nblocks" in df._v_attrs._f_list("user") else 1
    if nblocks != 1 or not hasattr(df, "block0_values"):
        raise ValueError("expected a pandas fixed-format frame with a single block")
    items = df.block0_items[:]
    if not np.array_equal(items.astype(np.int64), np.arange(len(items))):
        raise ValueError("unexpected column order in block0_items")
    node = df.block0_values
    if "transposed" in node._v_attrs._f_list("user") and not bool(node._v_attrs["transposed"]):
        raise ValueError("block stored column-major; row slicing unsupported")
    return node


def n_rows(path: str | Path) -> int:
    """Number of events (rows) in an LHCO pandas-HDF5 file without loading it."""
    import tables

    with tables.open_file(str(path), mode="r") as h:
        return int(_values_node(h).shape[0])


def read_events(path: str | Path, start: int, stop: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Read events ``[start, stop)``: returns particles (n, 700, 3) float64 and labels (n,) int8 or None."""
    import tables

    with tables.open_file(str(path), mode="r") as h:
        arr = np.asarray(_values_node(h)[start:stop], dtype=np.float64)
    ncol = arr.shape[1]
    if ncol == 3 * N_PARTICLES + 1:
        labels = arr[:, -1].astype(np.int8)
        parts = arr[:, :-1]
    elif ncol == 3 * N_PARTICLES:
        labels, parts = None, arr
    else:
        raise ValueError(f"unexpected number of columns {ncol} in {path}")
    return np.ascontiguousarray(parts).reshape(len(arr), N_PARTICLES, 3), labels


def read_master_key(path: str | Path) -> np.ndarray:
    """Labels of a black box: one per line in event order (the files hold floats such as ``0.0``).
    Returns int8 (n,); raises if a value is not 0, 1 or 2."""
    vals = np.loadtxt(path, dtype=np.float64, ndmin=1)
    lab = vals.astype(np.int8)
    if not np.array_equal(lab, vals) or not np.isin(lab, MASTER_KEY_LABELS).all():
        raise ValueError(f"{path}: master key labels must be 0, 1 or 2")
    return lab


# -------------------------------------------------------------------------------------------- clustering
def cluster_events(
    parts: np.ndarray, n_max: int = N_MAX, r: float = JET_R, ptmin: float = JET_PTMIN
) -> dict[str, np.ndarray]:
    """Cluster events and extract the two leading (by pT) jets and their constituents.

    Parameters
    ----------
    parts : (n, P, 3) array of massless particles (pT, eta, phi); rows with pT <= 0 are padding.

    Returns a dict with ``constituents`` (n, 2, n_max, 3) float32, ``n_const`` (n, 2) int32,
    ``jets_p4`` (n, 2, 4) float64 and ``n_jets`` (n,) int16.
    """
    import awkward as ak
    import fastjet
    import vector

    vector.register_awkward()

    n = parts.shape[0]
    pt, eta, phi = parts[..., 0], parts[..., 1], parts[..., 2]
    mask = pt > 0.0
    counts = mask.sum(axis=1)
    f_pt, f_eta, f_phi = pt[mask], eta[mask], phi[mask]  # row-major -> event-contiguous
    offsets = np.zeros(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])

    def jag(a: np.ndarray) -> ak.Array:
        return ak.unflatten(a, counts)

    particles = ak.zip(
        {
            "px": jag(f_pt * np.cos(f_phi)),
            "py": jag(f_pt * np.sin(f_phi)),
            "pz": jag(f_pt * np.sinh(f_eta)),
            "E": jag(f_pt * np.cosh(f_eta)),
        },
        with_name="Momentum4D",
    )
    cs = fastjet.ClusterSequence(particles, fastjet.JetDefinition(fastjet.antikt_algorithm, r))
    jets = cs.inclusive_jets(min_pt=ptmin)
    cidx = cs.constituent_index(min_pt=ptmin)

    # order jets by pT (descending); constituent_index is aligned with inclusive_jets
    jpt = np.sqrt(jets.px**2 + jets.py**2)
    order = ak.argsort(jpt, axis=1, ascending=False, stable=True)
    jets, cidx = jets[order], cidx[order]
    n_jets = ak.to_numpy(ak.num(jets, axis=1)).astype(np.int16)

    lead, lead_c = jets[:, :2], cidx[:, :2]

    def pad2(x: ak.Array) -> np.ndarray:
        return ak.to_numpy(ak.fill_none(ak.pad_none(x, 2, axis=1, clip=True), np.nan)).astype(np.float64)

    jets_p4 = np.stack([pad2(lead.px), pad2(lead.py), pad2(lead.pz), pad2(lead.E)], axis=-1)

    # flatten (event, jet rank, constituent) -> 1D arrays, fully vectorised
    rank = ak.local_index(lead_c, axis=1)
    ev = ak.broadcast_arrays(np.arange(n)[:, None], rank)[0]
    ev_b, rank_b, _ = ak.broadcast_arrays(ev[:, :, None], rank[:, :, None], lead_c)
    ev_f = ak.to_numpy(ak.flatten(ev_b, axis=None)).astype(np.int64)
    rk_f = ak.to_numpy(ak.flatten(rank_b, axis=None)).astype(np.int64)
    loc_f = ak.to_numpy(ak.flatten(lead_c, axis=None)).astype(np.int64)
    gidx = offsets[ev_f] + loc_f
    c_pt, c_eta, c_phi = f_pt[gidx], f_eta[gidx], f_phi[gidx]

    group = ev_f * 2 + rk_f
    srt = np.lexsort((-c_pt, group))  # by group, then pT descending
    group, c_pt, c_eta, c_phi = group[srt], c_pt[srt], c_eta[srt], c_phi[srt]
    n_const = np.bincount(group, minlength=2 * n).astype(np.int32)
    starts = np.zeros(2 * n, dtype=np.int64)
    np.cumsum(n_const[:-1], out=starts[1:])
    pos = np.arange(len(group)) - starts[group]
    keep = pos < n_max

    constituents = np.zeros((n, 2, n_max, 3), dtype=np.float32)
    g, p = group[keep], pos[keep]
    constituents[g // 2, g % 2, p, 0] = c_pt[keep]
    constituents[g // 2, g % 2, p, 1] = c_eta[keep]
    constituents[g // 2, g % 2, p, 2] = c_phi[keep]

    return {
        "constituents": constituents,
        "n_const": n_const.reshape(n, 2),
        "jets_p4": jets_p4,
        "n_jets": n_jets,
    }


# ----------------------------------------------------------------------------------------- orchestration
_OUT_SPECS = {
    "constituents": (np.float32, (2, N_MAX, 3)),
    "n_const": (np.int32, (2,)),
    "jets_p4": (np.float64, (2, 4)),
    "n_jets": (np.int16, ()),
    "labels": (np.int8, ()),
}


def _allocate(out_dir: Path, n: int, n_max: int) -> None:
    from numpy.lib.format import open_memmap

    for name, (dtype, shape) in _OUT_SPECS.items():
        path = out_dir / f"{name}.npy"
        shape = (2, n_max, 3) if name == "constituents" else shape
        if path.exists():
            mm = np.load(path, mmap_mode="r")
            if mm.shape == (n, *shape) and mm.dtype == dtype:
                continue
        mm = open_memmap(path, mode="w+", dtype=dtype, shape=(n, *shape))
        if name == "jets_p4":
            mm[:] = np.nan
        mm.flush()
        del mm


def _work(job: tuple) -> tuple[int, int, float]:
    """Cluster events ``[start, stop)``.  ``label_source`` is an int (the label of every event, used when the
    file has no label column) or an int8 array with the labels of the chunk (the master key)."""
    src, out_dir, start, stop, n_max, label_source = job
    out = Path(out_dir)
    t0 = time.time()
    parts, labels = read_events(src, start, stop)
    res = cluster_events(parts, n_max=n_max)
    if labels is None:
        labels = (
            np.full(len(parts), label_source, dtype=np.int8)
            if np.isscalar(label_source)
            else np.asarray(label_source, dtype=np.int8)
        )
    res["labels"] = labels
    for name, arr in res.items():
        mm = np.load(out / f"{name}.npy", mmap_mode="r+")
        mm[start:stop] = arr
        mm.flush()
        del mm
    (out / "chunks").mkdir(exist_ok=True)
    (out / "chunks" / f"{start:08d}_{stop:08d}.ok").touch()
    return start, stop, time.time() - t0


def verify_processed(out_dir: Path, n: int, n_max: int, labels: np.ndarray | None, chunk: int) -> list[str]:
    """Problems with a finished run (empty list = verified): every chunk marked done, the arrays complete
    (the two leading jets of every event with >= 2 jets are finite, the rest is NaN-free where defined),
    and the labels equal to the master key."""
    problems = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        if not (out_dir / "chunks" / f"{s:08d}_{e:08d}.ok").exists():
            problems.append(f"chunk {s}-{e} is not marked done")
    if problems:
        return problems
    n_jets = np.load(out_dir / "n_jets.npy")
    p4 = np.load(out_dir / "jets_p4.npy", mmap_mode="r")
    const = np.load(out_dir / "constituents.npy", mmap_mode="r")
    if len(n_jets) != n or p4.shape[0] != n or const.shape != (n, 2, n_max, 3):
        problems.append("array shapes do not match the number of events")
        return problems
    two = np.flatnonzero(n_jets >= 2)
    for lo in range(0, len(two), 100_000):
        if not np.isfinite(p4[two[lo : lo + 100_000]]).all():
            problems.append("an event with two jets has a non-finite jet four-vector")
            break
    if labels is not None and not np.array_equal(np.load(out_dir / "labels.npy"), labels):
        problems.append("labels.npy differs from the master key")
    return problems


def preprocess(
    sample: str,
    src: Path | None = None,
    out_root: Path | None = None,
    chunk: int = 20000,
    workers: int = 2,
    limit: int | None = None,
    n_max: int = N_MAX,
    delete_raw: bool = False,
    master_key: Path | None = None,
) -> Path:
    """Cluster ``sample``.  ``delete_raw`` removes the raw events file once the run is verified
    (:func:`verify_processed`); it is refused for a ``limit``-ed run.  ``master_key`` overrides the black
    box's key file (default: next to the raw file)."""
    src = Path(src) if src else raw_dir() / RAW_FILES[sample]
    out_dir = (Path(out_root) if out_root else processed_dir()) / sample
    meta_path = out_dir / "meta.json"
    if not src.exists() and meta_path.exists() and load_json(meta_path).get("raw_deleted"):
        print(f"[{sample}] already processed (raw file deleted after verification): {out_dir}", flush=True)
        return out_dir
    if delete_raw and limit:
        raise ValueError("--delete-raw is refused for a --limit-ed (partial) run")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = n_rows(src)
    if limit:
        n = min(n, limit)
    _allocate(out_dir, n, n_max)
    key = None
    if sample in MASTER_KEYS:
        key_path = Path(master_key) if master_key else src.with_name(MASTER_KEYS[sample])
        key = read_master_key(key_path)
        if len(key) < n:
            raise ValueError(f"{key_path}: {len(key)} labels for {n} events")
        if not limit and len(key) != n:
            raise ValueError(f"{key_path}: {len(key)} labels for {n} events")
        key = key[:n]
    default_label = 1 if sample == "qqq" else 0
    jobs = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        if (out_dir / "chunks" / f"{s:08d}_{e:08d}.ok").exists():
            continue
        jobs.append((str(src), str(out_dir), s, e, n_max, default_label if key is None else key[s:e]))
    print(f"[{sample}] {n} events, {len(jobs)} chunk(s) to process with {workers} worker(s)", flush=True)
    t0 = time.time()
    done = 0
    if jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, maxtasksperchild=4) as pool:
            for s, e, dt in pool.imap_unordered(_work, jobs):
                done += 1
                el = time.time() - t0
                eta = el / done * (len(jobs) - done)
                print(
                    f"  chunk {s}-{e} done in {dt:.1f}s ({done}/{len(jobs)}, elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min)",
                    flush=True,
                )
    meta = {
        "sample": sample,
        "source": str(src),
        "n_events": n,
        "n_max": n_max,
        "jet_algorithm": "anti-kt",
        "R": JET_R,
        "ptmin": JET_PTMIN,
        "chunk": chunk,
        "master_key": None if key is None else str(key_path),
        "seconds": time.time() - t0,
        "raw_deleted": False,
        **run_metadata(),
    }
    save_json(meta, meta_path)
    print(f"[{sample}] finished in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)
    if delete_raw:
        problems = verify_processed(out_dir, n, n_max, key, chunk)
        if problems:
            raise RuntimeError(f"[{sample}] not deleting {src}: " + "; ".join(problems))
        src.unlink()
        save_json({**meta, "raw_deleted": True}, meta_path)
        print(f"[{sample}] verified; deleted {src}", flush=True)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", choices=list(RAW_FILES), required=True)
    p.add_argument("--src", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="only the first N events (smoke tests)")
    p.add_argument("--n-max", type=int, default=N_MAX)
    p.add_argument(
        "--master-key", type=Path, default=None, help="black box labels (default: next to the raw file)"
    )
    p.add_argument(
        "--delete-raw", action="store_true", help="delete the raw events file after a verified full run"
    )
    a = p.parse_args(argv)
    preprocess(a.sample, a.src, a.out_root, a.chunk, a.workers, a.limit, a.n_max, a.delete_raw, a.master_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
