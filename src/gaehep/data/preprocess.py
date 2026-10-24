"""Cluster LHC Olympics R&D events into anti-kT R=1.0 jets and store the two leading jets.

Input: a pandas HDF5 file whose rows are events and whose columns are 700 x (pT, eta, phi) zero-padded
particles (+ a truth-label column in the 2-prong R&D file).  The official LHCO examples cluster with
anti-kT, R = 1.0 and keep inclusive jets with pT > 20 GeV; we do the same with scikit-hep ``fastjet``.

Outputs, written to ``processed/<sample>/`` (``.npy`` files are memory-mappable):

``constituents.npy``  float32 (n, 2, N_MAX, 3)  (pT, eta, phi) of each leading jet's constituents,
                                                  sorted by pT (descending), truncated / zero padded
``n_const.npy``       int32   (n, 2)            untruncated constituent multiplicity (0 if no jet)
``jets_p4.npy``       float64 (n, 2, 4)         (px, py, pz, E) of the two leading jets (NaN if missing)
``n_jets.npy``        int16   (n,)              number of inclusive jets with pT > ptmin
``labels.npy``        int8    (n,)              truth label (1 = signal); 1 for the 3-prong-only file
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

from gaehep.config import processed_dir, raw_dir, run_metadata, save_json

N_PARTICLES = 700
N_MAX = 100
JET_R = 1.0
JET_PTMIN = 20.0

RAW_FILES = {
    "rnd": "events_anomalydetection_v2.h5",
    "qqq": "events_anomalydetection_Z_XY_qqq.h5",
}


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


def _work(job: tuple[str, str, int, int, int, int]) -> tuple[int, int, float]:
    src, out_dir, start, stop, n_max, default_label = job
    out = Path(out_dir)
    t0 = time.time()
    parts, labels = read_events(src, start, stop)
    res = cluster_events(parts, n_max=n_max)
    res["labels"] = labels if labels is not None else np.full(len(parts), default_label, dtype=np.int8)
    for name, arr in res.items():
        mm = np.load(out / f"{name}.npy", mmap_mode="r+")
        mm[start:stop] = arr
        mm.flush()
        del mm
    (out / "chunks").mkdir(exist_ok=True)
    (out / "chunks" / f"{start:08d}_{stop:08d}.ok").touch()
    return start, stop, time.time() - t0


def preprocess(
    sample: str,
    src: Path | None = None,
    out_root: Path | None = None,
    chunk: int = 20000,
    workers: int = 2,
    limit: int | None = None,
    n_max: int = N_MAX,
) -> Path:
    src = Path(src) if src else raw_dir() / RAW_FILES[sample]
    out_dir = (Path(out_root) if out_root else processed_dir()) / sample
    out_dir.mkdir(parents=True, exist_ok=True)
    n = n_rows(src)
    if limit:
        n = min(n, limit)
    _allocate(out_dir, n, n_max)
    default_label = 1 if sample == "qqq" else 0
    jobs = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        if (out_dir / "chunks" / f"{s:08d}_{e:08d}.ok").exists():
            continue
        jobs.append((str(src), str(out_dir), s, e, n_max, default_label))
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
        "seconds": time.time() - t0,
        **run_metadata(),
    }
    save_json(meta, out_dir / "meta.json")
    print(f"[{sample}] finished in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)
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
    a = p.parse_args(argv)
    preprocess(a.sample, a.src, a.out_root, a.chunk, a.workers, a.limit, a.n_max)
    return 0


if __name__ == "__main__":
    sys.exit(main())
