"""Deterministic event splits (saved to ``processed/splits.npz``).

* background (``rnd``, label 0)  -> ``bkg_train`` / ``bkg_val`` / ``bkg_test``  = 60 / 10 / 30 %
* 2-prong signal (``rnd``, label 1) -> ``sig_train`` / ``sig_val`` / ``sig_test`` = 40 / 10 / 50 %
* 3-prong signal (``qqq``)         -> ``sig3_test`` = 100 %
* black boxes (``bb1``, ``bb2``, ``bb3``; test only, a separate file ``splits_blackbox.npz`` so that the R&D
  ``splits.npz`` and its content hash stay as they are) -> ``bb<i>_test`` = every event with at least two jets
  and ``bb<i>_test_sub`` = 200 000 background events plus every signal event

Only events with at least two jets enter any split.  Unsupervised models are *trained* on ``bkg_train``
only, with early stopping on the ``bkg_val`` loss.  Signal enters elsewhere: ``sig_train``/``sig_val`` train
the supervised reference classifier, ``sig_val`` drives the design choices of the development protocol
(``gaehep dev``: latent size, kernels, training budget, event aggregation), and ``sig_train`` is the
injection pool of the weak-supervision study.  ``sig_test`` and ``sig3_test`` are used for the reported
test numbers.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import sys
from pathlib import Path

import numpy as np

from gaehep.config import BLACKBOX_SAMPLES, processed_dir, run_metadata, save_json

BKG_FRACTIONS = (0.6, 0.1, 0.3)
SIG_FRACTIONS = (0.4, 0.1, 0.5)

SEED = 20201
SUB_N_BKG = 200_000  # background events of a black box's ``_test_sub`` split
TRAIN_FRACTION = 0.9  # train-on-data scenario: share of the training events (the rest validates)
BLACKBOX_SPLITS = "splits_blackbox.npz"
# content hash of the R&D splits (docs/STUDY_PROTOCOL.md, section 5); they must stay bit-identical
PROTOCOL_SPLITS_SHA256 = "5217cca54367efda6b33a1626da8cdb21e87ef7b6fd6b2abfa73d8ac510f7af2"


def _split(
    idx: np.ndarray, fractions: tuple[float, float, float], rng: np.random.Generator
) -> list[np.ndarray]:
    idx = idx.copy()
    rng.shuffle(idx)
    n = len(idx)
    n_a = round(fractions[0] * n)
    n_b = round(fractions[1] * n)
    parts = [idx[:n_a], idx[n_a : n_a + n_b], idx[n_a + n_b :]]
    return [np.sort(p) for p in parts]


def make_splits(
    n_jets_rnd: np.ndarray,
    labels_rnd: np.ndarray,
    n_jets_qqq: np.ndarray | None,
    seed: int = 20201,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ok = n_jets_rnd >= 2
    bkg = np.flatnonzero(ok & (labels_rnd == 0))
    sig = np.flatnonzero(ok & (labels_rnd == 1))
    out: dict[str, np.ndarray] = {}
    out["bkg_train"], out["bkg_val"], out["bkg_test"] = _split(bkg, BKG_FRACTIONS, rng)
    out["sig_train"], out["sig_val"], out["sig_test"] = _split(sig, SIG_FRACTIONS, rng)
    if n_jets_qqq is not None:
        out["sig3_test"] = np.flatnonzero(n_jets_qqq >= 2)
    return out


def check_disjoint(splits: dict[str, np.ndarray]) -> None:
    """Raise if any rnd split overlaps another."""
    keys = [k for k in splits if k != "sig3_test"]
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            inter = np.intersect1d(splits[a], splits[b])
            if len(inter):
                raise AssertionError(f"splits {a} and {b} overlap ({len(inter)} events)")


def make_blackbox_splits(
    sample: str,
    n_jets: np.ndarray,
    labels: np.ndarray,
    seed: int = SEED,
    n_bkg_sub: int = SUB_N_BKG,
) -> dict[str, np.ndarray]:
    """Test splits of black box ``sample``: ``<sample>_test`` (every event with >= 2 jets) and
    ``<sample>_test_sub`` (``n_bkg_sub`` background events drawn without replacement with
    ``default_rng(seed).choice`` plus every signal event; all the background if there are fewer).  The
    labels (0 = background, any other value = signal) only decide which events the subsample keeps."""
    ok = np.asarray(n_jets) >= 2
    labels = np.asarray(labels)
    bkg = np.flatnonzero(ok & (labels == 0))
    sig = np.flatnonzero(ok & (labels != 0))
    if len(bkg) > n_bkg_sub:
        bkg = np.random.default_rng(seed).choice(bkg, n_bkg_sub, replace=False)
    return {f"{sample}_test": np.flatnonzero(ok), f"{sample}_test_sub": np.sort(np.concatenate([bkg, sig]))}


def train_val_split(
    events: np.ndarray, seed: int = SEED, train_fraction: float = TRAIN_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded 90/10 split (train, validation) of an event index set, e.g. all BB1 events with >= 2 jets
    (``train_sample: bb1``).  It sees the indices only: labels never enter."""
    idx = np.array(events, dtype=np.int64)
    np.random.default_rng(seed).shuffle(idx)
    n_train = round(train_fraction * len(idx))
    return np.sort(idx[:n_train]), np.sort(idx[n_train:])


def load_splits(root: Path | None = None) -> dict[str, np.ndarray]:
    """The R&D splits (``splits.npz``) together with the black-box splits (``splits_blackbox.npz``) of
    whichever of the two files exist."""
    root = Path(root) if root else processed_dir()
    out: dict[str, np.ndarray] = {}
    for name in ("splits.npz", BLACKBOX_SPLITS):
        if (root / name).exists():
            with np.load(root / name) as f:
                out.update({k: f[k] for k in f.files})
    if not out:
        raise FileNotFoundError(f"no {root / 'splits.npz'} (run `gaehep splits`)")
    return out


@functools.lru_cache(maxsize=8)
def _hash_file(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with np.load(path) as f:
        for k in sorted(f.files):
            a = f[k]
            h.update(k.encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def content_hash(path: Path | str | None = None) -> str:
    """sha256 over the sorted split names of ``splits.npz``, each followed by its dtype string and its raw
    bytes (docs/STUDY_PROTOCOL.md, section 5).  ``path``: the file, or the processed directory holding it
    (default: ``processed_dir()``).  Raises FileNotFoundError if the file does not exist."""
    path = Path(path) if path else processed_dir()
    if path.is_dir():
        path = path / "splits.npz"
    st = path.stat()
    return _hash_file(str(path.resolve()), st.st_mtime_ns, st.st_size)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proc-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--sample",
        nargs="+",
        choices=BLACKBOX_SAMPLES,
        default=None,
        help="make the test splits of these black boxes (splits_blackbox.npz) instead of the R&D splits",
    )
    p.add_argument("--n-bkg-sub", type=int, default=SUB_N_BKG, help="background events of a _test_sub split")
    p.add_argument(
        "--check",
        action="store_true",
        help="print the content hash of splits.npz and exit 1 unless it equals the protocol's",
    )
    a = p.parse_args(argv)
    root = Path(a.proc_root) if a.proc_root else processed_dir()
    if a.check:
        try:
            h = content_hash(root)
        except FileNotFoundError as err:
            print(f"splits hash: {err}")
            return 1
        print(f"splits_sha256 {h}")
        if h != PROTOCOL_SPLITS_SHA256:
            print(f"DIFFERS from the protocol's {PROTOCOL_SPLITS_SHA256}")
            return 1
        print("equals the protocol's value")
        return 0
    if a.sample:
        path = root / BLACKBOX_SPLITS
        splits: dict[str, np.ndarray] = {}
        if path.exists():  # keep the boxes made earlier
            with np.load(path) as f:
                splits = {k: f[k] for k in f.files}
        for s_ in a.sample:
            ev = np.load(root / s_ / "events.npz")
            splits.update(make_blackbox_splits(s_, ev["n_jets"], ev["label"], a.seed, a.n_bkg_sub))
        np.savez(path, **splits)
        summary = {k: len(v) for k, v in splits.items()}
        save_json({"seed": a.seed, "sizes": summary, **run_metadata()}, root / "splits_blackbox.json")
        print(summary)
        return 0
    rnd = np.load(root / "rnd" / "events.npz")
    qqq_path = root / "qqq" / "events.npz"
    n_jets_qqq = np.load(qqq_path)["n_jets"] if qqq_path.exists() else None
    splits = make_splits(rnd["n_jets"], rnd["label"], n_jets_qqq, seed=a.seed)
    check_disjoint(splits)
    np.savez(root / "splits.npz", **splits)
    summary = {k: len(v) for k, v in splits.items()}
    save_json({"seed": a.seed, "sizes": summary, **run_metadata()}, root / "splits.json")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
