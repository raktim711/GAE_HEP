"""Deterministic event splits (saved to ``processed/splits.npz``).

* background (``rnd``, label 0)  -> ``bkg_train`` / ``bkg_val`` / ``bkg_test``  = 60 / 10 / 30 %
* 2-prong signal (``rnd``, label 1) -> ``sig_train`` / ``sig_val`` / ``sig_test`` = 40 / 10 / 50 %
* 3-prong signal (``qqq``)         -> ``sig3_test`` = 100 %

Only events with at least two jets enter any split.  Unsupervised models are *trained* on ``bkg_train``
only, with early stopping on the ``bkg_val`` loss.  Signal enters elsewhere: ``sig_train``/``sig_val`` train
the supervised reference classifier, ``sig_val`` drives the design choices of the development protocol
(``gaehep dev``: latent size, kernels, training budget, event aggregation), and ``sig_train`` is the
injection pool of the weak-supervision study.  ``sig_test`` and ``sig3_test`` are used for the reported
test numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gaehep.config import processed_dir, run_metadata, save_json

BKG_FRACTIONS = (0.6, 0.1, 0.3)
SIG_FRACTIONS = (0.4, 0.1, 0.5)


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


def load_splits(root: Path | None = None) -> dict[str, np.ndarray]:
    with np.load((Path(root) if root else processed_dir()) / "splits.npz") as f:
        return {k: f[k] for k in f.files}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proc-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20201)
    a = p.parse_args(argv)
    root = Path(a.proc_root) if a.proc_root else processed_dir()
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
