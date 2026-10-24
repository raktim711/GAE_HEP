"""Write a small *synthetic* processed dataset with the same layout as the real one.

Used for end-to-end tests of the orchestration (train -> score -> evaluate -> report) without the LHCO
files.  Background jets are 1-prong blobs, 2-prong signal jets have two sub-clusters and masses near
(500, 100) GeV, 3-prong signal jets three sub-clusters; signal m_jj peaks at 3.5 TeV.

    python -m gaehep.data.fake --out /tmp/fake_lhco --n-bkg 20000 --n-sig 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gaehep.data.splits import check_disjoint, make_splits


def _jets(rng: np.random.Generator, n: int, n_max: int, prongs: int, masses: tuple[float, float] | None):
    const = np.zeros((n, 2, n_max, 3), dtype=np.float32)
    n_const = rng.integers(15, 2 * n_max, size=(n, 2)).astype(np.int32)
    eta = rng.uniform(-1.5, 1.5, (n, 2))
    phi = rng.uniform(-np.pi, np.pi, (n, 2))
    phi[:, 1] = (phi[:, 0] + np.pi + rng.normal(0, 0.1, n) + np.pi) % (2 * np.pi) - np.pi
    pt = rng.normal(1500, 150, (n, 2)).clip(1200)
    for i in range(n):
        for k in range(2):
            m = min(n_const[i, k], n_max)
            axes = rng.normal(0, 0.25 if prongs > 1 else 0.05, size=(prongs, 2))
            d = axes[rng.integers(0, prongs, m)] + rng.normal(0, 0.06, (m, 2))
            frac = np.sort(rng.dirichlet(np.ones(m) * 0.5))[::-1]
            const[i, k, :m, 0] = frac * pt[i, k]
            const[i, k, :m, 1] = eta[i, k] + d[:, 0]
            const[i, k, :m, 2] = (phi[i, k] + d[:, 1] + np.pi) % (2 * np.pi) - np.pi
    if masses is None:
        mj = rng.exponential(80, (n, 2)) + 10
    else:
        mj = np.stack([rng.normal(masses[0], 30, n), rng.normal(masses[1], 15, n)], axis=1).clip(5)
    e = np.sqrt((pt * np.cosh(eta)) ** 2 + mj**2)
    return const, n_const, pt, eta, phi, mj, e


def _tau(rng: np.random.Generator, n: int, prongs: int) -> dict[str, np.ndarray]:
    t1 = rng.uniform(0.2, 0.6, (n, 2))
    r21 = rng.uniform(0.15, 0.45, (n, 2)) if prongs >= 2 else rng.uniform(0.4, 0.9, (n, 2))
    r32 = rng.uniform(0.3, 0.6, (n, 2)) if prongs >= 3 else rng.uniform(0.5, 0.9, (n, 2))
    return {"tau1": t1, "tau2": t1 * r21, "tau3": t1 * r21 * r32}


def write_sample(out: Path, parts: list[tuple[int, int, tuple | None, int]], n_max: int, rng: np.random.Generator) -> None:
    """parts: list of (n_events, prongs, masses, label)."""
    cols: dict[str, list[np.ndarray]] = {}
    consts = []
    for n, prongs, masses, label in parts:
        const, n_const, pt, eta, phi, mj, e = _jets(rng, n, n_max, prongs, masses)
        consts.append(const)
        mjj = rng.normal(3500, 120, n) if label else 2400 + rng.exponential(600, n)
        tau = _tau(rng, n, prongs)
        add = {"pt": pt, "eta": eta, "phi": phi, "m": mj, "e": e, "n_const": n_const, "mjj": mjj,
               "label": np.full(n, label, np.int8), "n_jets": np.full(n, 2, np.int16), **tau}
        for k, v in add.items():
            cols.setdefault(k, []).append(v)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "constituents.npy", np.concatenate(consts))
    table = {k: np.concatenate(v) for k, v in cols.items()}
    table = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in table.items()}
    n_tot = len(table["mjj"])
    table["matched"] = np.ones(n_tot, bool)
    table["swapped"] = np.zeros(n_tot, bool)
    np.savez(out / "events.npz", **table)


def make_fake(root: Path, n_bkg: int = 20000, n_sig: int = 3000, n_max: int = 50, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    proc = Path(root) / "processed"
    write_sample(proc / "rnd", [(n_bkg, 1, None, 0), (n_sig, 2, (500.0, 100.0), 1)], n_max, rng)
    write_sample(proc / "qqq", [(n_sig, 3, (500.0, 100.0), 1)], n_max, rng)
    rnd = np.load(proc / "rnd" / "events.npz")
    qqq = np.load(proc / "qqq" / "events.npz")
    splits = make_splits(rnd["n_jets"], rnd["label"], qqq["n_jets"], seed=seed)
    check_disjoint(splits)
    np.savez(proc / "splits.npz", **splits)
    return proc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-bkg", type=int, default=20000)
    p.add_argument("--n-sig", type=int, default=3000)
    p.add_argument("--n-max", type=int, default=50)
    a = p.parse_args(argv)
    print(make_fake(a.out, a.n_bkg, a.n_sig, a.n_max))
    return 0


if __name__ == "__main__":
    sys.exit(main())
