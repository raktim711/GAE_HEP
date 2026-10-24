"""Event/jet tables from the clustered output, validated against the official LHCO feature files.

The official ``*.features.h5`` files contain, for the two leading anti-kT R=1.0 jets,
``px, py, pz, m, tau1, tau2, tau3`` (columns ``pxj1 ... tau3j2``) plus ``label`` for the 2-prong file.
We do **not** assume their jet ordering: for every event the official (j1, j2) pair is matched to our
pT-ordered jets either directly or swapped, whichever agrees.  Only matched events receive tau values.

Output: ``processed/<sample>/events.npz`` with per-event arrays

``mjj, label, n_jets, matched, swapped`` and per-jet (n, 2) arrays ``pt, eta, phi, m, e, n_const,
tau1, tau2, tau3`` (NaN where undefined), plus ``validation.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gaehep.config import processed_dir, raw_dir, run_metadata, save_json

FEATURE_FILES = {
    "rnd": "events_anomalydetection_v2.features.h5",
    "qqq": "events_anomalydetection_Z_XY_qqq.features.h5",
}


def jet_observables(p4: np.ndarray) -> dict[str, np.ndarray]:
    """(…, 4) array of (px, py, pz, E) -> pt, eta, phi, m, e (NaN propagates for missing jets)."""
    px, py, pz, e = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    pt = np.hypot(px, py)
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = np.arcsinh(pz / pt)
        m = np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0.0, None))
    phi = np.arctan2(py, px)
    return {"pt": pt, "eta": eta, "phi": phi, "m": m, "e": e}


def dijet_mass(p4: np.ndarray) -> np.ndarray:
    """Invariant mass of the sum of the two leading jets; p4 shape (n, 2, 4)."""
    s = p4[:, 0, :] + p4[:, 1, :]
    return np.sqrt(np.clip(s[:, 3] ** 2 - s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2, 0.0, None))


def _rel_dev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Max over components of |a-b| / max(|b|, 1 GeV); shape (n,)."""
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0), axis=-1)


def match_features(ours_p4: np.ndarray, feat: dict[str, np.ndarray], tol: float = 1e-3) -> dict[str, np.ndarray]:
    """Match official (j1, j2) to our pT-ordered jets (direct or swapped).

    ``feat`` holds 1D arrays pxj1, pyj1, pzj1, mj1, pxj2, pyj2, pzj2, mj2 (and tau*).
    Returns ``matched`` (bool), ``swapped`` (bool) and the deviations of the chosen assignment.
    """
    obs = jet_observables(ours_p4)
    ours = [np.stack([ours_p4[:, k, 0], ours_p4[:, k, 1], ours_p4[:, k, 2], obs["m"][:, k]], axis=-1) for k in (0, 1)]
    off = [np.stack([feat[f"px{j}"], feat[f"py{j}"], feat[f"pz{j}"], feat[f"m{j}"]], axis=-1) for j in ("j1", "j2")]
    dev_direct = np.maximum(_rel_dev(ours[0], off[0]), _rel_dev(ours[1], off[1]))
    dev_swap = np.maximum(_rel_dev(ours[0], off[1]), _rel_dev(ours[1], off[0]))
    dev_direct = np.where(np.isnan(dev_direct), np.inf, dev_direct)
    dev_swap = np.where(np.isnan(dev_swap), np.inf, dev_swap)
    swapped = dev_swap < dev_direct
    dev = np.where(swapped, dev_swap, dev_direct)
    return {"matched": dev < tol, "swapped": swapped & (dev < tol), "dev": dev}


def read_features(path: str | Path) -> dict[str, np.ndarray]:
    import pandas as pd

    df = pd.read_hdf(path)
    return {c: df[c].to_numpy() for c in df.columns}


def build_tables(sample: str, proc_root: Path | None = None, feat_path: Path | None = None, tol: float = 1e-3) -> dict:
    root = (Path(proc_root) if proc_root else processed_dir()) / sample
    p4 = np.load(root / "jets_p4.npy")
    n = len(p4)
    obs = jet_observables(p4)
    out: dict[str, np.ndarray] = {k: v.astype(np.float32) for k, v in obs.items()}
    out["n_const"] = np.load(root / "n_const.npy")
    out["n_jets"] = np.load(root / "n_jets.npy")
    out["label"] = np.load(root / "labels.npy")
    out["mjj"] = dijet_mass(p4).astype(np.float32)

    feat_path = Path(feat_path) if feat_path else raw_dir() / FEATURE_FILES[sample]
    summary: dict = {"sample": sample, "n_events": n}
    taus = {f"tau{i}": np.full((n, 2), np.nan, dtype=np.float32) for i in (1, 2, 3)}
    if feat_path.exists():
        feat = read_features(feat_path)
        nf = len(next(iter(feat.values())))
        if nf < n:
            raise ValueError(f"feature file has {nf} rows < {n} events")
        feat = {k: v[:n] for k, v in feat.items()}
        m = match_features(p4, feat, tol=tol)
        sw = m["swapped"]
        for i in (1, 2, 3):
            a, b = feat[f"tau{i}j1"].astype(np.float32), feat[f"tau{i}j2"].astype(np.float32)
            t0 = np.where(sw, b, a)
            t1 = np.where(sw, a, b)
            taus[f"tau{i}"][:, 0] = np.where(m["matched"], t0, np.nan)
            taus[f"tau{i}"][:, 1] = np.where(m["matched"], t1, np.nan)
        out["matched"], out["swapped"] = m["matched"], sw
        if "label" in feat:
            agree = float(np.mean(feat["label"].astype(np.int8) == out["label"]))
            summary["label_agreement"] = agree
        finite = np.isfinite(m["dev"])
        summary.update(
            {
                "feature_file": str(feat_path),
                "match_tolerance": tol,
                "matched_fraction": float(np.mean(m["matched"])),
                "swapped_fraction_of_matched": float(np.mean(sw[m["matched"]])) if m["matched"].any() else None,
                "dev_quantiles": {
                    q: float(np.quantile(m["dev"][finite], q)) for q in (0.5, 0.9, 0.99, 0.999)
                }
                if finite.any()
                else None,
                "n_unmatched": int(np.sum(~m["matched"])),
            }
        )
    else:
        out["matched"] = np.zeros(n, dtype=bool)
        out["swapped"] = np.zeros(n, dtype=bool)
        summary["feature_file"] = None
    out.update(taus)
    summary.update(
        {
            "frac_two_jets": float(np.mean(out["n_jets"] >= 2)),
            "n_signal": int(np.sum(out["label"] == 1)),
            "n_background": int(np.sum(out["label"] == 0)),
            "mjj_quantiles": {q: float(np.nanquantile(out["mjj"], q)) for q in (0.01, 0.5, 0.99)},
            **run_metadata(),
        }
    )
    np.savez(root / "events.npz", **out)
    save_json(summary, root / "validation.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", choices=list(FEATURE_FILES), required=True)
    p.add_argument("--proc-root", type=Path, default=None)
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--tol", type=float, default=1e-3)
    a = p.parse_args(argv)
    s = build_tables(a.sample, a.proc_root, a.features, a.tol)
    import json

    print(json.dumps(s, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
