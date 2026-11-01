"""N-subjettiness tau_1..tau_3 computed from the stored constituents (docs/STUDY_PROTOCOL.md, Deviations D3).

The black boxes have no LHCO feature files, so their jet tables had no tau.  This module computes, for every jet,

    tau_N = sum_i pT_i * min_k DeltaR(i, axis_k)        (beta = 1, unnormalised, GeV - the convention of the
                                                         official R&D feature files)

with DeltaR in (rapidity, phi) and axes that minimise tau_N: exclusive-kT axes (fastjet, E-scheme) as the start,
then Weiszfeld updates (the exact minimiser of a pT-weighted sum of distances for a fixed assignment) with the
constituents re-assigned to their nearest axis at every step, until tau_N stops decreasing.  The constituents are
the stored ones (the 100 leading by pT), treated as massless.

On R&D jets this reproduces the official values to better than 1e-3 (relative) for most jets with fewer than 100
stored constituents; jets truncated at 100 constituents and jets where the minimisation ends in a different
local minimum differ more (``gaehep tau --check`` reports the agreement).  The study therefore uses the computed
values for every sample, R&D included, wherever tau is compared across samples (protocol Deviations D3).

``gaehep tau --sample S [--workers N]`` writes ``<processed>/<S>/tau_computed.npz`` with arrays tau1c, tau2c,
tau3c of shape (n_events, 2) (NaN where a jet has fewer than N constituents or does not exist); the jet table
then exposes them as the keys ``tau1c``-``tau3c`` (``JetStore``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

N_MAX = 3
MAX_ITER = 300
TOL = 1e-9  # stop when no jet's tau_N decreases by more than this fraction
FILE = "tau_computed.npz"
CHUNK_EVENTS = 20_000


def _wrap(dphi: np.ndarray) -> np.ndarray:
    return np.mod(dphi + np.pi, 2 * np.pi) - np.pi


def _kt_axes(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, mask: np.ndarray, n: int):
    """(rapidity, phi) of the n exclusive-kT subjets of every jet (all jets must have >= n constituents)."""
    import awkward as ak
    import fastjet

    px, py, pz, e = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    arr = ak.unflatten(
        ak.zip({"px": px[mask], "py": py[mask], "pz": pz[mask], "E": e[mask]}, with_name="Momentum4D"),
        mask.sum(1),
    )
    cs = fastjet.ClusterSequence(arr, fastjet.JetDefinition(fastjet.kt_algorithm, 1000.0))
    sub = cs.exclusive_jets(n_jets=n)
    apx, apy, apz, ae = (np.asarray(ak.to_numpy(sub[k]), dtype=np.float64) for k in ("px", "py", "pz", "E"))
    with np.errstate(divide="ignore", invalid="ignore"):
        rap = 0.5 * np.log((ae + apz) / (ae - apz))
    return np.nan_to_num(rap), np.arctan2(apy, apx)


def _tau(pt, eta, phi, ar, ap):
    d = np.hypot(eta[:, :, None] - ar[:, None, :], _wrap(phi[:, :, None] - ap[:, None, :]))
    return (pt * d.min(2)).sum(1), d


def nsubjettiness(const: np.ndarray, n_max: int = N_MAX, max_iter: int = MAX_ITER) -> np.ndarray:
    """tau_1..tau_n_max (beta = 1, unnormalised) of jets ``const`` (n_jets, N, 3) = (pT, eta, phi), zero-padded.

    Returns (n_jets, n_max) float64; NaN for jets with fewer than N real constituents (and for empty jets)."""
    c = np.asarray(const, dtype=np.float64)
    pt, eta, phi = c[..., 0], c[..., 1], c[..., 2]
    mask = pt > 0
    counts = mask.sum(1)
    out = np.full((len(c), n_max), np.nan)
    for n in range(1, n_max + 1):
        ok = counts >= n
        if not ok.any():
            continue
        p, e, f, m = pt[ok], eta[ok], phi[ok], mask[ok]
        ar, ap = _kt_axes(p, e, f, m, n)
        best, d = _tau(p, e, f, ar, ap)
        act = np.arange(len(p))  # jets whose tau_N still decreases
        for _ in range(max_iter):
            pa, ea, fa, da, ra, pha = p[act], e[act], f[act], d[act], ar[act], ap[act]
            assign = da.argmin(2)
            nr, nphi = ra.copy(), pha.copy()
            for k in range(n):
                w = pa * (assign == k) / np.maximum(da[:, :, k], 1e-12)
                sw = w.sum(1)
                has = sw > 0
                safe = np.where(has, sw, 1.0)
                nr[has, k] = ((w * ea).sum(1) / safe)[has]
                nphi[has, k] = (pha[:, k] + (w * _wrap(fa - pha[:, k : k + 1])).sum(1) / safe)[has]
            tau_new, d_new = _tau(pa, ea, fa, nr, nphi)
            better = tau_new < best[act]
            gain = np.where(better, (best[act] - tau_new) / np.maximum(best[act], 1e-300), 0.0)
            upd = act[better]
            ar[upd], ap[upd] = nr[better], nphi[better]
            best[upd] = tau_new[better]
            d[upd] = d_new[better]
            act = act[gain >= TOL]
            if not len(act):
                break
        out[ok, n - 1] = best
    return out


def _chunk(args: tuple[str, int, int]) -> tuple[int, np.ndarray]:
    path, start, stop = args
    const = np.load(path, mmap_mode="r")
    c = np.asarray(const[start:stop], dtype=np.float64)  # (n, 2, N, 3)
    t = nsubjettiness(c.reshape(-1, c.shape[2], 3)).reshape(len(c), 2, N_MAX)
    return start, t


def compute_sample(sample_dir: Path, workers: int = 1, log=print) -> Path:
    """tau1c-tau3c for every event of a processed sample; resumable per chunk (``_tau_chunks/``)."""
    from multiprocessing import Pool

    sample_dir = Path(sample_dir)
    const_path = sample_dir / "constituents.npy"
    n = len(np.load(const_path, mmap_mode="r"))
    parts = sample_dir / "_tau_chunks"
    parts.mkdir(exist_ok=True)
    todo = [
        (str(const_path), s, min(s + CHUNK_EVENTS, n))
        for s in range(0, n, CHUNK_EVENTS)
        if not (parts / f"{s:09d}.npy").exists()
    ]
    t0 = time.time()
    log(f"[tau] {sample_dir.name}: {n} events, {len(todo)} chunks to do, {workers} workers")
    with Pool(workers) as pool:
        for i, (start, t) in enumerate(pool.imap_unordered(_chunk, todo), 1):
            np.save(parts / f"{start:09d}.tmp.npy", t)
            (parts / f"{start:09d}.tmp.npy").replace(parts / f"{start:09d}.npy")
            if i % 10 == 0 or i == len(todo):
                log(f"[tau] {sample_dir.name}: {i}/{len(todo)} chunks, {(time.time() - t0) / 60:.1f} min")
    tau = np.concatenate([np.load(parts / f"{s:09d}.npy") for s in range(0, n, CHUNK_EVENTS)])
    out = sample_dir / FILE
    np.savez(out, **{f"tau{i + 1}c": tau[..., i].astype(np.float32) for i in range(N_MAX)})
    for p in parts.glob("*.npy"):
        p.unlink()
    parts.rmdir()
    log(f"[tau] wrote {out}")
    return out


def check_against_official(sample_dir: Path, n_events: int = 5000, seed: int = 0) -> dict[str, float]:
    """Agreement of the computed tau with the official feature-file values on matched events of an R&D sample."""
    sample_dir = Path(sample_dir)
    with np.load(sample_dir / "events.npz") as ev:
        matched = np.flatnonzero(ev["matched"])
        rng = np.random.default_rng(seed)
        e = np.sort(rng.choice(matched, min(n_events, len(matched)), replace=False))
        ref = np.stack([ev[f"tau{i}"][e].reshape(-1) for i in (1, 2, 3)], 1).astype(np.float64)
        n_const = ev["n_const"][e].reshape(-1)
    const = np.load(sample_dir / "constituents.npy", mmap_mode="r")
    c = np.asarray(const[e], dtype=np.float64)
    mine = nsubjettiness(c.reshape(-1, c.shape[2], 3))
    good = np.isfinite(ref).all(1) & np.isfinite(mine).all(1) & (ref > 0).all(1)
    rel = np.abs(mine[good] - ref[good]) / ref[good]
    full = n_const[good] < c.shape[2]
    r21 = (mine[good, 1] / mine[good, 0], ref[good, 1] / ref[good, 0])
    r32 = (mine[good, 2] / mine[good, 1], ref[good, 2] / ref[good, 1])
    return {
        "n_jets": int(good.sum()),
        **{f"tau{i + 1}_frac_rel_lt_1e-3": float(np.mean(rel[:, i] < 1e-3)) for i in range(3)},
        **{f"tau{i + 1}_frac_rel_lt_1e-2": float(np.mean(rel[:, i] < 1e-2)) for i in range(3)},
        "tau1_frac_rel_lt_1e-3_untruncated": float(np.mean(rel[full, 0] < 1e-3)),
        "frac_truncated": float(np.mean(~full)),
        "tau21_corr": float(np.corrcoef(*r21)[0, 1]),
        "tau21_median_abs_diff": float(np.median(np.abs(r21[0] - r21[1]))),
        "tau32_corr": float(np.corrcoef(*r32)[0, 1]),
        "tau32_median_abs_diff": float(np.median(np.abs(r32[0] - r32[1]))),
    }


def main(argv: list[str] | None = None) -> int:
    from gaehep.config import processed_dir, save_json

    p = argparse.ArgumentParser(
        prog="gaehep tau", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sample", required=True, help="processed sample: rnd, qqq, bb1, bb2, bb3")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--check", action="store_true", help="compare with the official values (R&D samples) and exit"
    )
    p.add_argument("--n-check", type=int, default=5000, help="events for --check")
    a = p.parse_args(argv)
    d = processed_dir() / a.sample
    if a.check:
        res = check_against_official(d, a.n_check)
        save_json(res, d / "tau_check.json")
        for k, v in res.items():
            print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
        return 0
    compute_sample(d, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
