"""Sideband-fit bump hunt in m_jj using an (unsupervised) anomaly score as the selection.

Procedure per signal-injection level and score-cut efficiency:

1. "data" = background test events + ``n_inj`` signal events drawn from the signal test pool;
2. keep the events whose score exceeds the threshold that passes a fraction ``eff`` of the *sideband*
   events (the threshold never looks at the signal region, SR = 3.3-3.7 TeV);
3. fit the standard 4-parameter dijet function  f(x) = p0 (1-x)^p1 / x^(p2 + p3 ln x),  x = m_jj / 13 TeV,
   to the binned sideband spectrum with a Poisson likelihood;
4. compare the observed SR count n with the fitted expectation b: local significance
   Z = sign(n-b) sqrt(2 (n ln(n/b) - (n-b))).

A score that sculpts m_jj shows up as a non-zero Z at zero injected signal.  The fit range (2.7-5.0 TeV)
starts above the trigger turn-on; with a 2.5 TeV start the 4-parameter function under-describes the
background (Z ~ -3.5 at zero signal without any cut).  The significance ignores the fit-parameter
uncertainty, so it is optimistic at large statistics (a demonstration, not a full statistical analysis).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from gaehep.config import SR_MJJ

SQRT_S = 13000.0
DEFAULT_EDGES = np.arange(
    2700.0, 5000.0 + 1e-6, 100.0
)  # starts above the trigger turn-on (m_jj peak ~2.6 TeV)


def dijet_fn(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = p
    return np.exp(p0) * (1.0 - x) ** p1 / x ** (p2 + p3 * np.log(x))


def _bin_expectation(edges: np.ndarray, p: np.ndarray, n_sub: int = 5) -> np.ndarray:
    """Integral of the fit function over each bin (Simpson-like sub-sampling)."""
    lo, hi = edges[:-1], edges[1:]
    t = (np.arange(n_sub) + 0.5) / n_sub
    xs = (lo[:, None] + (hi - lo)[:, None] * t[None, :]) / SQRT_S
    return dijet_fn(xs, p).mean(1) * (hi - lo) / SQRT_S


@dataclass
class FitResult:
    params: np.ndarray
    ok: bool
    nll: float


def fit_sidebands(counts: np.ndarray, edges: np.ndarray, sb_mask: np.ndarray) -> FitResult:
    """Poisson maximum-likelihood fit of the dijet function to the sideband bins."""
    c = counts[sb_mask].astype(np.float64)
    e_sb = sb_mask

    def nll(p: np.ndarray) -> float:
        mu = _bin_expectation(edges, p)[e_sb]
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
            return 1e30
        return float(np.sum(mu - c * np.log(mu)))

    x_mid = 0.5 * (edges[:-1] + edges[1:]) / SQRT_S
    # start: power-law fit through log counts of the sideband bins
    with np.errstate(divide="ignore"):
        y = np.log(np.maximum(c, 0.5) / ((edges[1:] - edges[:-1])[e_sb] / SQRT_S))
    a = np.polyfit(-np.log(x_mid[e_sb]), y, 1)
    best = None
    for p1 in (1.0, 5.0, 10.0):
        p_start = np.array([a[1], p1, a[0], 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(
                nll, p_start, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6}
            )
            r = minimize(nll, r.x, method="Powell", options={"maxiter": 20000})
        if best is None or r.fun < best.fun:
            best = r
    return FitResult(params=best.x, ok=bool(best.success), nll=float(best.fun))


def poisson_z(n: float, b: float) -> float:
    if b <= 0:
        return float("nan")
    if n <= 0:
        return -float(np.sqrt(2 * b))
    t = 2 * (n * np.log(n / b) - (n - b))
    return float(np.sign(n - b) * np.sqrt(max(t, 0.0)))


def search(
    mjj: np.ndarray,
    score: np.ndarray,
    eff: float,
    edges: np.ndarray = DEFAULT_EDGES,
    sr: tuple[float, float] = SR_MJJ,
) -> dict[str, float]:
    """One bump hunt on a given dataset (mjj, score) with a sideband-defined score threshold."""
    in_sr = (mjj >= sr[0]) & (mjj < sr[1])
    in_range = (mjj >= edges[0]) & (mjj < edges[-1])
    sb = in_range & ~in_sr
    thr = -np.inf if eff >= 1 else np.quantile(score[sb], 1 - eff)
    sel = score >= thr
    counts, _ = np.histogram(mjj[sel], bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sr_bins = (centres > sr[0]) & (centres < sr[1])
    fit = fit_sidebands(counts, edges, ~sr_bins)
    b = float(_bin_expectation(edges, fit.params)[sr_bins].sum())
    n = float(counts[sr_bins].sum())
    return {"eff": eff, "n_sr": n, "b_sr": b, "z": poisson_z(n, b), "fit_ok": fit.ok, "threshold": float(thr)}


def run_bumphunt(
    mjj_bkg: np.ndarray,
    score_bkg: np.ndarray,
    mjj_sig: np.ndarray,
    score_sig: np.ndarray,
    n_inj: Sequence[int] = (0, 150, 300, 600, 1200),
    effs: Sequence[float] = (1.0, 0.1, 0.02, 0.005),
    seeds: Sequence[int] = (0, 1, 2),
) -> list[dict[str, float]]:
    rows = []
    for n in n_inj:
        for seed in seeds:
            rng = np.random.default_rng([seed, n])
            pick = rng.choice(len(mjj_sig), size=n, replace=False) if n else np.zeros(0, dtype=int)
            mjj = np.r_[mjj_bkg, mjj_sig[pick]]
            score = np.r_[score_bkg, score_sig[pick]]
            n_sig_sr = int(np.sum((mjj_sig[pick] >= SR_MJJ[0]) & (mjj_sig[pick] < SR_MJJ[1])))
            for eff in effs:
                r = search(mjj, score, eff)
                r.update({"n_inj": n, "n_sig_sr": n_sig_sr, "seed": seed})
                rows.append(r)
    return rows
