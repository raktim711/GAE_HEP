"""How much does a cut on the anomaly score distort (sculpt) background distributions?

* Jensen-Shannon divergence (base 2, in [0, 1]) between the background distribution of an observable
  (m_jj, jet mass) before and after keeping the top ``eff`` fraction of background by score.
* Distance correlation between score and observable (0 iff independent, for large samples).
"""

from __future__ import annotations

import numpy as np


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two histograms (normalised internally)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def sculpting_jsd(
    score_bkg: np.ndarray,
    x_bkg: np.ndarray,
    effs: tuple[float, ...] = (0.1, 0.01),
    bins: np.ndarray | int = 50,
) -> dict[str, float]:
    """JSD between the background ``x`` distribution before and after a score cut at background eff ``eff``."""
    score_bkg, x_bkg = np.asarray(score_bkg), np.asarray(x_bkg)
    ok = np.isfinite(x_bkg) & np.isfinite(score_bkg)
    score_bkg, x_bkg = score_bkg[ok], x_bkg[ok]
    edges = np.histogram_bin_edges(x_bkg, bins=bins) if np.ndim(bins) == 0 else np.asarray(bins)
    h_all, _ = np.histogram(x_bkg, bins=edges)
    out = {}
    for eff in effs:
        thr = np.quantile(score_bkg, 1.0 - eff)
        h_cut, _ = np.histogram(x_bkg[score_bkg >= thr], bins=edges)
        out[f"jsd@{eff}"] = jsd(h_all, h_cut)
    return out


def distance_correlation(x: np.ndarray, y: np.ndarray, n_max: int = 4000, seed: int = 0) -> float:
    """Sample distance correlation (Szekely et al.) on a random subset of at most ``n_max`` points."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) > n_max:
        idx = np.random.default_rng(seed).choice(len(x), n_max, replace=False)
        x, y = x[idx], y[idx]

    def centred(v: np.ndarray) -> np.ndarray:
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(0)[None, :] - d.mean(1)[:, None] + d.mean()

    a, b = centred(x), centred(y)
    dcov2 = (a * b).mean()
    dvar = np.sqrt((a * a).mean() * (b * b).mean())
    return float(np.sqrt(max(dcov2, 0.0) / dvar)) if dvar > 0 else 0.0
