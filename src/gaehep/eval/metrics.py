"""Detection metrics for anomaly scores (larger score = more signal-like).

* ROC AUC.
* SIC = eps_S / sqrt(eps_B) (significance improvement of a cut); ``max_sic`` only considers cuts that
  keep at least ``min_bkg`` background events, so that it is not driven by a handful of events.
* Background rejection 1/eps_B at fixed signal efficiency.
* Bootstrap uncertainties (signal and background resampled independently).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

DEFAULT_MIN_BKG = 50
EFFS = (0.2, 0.3, 0.5)


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite scores")
    return x


def roc(sig: np.ndarray, bkg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig, bkg = _clean(sig), _clean(bkg)
    y = np.r_[np.ones(len(sig)), np.zeros(len(bkg))]
    fpr, tpr, thr = roc_curve(y, np.r_[sig, bkg], drop_intermediate=False)
    return fpr, tpr, thr


def auc(sig: np.ndarray, bkg: np.ndarray) -> float:
    sig, bkg = _clean(sig), _clean(bkg)
    return float(roc_auc_score(np.r_[np.ones(len(sig)), np.zeros(len(bkg))], np.r_[sig, bkg]))


def sic_curve(
    sig: np.ndarray, bkg: np.ndarray, min_bkg: int = DEFAULT_MIN_BKG
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eps_S, eps_B, SIC) for all cuts that keep >= ``min_bkg`` background events."""
    fpr, tpr, _ = roc(sig, bkg)
    ok = fpr * len(bkg) >= min_bkg
    return tpr[ok], fpr[ok], tpr[ok] / np.sqrt(fpr[ok])


def max_sic(sig: np.ndarray, bkg: np.ndarray, min_bkg: int = DEFAULT_MIN_BKG) -> tuple[float, float]:
    """Maximum SIC and the signal efficiency where it occurs."""
    es, _, s = sic_curve(sig, bkg, min_bkg)
    if len(s) == 0:
        return float("nan"), float("nan")
    i = int(np.argmax(s))
    return float(s[i]), float(es[i])


def at_signal_eff(sig: np.ndarray, bkg: np.ndarray, eff: float) -> tuple[float, float]:
    """(1/eps_B, SIC) at the loosest cut with eps_S >= eff.  1/eps_B is inf if no background passes."""
    fpr, tpr, _ = roc(sig, bkg)
    i = int(np.searchsorted(tpr, eff, side="left"))
    i = min(i, len(tpr) - 1)
    f, t = fpr[i], tpr[i]
    rej = float("inf") if f == 0 else 1.0 / f
    sic = float("inf") if f == 0 else t / np.sqrt(f)
    return rej, sic


def summarize(sig: np.ndarray, bkg: np.ndarray, min_bkg: int = DEFAULT_MIN_BKG) -> dict[str, float]:
    out: dict[str, float] = {"n_sig": int(np.size(sig)), "n_bkg": int(np.size(bkg)), "auc": auc(sig, bkg)}
    out["max_sic"], out["eff_s_at_max_sic"] = max_sic(sig, bkg, min_bkg)
    for e in EFFS:
        rej, sic = at_signal_eff(sig, bkg, e)
        out[f"rej@{e}"] = rej
        out[f"sic@{e}"] = sic
    return out


def bootstrap(
    sig: np.ndarray, bkg: np.ndarray, n_boot: int = 30, seed: int = 0, min_bkg: int = DEFAULT_MIN_BKG
) -> dict[str, dict[str, float]]:
    """Bootstrap mean / std / 16-84% interval of AUC and max SIC."""
    rng = np.random.default_rng(seed)
    sig, bkg = _clean(sig), _clean(bkg)
    aucs, sics = [], []
    for _ in range(n_boot):
        s = sig[rng.integers(0, len(sig), len(sig))]
        b = bkg[rng.integers(0, len(bkg), len(bkg))]
        aucs.append(auc(s, b))
        sics.append(max_sic(s, b, min_bkg)[0])
    res = {}
    for k, v in (("auc", np.asarray(aucs)), ("max_sic", np.asarray(sics))):
        res[k] = {
            "mean": float(np.nanmean(v)),
            "std": float(np.nanstd(v)),
            "q16": float(np.nanquantile(v, 0.16)),
            "q84": float(np.nanquantile(v, 0.84)),
        }
    return res
