"""Evaluation driver: turns per-jet scores of the test splits into metrics and figures.

Per-jet scores are stored as (n_events, 2) arrays (leading, sub-leading jet in pT order).  Event scores
are symmetric functions of the two jets (max / min / sum), hence independent of jet order.  Metrics are
reported for the 2-prong (``sig_test``) and 3-prong (``sig3_test``) signals against the background test
split, over the full m_jj range and in the signal region (SR, 3.3-3.7 TeV).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gaehep.config import SR_MJJ, save_json
from gaehep.eval import plots
from gaehep.eval.metrics import DEFAULT_MIN_BKG, bootstrap, summarize
from gaehep.eval.sculpting import distance_correlation, sculpting_jsd

AGGREGATIONS = ("max", "min", "sum", "prod")
SIGNALS = {"sig_test": "2-prong", "sig3_test": "3-prong"}
TEST_KEYS = ("bkg_test", "sig_test", "sig3_test")


def jet_reference(scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Sorted background jet scores defining the background CDF used by the ``prod`` aggregation
    (background validation split when available, else the background test split; labels are not used)."""
    key = "bkg_val" if "bkg_val" in scores else "bkg_test"
    return np.sort(np.asarray(scores[key], dtype=np.float64).ravel())


def aggregate(jet_scores: np.ndarray, how: str, ref: np.ndarray | None = None) -> np.ndarray:
    """Symmetric event score from the two jets' scores.

    ``max`` / ``min`` / ``sum`` of the raw scores, or ``prod``: log F_B(s_1) + log F_B(s_2), the log of the
    product of the two jets' background CDF values (``ref`` = sorted background jet scores) - large only if
    *both* jets are in the upper tail, and insensitive to the scale of the raw score.
    """
    if how == "max":
        return jet_scores.max(1)
    if how == "min":
        return jet_scores.min(1)
    if how == "sum":
        return jet_scores.sum(1)
    if how == "prod":
        if ref is None:
            raise ValueError("the 'prod' aggregation needs the background reference scores")
        cdf = (np.searchsorted(ref, jet_scores, side="right") + 0.5) / (len(ref) + 1.0)
        return np.log(cdf).sum(1)
    raise ValueError(how)


def region_mask(mjj: np.ndarray, region: str, window: tuple[float, float] | None = None) -> np.ndarray:
    """``full``: every event; ``SR``: the R&D signal region (3.3-3.7 TeV); ``window``: the given ``window``
    (lo, hi) in GeV, e.g. a black box's signal window (``gaehep.config.BLACKBOX_WINDOWS``)."""
    if region == "full":
        return np.ones(len(mjj), dtype=bool)
    if region == "SR":
        return (mjj >= SR_MJJ[0]) & (mjj <= SR_MJJ[1])
    if region == "window":
        if window is None:
            raise ValueError("region 'window' needs the window (lo, hi)")
        return (mjj >= window[0]) & (mjj <= window[1])
    raise ValueError(region)


def evaluate_scores(
    scores: Mapping[str, np.ndarray],
    tables: Mapping[str, Mapping[str, np.ndarray]],
    n_boot: int = 30,
    min_bkg: int = DEFAULT_MIN_BKG,
    aggregations: tuple[str, ...] = AGGREGATIONS,
) -> dict[str, Any]:
    """scores[split] -> (n, 2) per-jet scores; tables[split] -> {'mjj': (n,), 'm': (n, 2), ...}."""
    res: dict[str, Any] = {"event": {}, "jet": {}, "sculpting": {}, "correlation": {}}
    b_tab = tables["bkg_test"]
    ref = jet_reference(scores)
    for agg in aggregations:
        b_ev = aggregate(scores["bkg_test"], agg, ref)
        res["event"][agg] = {}
        for sk in SIGNALS:
            if sk not in scores:
                continue
            s_ev = aggregate(scores[sk], agg, ref)
            for region in ("full", "SR"):
                bs = b_ev[region_mask(b_tab["mjj"], region)]
                ss = s_ev[region_mask(tables[sk]["mjj"], region)]
                entry = summarize(ss, bs, min_bkg)
                if n_boot and agg in ("max", "prod"):
                    entry["bootstrap"] = bootstrap(ss, bs, n_boot=n_boot, min_bkg=min_bkg)
                res["event"][agg][f"{sk}/{region}"] = entry
        res["sculpting"][agg] = {
            "mjj": sculpting_jsd(b_ev, b_tab["mjj"]),
            "mj_max": sculpting_jsd(b_ev, np.nanmax(b_tab["m"], axis=1)),
        }
        res["correlation"][agg] = {
            "dcor_mjj": distance_correlation(b_ev, b_tab["mjj"]),
            "dcor_mj_max": distance_correlation(b_ev, np.nanmax(b_tab["m"], axis=1)),
        }
    # jet level: every jet of background events vs every jet of signal events
    b_j = scores["bkg_test"].ravel()
    for sk in SIGNALS:
        if sk in scores:
            res["jet"][sk] = summarize(scores[sk].ravel(), b_j, min_bkg)
    bm = b_tab["m"].ravel()
    res["correlation"]["jet"] = {
        "dcor_mj": distance_correlation(b_j, bm),
        "dcor_nconst": distance_correlation(b_j, b_tab["n_const"].ravel().astype(np.float64)),
    }
    return res


def headline(metrics: Mapping[str, Any], agg: str = "max") -> dict[str, float]:
    """Compact numbers for tables: AUC and max SIC (2-prong/3-prong, SR) for one aggregation."""
    ev = metrics["event"][agg]
    out = {}
    for sk, short in (("sig_test", "2p"), ("sig3_test", "3p")):
        for region in ("SR", "full"):
            e = ev.get(f"{sk}/{region}")
            if e:
                out[f"auc_{short}_{region}"] = e["auc"]
                out[f"maxsic_{short}_{region}"] = e["max_sic"]
                out[f"rej30_{short}_{region}"] = e["rej@0.3"]
    sc = metrics["sculpting"][agg]["mjj"]
    out["jsd_mjj@0.1"] = sc["jsd@0.1"]
    out["jsd_mjj@0.01"] = sc["jsd@0.01"]
    out["dcor_mj_jet"] = metrics["correlation"]["jet"]["dcor_mj"]
    return out


def make_figures(
    scores: Mapping[str, np.ndarray],
    tables: Mapping[str, Mapping[str, np.ndarray]],
    out_dir: Path,
    name: str,
    agg: str = "max",
    history: list[dict] | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    paths = []
    ref = jet_reference(scores)
    b = aggregate(scores["bkg_test"], agg, ref)
    bsr = b[region_mask(tables["bkg_test"]["mjj"], "SR")]
    curves = {}
    for sk, lab in SIGNALS.items():
        if sk in scores:
            s = aggregate(scores[sk], agg, ref)
            curves[f"{lab} (SR)"] = (s[region_mask(tables[sk]["mjj"], "SR")], bsr)
    paths.append(plots.plot_rejection(curves, out_dir / "rejection_SR.png", title=f"{name} [{agg}]"))
    paths.append(plots.plot_sic(curves, out_dir / "sic_SR.png", title=f"{name} [{agg}]"))
    series = {"background": b}
    for sk, lab in SIGNALS.items():
        if sk in scores:
            series[lab] = aggregate(scores[sk], agg, ref)
    paths.append(plots.plot_score_hist(series, out_dir / "score_hist.png", title=f"{name} [{agg}]"))
    paths.append(
        plots.plot_sculpting(tables["bkg_test"]["mjj"], b, out_dir / "sculpting_mjj.png", title=name)
    )
    if history:
        paths.append(plots.plot_history(history, out_dir / "history.png", title=name))
    return paths


def write_metrics(metrics: dict[str, Any], path: Path, meta: dict[str, Any] | None = None) -> None:
    save_json({**(meta or {}), "headline": headline(metrics), "metrics": metrics}, path)


# ------------------------------------------------------------------------------------------ baselines
def observable_scores(tab: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-jet scores from single observables (larger = more anomalous).  NaN -> very low score."""
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = {
            "m_j": tab["m"].astype(np.float64),
            "n_const": tab["n_const"].astype(np.float64),
            "-tau21": -(tab["tau2"] / tab["tau1"]).astype(np.float64),
            "-tau32": -(tab["tau3"] / tab["tau2"]).astype(np.float64),
        }
    return {k: np.where(np.isfinite(v), v, -1e9) for k, v in obs.items()}
