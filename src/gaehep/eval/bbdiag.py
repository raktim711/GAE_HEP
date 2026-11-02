"""Diagnostics of the black-box results (docs/STUDY_PROTOCOL.md, Deviations D4).

E1 found every encoder below AUC 0.5 on BB3 (dijet mode 0.43).  Three readings are possible: the encoders score
the box's signal jets as *less* unusual than its background (a physics statement about the jets), the box's
background is shifted with respect to the R&D background the models were trained on (generator change), or the
labels/modes are mixed up.  ``gaehep bb-diagnose RUN [RUN ...] --sample bb3`` reports what separates them:

* ``observables``: jet-level AUCs of single observables (m_J, n_const, pT, tau21 from ``gaehep tau`` if present,
  each in both directions) for every signal set (all, and per BB3 mode) against the box background, in the
  window and over the full range, plus the mean m_J and n_const of each set.  These need no model.
* per run ``shift``: KS distance between the box-background jet scores and the run's R&D ``bkg_test`` jet scores,
  and the fraction of box-background jets above the R&D 90 % and 99 % quantiles (0.10 and 0.01 without a shift).
* per run ``mass``: Spearman correlation of the jet score with m_J on the box background and on R&D ``bkg_test``.
* per run ``score``: jet-level AUC of the run's score for every signal set and region (as a check against
  ``bb<i>_metrics.json``).

It reads ``bb<i>_scores.npz`` (written by ``gaehep score-sample``, not committed) and the run's ``scores.npz``,
so it runs in the session that scored the boxes.  Output: ``results/study/<sample>_diagnostics.{json,md}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

QUANTILES = (0.90, 0.99)


def _auc(sig: np.ndarray, bkg: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    sig, bkg = np.asarray(sig, float), np.asarray(bkg, float)
    sig, bkg = sig[np.isfinite(sig)], bkg[np.isfinite(bkg)]
    if not len(sig) or not len(bkg):
        return None
    return float(roc_auc_score(np.r_[np.zeros(len(bkg)), np.ones(len(sig))], np.r_[bkg, sig]))


def jet_observables(tab: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """(n_events, 2) per-jet observables available in a jet table."""
    out = {k: np.asarray(tab[k], dtype=np.float64) for k in ("m", "n_const", "pt") if k in tab}
    if "tau1c" in tab and "tau2c" in tab:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["tau21c"] = np.asarray(tab["tau2c"], float) / np.asarray(tab["tau1c"], float)
    return out


def signal_sets(sample: str, label: np.ndarray, label_all: np.ndarray) -> dict[str, np.ndarray]:
    from gaehep.eval.blackbox import signal_masks

    return signal_masks(sample, label, label_all)


def observables_report(
    obs: Mapping[str, np.ndarray],
    mjj: np.ndarray,
    sets: Mapping[str, np.ndarray],
    bkg: np.ndarray,
    window: tuple[float, float] | None,
) -> dict[str, Any]:
    regions = {"full": np.ones(len(mjj), bool)}
    if window:
        regions["window"] = (mjj >= window[0]) & (mjj <= window[1])
    out: dict[str, Any] = {}
    for name, sm in sets.items():
        for reg, rm in regions.items():
            s, b = sm & rm, bkg & rm
            e: dict[str, Any] = {"n_sig_events": int(s.sum()), "n_bkg_events": int(b.sum())}
            for k, v in obs.items():
                e[f"auc_{k}"] = _auc(v[s].ravel(), v[b].ravel())
                e[f"auc_-{k}"] = None if e[f"auc_{k}"] is None else 1.0 - e[f"auc_{k}"]
                e[f"mean_{k}_sig"] = float(np.nanmean(v[s])) if s.any() else None
                e[f"mean_{k}_bkg"] = float(np.nanmean(v[b])) if b.any() else None
            out[f"{name}/{reg}"] = e
    return out


def run_report(
    jet_scores: np.ndarray,
    mjj: np.ndarray,
    sets: Mapping[str, np.ndarray],
    bkg: np.ndarray,
    window: tuple[float, float] | None,
    m_box: np.ndarray,
    rnd_bkg_scores: np.ndarray | None,
    rnd_bkg_m: np.ndarray | None,
) -> dict[str, Any]:
    from scipy.stats import ks_2samp, spearmanr

    out: dict[str, Any] = {}
    box_b = jet_scores[bkg].ravel()
    if rnd_bkg_scores is not None:
        r = np.asarray(rnd_bkg_scores, float).ravel()
        out["shift"] = {
            "ks_jet": float(ks_2samp(box_b, r).statistic),
            "median_box_bkg": float(np.median(box_b)),
            "median_rnd_bkg": float(np.median(r)),
            **{
                f"frac_above_rnd_q{int(100 * q)}": float(np.mean(box_b > np.quantile(r, q)))
                for q in QUANTILES
            },
        }
    out["mass"] = {"spearman_score_mJ_box_bkg": float(spearmanr(box_b, m_box[bkg].ravel()).statistic)}
    if rnd_bkg_scores is not None and rnd_bkg_m is not None:
        out["mass"]["spearman_score_mJ_rnd_bkg"] = float(
            spearmanr(np.asarray(rnd_bkg_scores).ravel(), np.asarray(rnd_bkg_m).ravel()).statistic
        )
    regions = {"full": np.ones(len(mjj), bool)}
    if window:
        regions["window"] = (mjj >= window[0]) & (mjj <= window[1])
    out["score"] = {
        f"{name}/{reg}": _auc(jet_scores[sm & rm].ravel(), jet_scores[bkg & rm].ravel())
        for name, sm in sets.items()
        for reg, rm in regions.items()
    }
    return out


def diagnose(runs: list[str], sample: str = "bb3", data=None, log=print) -> dict[str, Path]:
    from gaehep.config import BLACKBOX_WINDOWS, results_dir, save_json
    from gaehep.train.run import Data, _run_dir

    data = data or Data(n_const=50)
    store = data.store(sample)
    window = BLACKBOX_WINDOWS.get(sample)
    obs_report = None
    per_run: dict[str, Any] = {}
    for run in runs:
        rd = _run_dir(run)
        path = rd / f"{sample}_scores.npz"
        if not path.exists():
            log(f"[bb-diagnose] {run}: no {path.name}, skipped")
            continue
        with np.load(path) as f:
            js, ev, mjj, label = f["jet_scores"], f["event_index"], f["mjj"], f["label"]
        tab = {k: store[k][ev] for k in ("m", "n_const", "pt", "tau1c", "tau2c", "tau3c") if k in store.table}
        sets = signal_sets(sample, label, store["label"]) if "label" in store.table else {}
        bkg = label == 0
        if obs_report is None:
            obs_report = observables_report(jet_observables(tab), mjj, sets, bkg, window)
        rnd_s = rnd_m = None
        if (rd / "scores.npz").exists():
            with np.load(rd / "scores.npz") as f:
                rnd_s = f["bkg_test"] if "bkg_test" in f.files else None
            if rnd_s is not None and data.has("bkg_test"):
                rnd_m = data.table("bkg_test")["m"]
                if rnd_m.shape != rnd_s.shape:
                    rnd_m = None
        per_run[run] = run_report(js, mjj, sets, bkg, window, tab["m"], rnd_s, rnd_m)
        log(f"[bb-diagnose] {run}: done")
    out_dir = results_dir() / "study"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"sample": sample, "window": window, "observables": obs_report, "runs": per_run}
    paths = {"json": out_dir / f"{sample}_diagnostics.json", "md": out_dir / f"{sample}_diagnostics.md"}
    save_json(res, paths["json"])
    paths["md"].write_text(render(res))
    return paths


def _f(x, fmt="{:.3f}") -> str:
    return "–" if x is None else fmt.format(x)


def render(res: Mapping[str, Any]) -> str:
    lines = [f"# Diagnostics of {res['sample']}", "", "## Single observables (jet level, box background)", ""]
    obs = res.get("observables") or {}
    if obs:
        keys = [k for k in next(iter(obs.values())) if k.startswith("auc_") and not k.startswith("auc_-")]
        lines += [
            "| signal / region | n sig | n bkg | "
            + " | ".join(k[4:] for k in keys)
            + " | mean m_J sig / bkg |",
            "|---|---|---|" + "---|" * len(keys) + "---|",
        ]
        for name, e in obs.items():
            lines.append(
                f"| {name} | {e['n_sig_events']} | {e['n_bkg_events']} | "
                + " | ".join(_f(e[k]) for k in keys)
                + f" | {_f(e.get('mean_m_sig'), '{:.0f}')} / {_f(e.get('mean_m_bkg'), '{:.0f}')} |"
            )
    lines += [
        "",
        "## Per run",
        "",
        "| run | KS(box bkg, R&D bkg) | frac > R&D q90 | ρ(score, m_J) box | ρ(score, m_J) R&D |",
        "|---|---|---|---|---|",
    ]
    for run, r in res["runs"].items():
        sh, ms = r.get("shift", {}), r.get("mass", {})
        lines.append(
            f"| {run} | {_f(sh.get('ks_jet'))} | {_f(sh.get('frac_above_rnd_q90'))} | "
            f"{_f(ms.get('spearman_score_mJ_box_bkg'), '{:+.2f}')} | {_f(ms.get('spearman_score_mJ_rnd_bkg'), '{:+.2f}')} |"
        )
    return "\n".join(lines) + "\n"
