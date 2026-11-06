"""Follow-up analyses of the bias audit of 6 Nov 2026 (docs/STUDY_PROTOCOL.md, Deviations D9).

All of this was added after every study result had been seen.  It tests how far the paper's claims hold; it
replaces none of the pre-registered verdicts.

Analyses that need only committed files (``gaehep audit-followup`` -> ``results/study/audit_followup.{md,json}``):

* ``curve_checks``: how much of the spread of AUC between runs the fit-quality curve (a quadratic in the
  background validation loss, fitted to the E1 + E2 runs) explains, compared with knowing only each run's
  configuration (encoder x training length); the within-configuration seed test (seeds change the fit at a
  fixed architecture and training length, so it is a fixed-architecture test of the fit-AUC relation); and the
  out-of-sample R^2 on the E4 ablation runs.
* ``equal_fit_robust``: the equal-fit contrasts with five functional forms of the curve, a bootstrap standard
  error that includes the uncertainty of the curve (runs resampled within configurations), and 95 % intervals.
* ``irc_contrasts``: IRC-safe minus IRC-unsafe score stability for the collinear split and both soft-emission
  probes.
* ``decomposition``: inputs x pooling at equal fit for the set family (Deep Sets, EFN) and the graph family
  (EdgeConv, EWMP), from the E1 runs, the E4 weighted-pooling runs and the E5 runs (the IRC-safe encoders with
  mean + max pooling, ``e5_<enc>_meanmax_s<s>``).
* ``mass_mechanism``: whether the score's correlation with the jet mass rises with the background loss (reads
  ``<run>/mass_corr.json``).
* ``lee_summary``: the look-elsewhere calibration of the BB1 bump hunt (reads ``<run>/rnd_scan.json``).

Scoring commands that write those files from the committed weights (``results/models/<run>/model.pt``):

* ``gaehep mass-corr RUN...``: Spearman correlation of the jet score with m_J on R&D ``bkg_test`` events.
* ``gaehep rnd-scan RUN...``: the BB1 train-on-data bump hunt repeated without signal.  An R&D-trained run is
  scanned on its own training sample, the whole R&D background (1 000 000 events, as many as BB1), with the same
  :func:`gaehep.eval.bumphunt.search` over 14 windows of 400 GeV and the three cuts of the BB1 hunt.  Toy
  experiments drawn from smooth fits of the selected spectra give the distribution of the largest local Z of such
  a scan when there is no signal, i.e. the global significance of a local excess.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from gaehep.config import BLACKBOX_WINDOWS, load_json, results_dir, run_metadata, save_json
from gaehep.eval import posthoc as P
from gaehep.eval.bumphunt import DEFAULT_EDGES, SQRT_S, _bin_expectation, fit_sidebands, poisson_z, search
from gaehep.eval.study import Config, collect

ENCODERS = ("deepsets", "efn", "edgeconv", "ewmp", "transformer", "ew_attention", "gcn_complete")
SCAN_WINDOWS: tuple[tuple[float, float], ...] = tuple(
    (float(lo), float(lo) + 400.0) for lo in range(3000, 4301, 100)
)
SCAN_EFFS = (1.0, 0.1, 0.02)  # the cuts of the BB1 hunt (scripts/e3_bumphunt.py)
IRC_PAIRS = (("efn", "deepsets"), ("ewmp", "edgeconv"), ("ew_attention", "transformer"))
IRC_PROBES = ("split top-1", "split all", "10 soft (0.2% pT each)", "10 soft (1% pT each)")
FAMILIES = {  # (inputs, pooling) -> (run prefix, configuration label)
    "set": {
        ("unsafe", "meanmax"): ("study", "deepsets"),
        ("unsafe", "weighted"): ("stat", "deepsets_weighted"),
        ("safe", "meanmax"): ("e5", "efn_meanmax"),
        ("safe", "weighted"): ("study", "efn"),
    },
    "graph": {
        ("unsafe", "meanmax"): ("study", "edgeconv"),
        ("unsafe", "weighted"): ("stat", "edgeconv_weighted"),
        ("safe", "meanmax"): ("e5", "ewmp_meanmax"),
        ("safe", "weighted"): ("study", "ewmp"),
    },
}
CURVE_FORMS = ("quadratic", "linear", "cubic", "quadratic in log(loss)", "quadratic without GCN")


# ------------------------------------------------------------------------------------------ small helpers
def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """1 - SS_res / SS_tot around the mean of ``y``; negative when ``yhat`` predicts worse than the mean."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def group_centred(v: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """``v`` minus the mean of its group."""
    out = np.asarray(v, float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean()
    return out


def within_group_test(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Spearman correlation and least-squares slope of y on x after removing every group's mean of both
    (groups with a single member carry no within-group information and are dropped)."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    keep = np.isin(groups, [g for g in np.unique(groups) if np.sum(groups == g) > 1])
    dx, dy = group_centred(x[keep], groups[keep]), group_centred(y[keep], groups[keep])
    rho, p = stats.spearmanr(dx, dy)
    slope = float(np.sum(dx * dy) / np.sum(dx * dx)) if np.sum(dx * dx) > 0 else float("nan")
    return {
        "n": int(keep.sum()),
        "groups": len(np.unique(groups[keep])),
        "spearman": float(rho),
        "p": float(p),
        "slope": slope,
    }


def welch(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a.mean() - b.mean())
    se = (
        float(math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))) if min(len(a), len(b)) > 1 else None
    )
    return {"delta": d, "se": se, "sig2se": None if se is None else bool(abs(d) > 2 * se)}


def _points(cfgs: Sequence[Config], signal: str) -> list[tuple[str, str, float, float]]:
    """(configuration label, run name, background validation loss, AUC) of every seeded run."""
    out = []
    for c in cfgs:
        v = P.seed_values(c, signal, "auc")
        out += [
            (c.label, r.name, float(r.best_val_loss), float(v[P._seed_of(r)]))
            for r in c.runs
            if P._seed_of(r) in v and r.best_val_loss is not None
        ]
    return out


def _arrays(pts: list[tuple[str, str, float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([p[0] for p in pts]), np.array([p[2] for p in pts]), np.array([p[3] for p in pts]))


def fit_curve(x: np.ndarray, y: np.ndarray, form: str = "quadratic") -> Callable[[np.ndarray], np.ndarray]:
    """The AUC-vs-loss curve in one of ``CURVE_FORMS`` (the GCN exclusion is applied by the caller)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if form == "quadratic in log(loss)":
        c = np.polyfit(np.log(x), y, 2)
        return lambda u, c=c: np.polyval(c, np.log(np.asarray(u, float)))
    deg = {"quadratic": 2, "quadratic without GCN": 2, "linear": 1, "cubic": 3}[form]
    c = np.polyfit(x, y, deg)
    return lambda u, c=c: np.polyval(c, np.asarray(u, float))


def _main_configs(root: Path) -> dict[str, Config]:
    return {c.label: c for c in collect("study", root) if c.runs and c.label in ENCODERS}


def _cfgs(prefix: str, root: Path) -> dict[str, Config]:
    return {c.label: c for c in collect(prefix, root) if c.runs}


# ------------------------------------------------------------------------------------------ 1 curve checks
def curve_checks(
    main: dict[str, Config], curves: dict[str, Config], ablations: dict[str, Config], signals=P.UNTUNED
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fit_cfgs = [*main.values(), *curves.values()]
    for s in signals:
        g, x, y = _arrays(_points(fit_cfgs, s))
        if len(y) < 10:
            continue
        f = fit_curve(x, y)
        res = y - f(x)
        cfg_mean = y - group_centred(y, g)
        enc = np.array([lbl.split("_e")[0] if lbl.split("_e")[0] in ENCODERS else lbl for lbl in g])
        # leave-one-out versions: the configuration means are 28 parameters fitted to ~100 runs, so their in-sample
        # R^2 is flattered; each run is predicted by the curve refitted without it and by the mean of the other
        # seeds of its configuration
        n = len(y)
        loo_c, loo_g = np.empty(n), np.empty(n)
        for k in range(n):
            m = np.ones(n, bool)
            m[k] = False
            loo_c[k] = fit_curve(x[m], y[m])(x[k : k + 1])[0]
            same = (g == g[k]) & m
            loo_g[k] = y[same].mean() if same.any() else np.nan
        ok = np.isfinite(loo_g)
        e_entry: dict[str, Any] = {
            "n_runs": len(y),
            "r2_curve_loo": r2(y, loo_c),
            "r2_configuration_loo": r2(y[ok], loo_g[ok]),
            "n_configs": len(np.unique(g)),
            "auc_sd": float(y.std(ddof=1)),
            "resid_sd": float(np.std(res, ddof=3)),
            "r2_curve": r2(y, f(x)),
            "r2_configuration": r2(y, cfg_mean),
            # the share of the curve's residual variance that configuration identity still explains
            "r2_configuration_on_residual": r2(res, res - group_centred(res, g)),
            "within_configuration": within_group_test(x, y, g),
            "within_configuration_no_deepsets": within_group_test(
                x[enc != "deepsets"], y[enc != "deepsets"], g[enc != "deepsets"]
            ),
            "curve_slope_at_mean_loss": float((f(x.mean() + 1e-4) - f(x.mean() - 1e-4)) / 2e-4),
        }
        if ablations:
            _, xa, ya = _arrays(_points(list(ablations.values()), s))
            if len(ya) > 2:
                ra = ya - f(xa)
                e_entry["out_of_sample"] = {
                    "n_runs": len(ya),
                    "r2": r2(ya, f(xa)),
                    "rmse": float(np.sqrt(np.mean(ra**2))),
                    "mean_residual": float(ra.mean()),
                    "auc_sd": float(ya.std(ddof=1)),
                }
        out[s] = e_entry
    return out


# ------------------------------------------------------------------------------- 2 equal-fit robustness
def _auc_pairs() -> list[tuple[str, str, str, str]]:
    return [(h, role, a, b) for h, role, a, b, metric, _, _ in P.HYPOTHESES if metric == "auc"]


def equal_fit_robust(
    main: dict[str, Config],
    curves: dict[str, Config],
    signals=P.UNTUNED,
    n_boot: int = 2000,
    seed: int = 20261106,
) -> dict[str, Any]:
    """Equal-fit contrasts (AUC residuals of A minus B over the E1 seeds) for every functional form of the
    curve, and for the quadratic a bootstrap SE that includes the curve's own uncertainty: every replica
    resamples the runs of each configuration with replacement, refits the curve and recomputes the contrast."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    cfgs = {**main, **curves}
    for s in signals:
        per = {lbl: _points([c], s) for lbl, c in cfgs.items()}
        per = {k: v for k, v in per.items() if v}
        if not per:
            continue
        rows = []
        for hyp, role, a, b in _auc_pairs():
            if a not in per or b not in per:
                continue
            row: dict[str, Any] = {"hypothesis": hyp, "role": role, "a": a, "b": b, "forms": {}}
            for form in CURVE_FORMS:
                pts = [
                    p for k, v in per.items() for p in v if not (form.endswith("GCN") and k.startswith("gcn"))
                ]
                _, x, y = _arrays(pts)
                f = fit_curve(x, y, form)
                ra = [p[3] - float(f(p[2])) for p in per[a]]
                rb = [p[3] - float(f(p[2])) for p in per[b]]
                row["forms"][form] = welch(ra, rb)
            labels = list(per)
            boot = np.empty(n_boot)
            for i in range(n_boot):
                pick = {k: rng.integers(0, len(per[k]), len(per[k])) for k in labels}
                pts = [per[k][j] for k in labels for j in pick[k]]
                _, x, y = _arrays(pts)
                c = np.polyfit(x, y, 2)
                ra = np.array([per[a][j][3] - np.polyval(c, per[a][j][2]) for j in pick[a]])
                rb = np.array([per[b][j][3] - np.polyval(c, per[b][j][2]) for j in pick[b]])
                boot[i] = ra.mean() - rb.mean()
            q = row["forms"]["quadratic"]
            se = max(q["se"], float(boot.std(ddof=1)))
            row.update(
                {
                    "delta": q["delta"],
                    "se_welch": q["se"],
                    "se_boot": float(boot.std(ddof=1)),
                    "ci95": [q["delta"] - 1.96 * se, q["delta"] + 1.96 * se],
                    "sig_all_forms": all(v["sig2se"] for v in row["forms"].values()),
                    "sig_any_form": any(v["sig2se"] for v in row["forms"].values()),
                    "sig_boot": bool(abs(q["delta"]) > 2 * se),
                }
            )
            rows.append(row)
        out[s] = rows
    return out


# ---------------------------------------------------------------------------------- 3 IRC contrasts
def irc_contrasts(main: dict[str, Config]) -> dict[str, Any]:
    def rho(enc: str, probe: str) -> list[float]:
        c = main.get(enc)
        return [r.irc[probe]["rho"] for r in (c.runs if c else []) if r.irc and probe in r.irc]

    per_enc = {
        e: {
            p: {
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                "n": len(v),
            }
            for p in IRC_PROBES
            if (v := rho(e, p))
        }
        for e in ENCODERS
    }
    rows = []
    for a, b in IRC_PAIRS:
        for p in IRC_PROBES:
            va, vb = rho(a, p), rho(b, p)
            if va and vb:
                rows.append({"a": a, "b": b, "probe": p, **welch(va, vb)})
    return {"encoders": per_enc, "contrasts": rows}


# ------------------------------------------------------------------------------- 4 inputs x pooling
def decomposition(
    main: dict[str, Config],
    curves: dict[str, Config],
    groups: dict[str, dict[str, Config]],
    signals=P.UNTUNED,
) -> dict[str, Any]:
    """The 2x2 (inputs x pooling) of each family, raw and at equal fit (residuals from the E1 + E2 quadratic).
    ``groups`` maps a run prefix (``study``, ``stat``, ``e5``) to its configurations."""
    out: dict[str, Any] = {}
    fit_cfgs = [*main.values(), *curves.values()]
    for s in signals:
        _, x, y = _arrays(_points(fit_cfgs, s))
        if len(y) < 10:
            continue
        f = fit_curve(x, y)
        fam_out = {}
        for fam, cells in FAMILIES.items():
            cell_out: dict[str, Any] = {}
            res: dict[tuple[str, str], list[float]] = {}
            for key, (prefix, label) in cells.items():
                c = groups.get(prefix, {}).get(label)
                pts = _points([c], s) if c else []
                if not pts:
                    continue
                _, xc, yc = _arrays(pts)
                rc = yc - f(xc)
                res[key] = list(rc)
                irc = {}
                for p in ("split top-1", "10 soft (1% pT each)"):
                    v = [r.irc[p]["rho"] for r in c.runs if r.irc and p in r.irc]
                    irc[p] = float(np.mean(v)) if v else None
                cell_out["/".join(key)] = {
                    "run_prefix": prefix,
                    "config": label,
                    "n": len(yc),
                    "loss": float(xc.mean()),
                    "auc": float(yc.mean()),
                    "auc_sd": float(yc.std(ddof=1)) if len(yc) > 1 else None,
                    "resid": float(rc.mean()),
                    "resid_se": float(rc.std(ddof=1) / math.sqrt(len(rc))) if len(rc) > 1 else None,
                    "irc_rho": irc,
                }
            effects = {}
            if len(res) == 4:
                um, uw, sm, sw = (
                    res[k]
                    for k in [
                        ("unsafe", "meanmax"),
                        ("unsafe", "weighted"),
                        ("safe", "meanmax"),
                        ("safe", "weighted"),
                    ]
                )
                effects = {
                    "pooling | unsafe inputs": welch(uw, um),
                    "pooling | safe inputs": welch(sw, sm),
                    "inputs | mean+max pooling": welch(sm, um),
                    "inputs | weighted pooling": welch(sw, uw),
                    "IRC-safe design (both)": welch(sw, um),
                }
                # interaction: (sw - sm) - (uw - um), SE from the four cell variances
                d = (np.mean(sw) - np.mean(sm)) - (np.mean(uw) - np.mean(um))
                se = math.sqrt(sum(np.var(v, ddof=1) / len(v) for v in (um, uw, sm, sw)))
                effects["interaction"] = {"delta": float(d), "se": se, "sig2se": bool(abs(d) > 2 * se)}
            # the same effects with every functional form of the curve (the cells' losses barely overlap)
            by_form: dict[str, dict[str, Any]] = {}
            if len(res) == 4:
                pts_all = [p for c in fit_cfgs for p in _points([c], s)]
                for form in CURVE_FORMS:
                    pts = [p for p in pts_all if not (form.endswith("GCN") and p[0].startswith("gcn"))]
                    _, xf, yf = _arrays(pts)
                    ff_ = fit_curve(xf, yf, form)
                    rr = {}
                    for key, (prefix, label) in cells.items():
                        _, xc, yc = _arrays(_points([groups[prefix][label]], s))
                        rr[key] = list(yc - ff_(xc))
                    by_form[form] = {
                        "pooling | safe inputs": welch(rr[("safe", "weighted")], rr[("safe", "meanmax")]),
                        "inputs | mean+max pooling": welch(
                            rr[("safe", "meanmax")], rr[("unsafe", "meanmax")]
                        ),
                        "pooling | unsafe inputs": welch(
                            rr[("unsafe", "weighted")], rr[("unsafe", "meanmax")]
                        ),
                        "IRC-safe design (both)": welch(rr[("safe", "weighted")], rr[("unsafe", "meanmax")]),
                    }
            fam_out[fam] = {"cells": cell_out, "effects_at_equal_fit": effects, "effects_by_form": by_form}
        out[s] = fam_out
    return out


# ---------------------------------------------------------------------------------- 5 mass mechanism
def mass_mechanism(main: dict[str, Config], curves: dict[str, Config]) -> dict[str, Any]:
    rows = []
    for c in [*main.values(), *curves.values()]:
        enc = c.label.split("_e")[0]
        bb1, rd3 = P.seed_values(c, "bb1", "auc"), P.seed_values(c, "rd3", "auc")
        for r in c.runs:
            mc = _json(r.dir)
            if not mc or r.best_val_loss is None:
                continue
            sd = P._seed_of(r)
            rows.append(
                {
                    "run": r.name,
                    "config": c.label,
                    "encoder": enc,
                    "loss": float(r.best_val_loss),
                    "rho_mass": float(mc["spearman_score_mJ"]),
                    "auc_bb1": bb1.get(sd),
                    "auc_rd3": rd3.get(sd),
                }
            )
    if len(rows) < 5:
        return {"n_runs": len(rows)}
    enc = np.array([r["encoder"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    rho = np.array([r["rho_mass"] for r in rows])
    out: dict[str, Any] = {"n_runs": len(rows), "runs": rows}

    def sp(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        r, p = stats.spearmanr(a[ok], b[ok])
        return {"spearman": float(r), "p": float(p), "n": int(ok.sum())}

    out["rho_mass_vs_loss"] = sp(rho, loss)
    out["rho_mass_vs_loss_within_encoder"] = within_group_test(loss, rho, enc)
    for s in ("bb1", "rd3"):
        auc = np.array([np.nan if r[f"auc_{s}"] is None else r[f"auc_{s}"] for r in rows], float)
        out[f"auc_{s}_vs_rho_mass"] = sp(auc, rho)
        ok = np.isfinite(auc)
        out[f"auc_{s}_vs_rho_mass_within_encoder"] = within_group_test(rho[ok], auc[ok], enc[ok])
    out["per_encoder"] = {
        e: {
            "rho_mass": float(rho[enc == e].mean()),
            "loss": float(loss[enc == e].mean()),
            "n": int(np.sum(enc == e)),
        }
        for e in ENCODERS
        if np.any(enc == e)
    }
    return out


def _json(run_dir: Path) -> dict | None:
    p = run_dir / "mass_corr.json"
    return load_json(p) if p.exists() else None


# ------------------------------------------------------------------------------------- scoring commands
def _load_model(run: str):
    import torch
    import yaml

    from gaehep.models import build_model

    rdir = results_dir() / "runs" / run
    cfg = yaml.safe_load((rdir / "config.yaml").read_text())
    weights = results_dir() / "models" / run / "model.pt"
    if not weights.exists():
        weights = rdir / "model.pt"
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    torch.set_num_threads(int((cfg.get("train") or {}).get("threads", 1)))
    model.eval()
    return model, cfg


def _jet_scores(model, data, events: np.ndarray, chunk: int = 20000, store=None) -> np.ndarray:
    from gaehep.train.loop import score_pool

    store, scaler = store if store is not None else data.store("rnd"), data.scaler()
    parts = []
    for i in range(0, len(events), chunk):
        pool = store.pool(events[i : i + chunk], data.n_const)
        parts.append(score_pool(model, pool, scaler)["score"].reshape(-1, 2).astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros((0, 2), np.float32)


def mass_corr(run: str, n_events: int = 5000, data=None, log=print) -> dict[str, Any]:
    """Spearman correlation of the jet score with the jet mass on ``n_events`` R&D ``bkg_test`` events (both
    jets), written to ``<run>/mass_corr.json``."""
    from gaehep.train.run import Data

    model, cfg = _load_model(run)
    data = data or Data.from_config(cfg.get("data"))
    ev = data.events("bkg_test", n_events, seed=0)
    s = _jet_scores(model, data, ev)
    m = np.asarray(data.store("rnd")["m"][ev], np.float64)
    rho = float(stats.spearmanr(s.ravel(), m.ravel())[0])
    out = {
        "name": run,
        "split": "bkg_test",
        "n_events": len(ev),
        "n_jets": int(s.size),
        "spearman_score_mJ": rho,
        **{k: v for k, v in run_metadata().items() if k in ("git", "time", "splits_sha256", "torch")},
    }
    save_json(out, results_dir() / "runs" / run / "mass_corr.json")
    log(f"[mass-corr] {run}: rho(score, m_J) = {rho:+.3f} over {s.size} jets")
    return out


class _FastFit:
    """Poisson maximum-likelihood fit of the dijet function with an analytic gradient (L-BFGS-B), for toys.
    It finds the same minimum as :func:`gaehep.eval.bumphunt.fit_sidebands` (checked on the real spectra in
    ``rnd_scan``) when started near it."""

    def __init__(self, edges: np.ndarray = DEFAULT_EDGES, n_sub: int = 5):
        lo, hi = edges[:-1], edges[1:]
        t = (np.arange(n_sub) + 0.5) / n_sub
        xs = (lo[:, None] + (hi - lo)[:, None] * t[None, :]) / SQRT_S
        self.l1, self.lx = np.log1p(-xs), np.log(xs)
        self.w = (hi - lo) / SQRT_S

    def _mu(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = np.exp(np.clip(p[0] + p[1] * self.l1 - p[2] * self.lx - p[3] * self.lx**2, -700, 700))
        mu = f.mean(1) * self.w
        grads = np.stack(
            [
                mu,
                (f * self.l1).mean(1) * self.w,
                -(f * self.lx).mean(1) * self.w,
                -(f * self.lx**2).mean(1) * self.w,
            ]
        )
        return mu, grads

    def fit(self, counts: np.ndarray, mask: np.ndarray, p0: np.ndarray) -> np.ndarray:
        """Minimise the Poisson deviance (the likelihood up to a constant, ~0 at a good fit, so that the
        optimiser's relative stopping rule is meaningful) over the bins in ``mask``."""
        c = counts[mask].astype(np.float64)
        clogc = np.where(c > 0, c * np.log(np.where(c > 0, c, 1.0)), 0.0)

        def dev(p):
            mu, g = self._mu(p)
            mu, g = mu[mask], g[:, mask]
            if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
                return 1e30, np.zeros(4)
            return float(np.sum(mu - c - c * np.log(mu) + clogc)), g @ (1.0 - c / mu)

        best = minimize(
            dev, p0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-7}
        )
        r = minimize(dev, best.x, jac=True, method="BFGS", options={"maxiter": 500, "gtol": 1e-6})
        return r.x if r.fun <= best.fun else best.x

    def expectation(self, p: np.ndarray) -> np.ndarray:
        return self._mu(p)[0]


def _sr_bins(window: tuple[float, float], edges: np.ndarray = DEFAULT_EDGES) -> np.ndarray:
    centres = 0.5 * (edges[:-1] + edges[1:])
    return (centres > window[0]) & (centres < window[1])  # as in bumphunt.search


def scan_z(counts: np.ndarray, ff: _FastFit, p0: np.ndarray, windows=SCAN_WINDOWS) -> np.ndarray:
    """Local Z of every window for one selected spectrum (fast fit started at ``p0``)."""
    z = np.empty(len(windows))
    for i, w in enumerate(windows):
        sr = _sr_bins(w)
        p = ff.fit(counts, ~sr, p0)
        z[i] = poisson_z(float(counts[sr].sum()), float(ff.expectation(p)[sr].sum()))
    return z


def null_toys(
    mjj: np.ndarray, score: np.ndarray, n_toys: int, rng: np.random.Generator, windows=SCAN_WINDOWS
) -> dict[str, Any]:
    """Toy experiments of the scan without signal.  The selections are the three cuts with the thresholds of
    the BB1 window's sidebands; the smooth expectation of each selected spectrum is the dijet fit to all its
    bins.  Toys are nested like the cuts: n(no cut) ~ Poisson, n(10 %) ~ Binomial(n(no cut), ratio), n(2 %) ~
    Binomial(n(10 %), ratio), so the correlation between the cuts is kept.  Every toy is scanned like the data."""
    edges = DEFAULT_EDGES
    in_range = (mjj >= edges[0]) & (mjj < edges[-1])
    w0, w1 = BLACKBOX_WINDOWS["bb1"]
    sb = in_range & ~((mjj >= w0) & (mjj < w1))
    ff = _FastFit(edges)
    mu, params = {}, {}
    for eff in SCAN_EFFS:
        thr = -np.inf if eff >= 1 else float(np.quantile(score[sb], 1 - eff))
        counts, _ = np.histogram(mjj[score >= thr], bins=edges)
        fit = fit_sidebands(counts, edges, np.ones(len(counts), bool))
        params[eff] = fit.params
        mu[eff] = np.clip(np.nan_to_num(_bin_expectation(edges, fit.params), nan=0.0, posinf=0.0), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        r10 = np.clip(np.nan_to_num(mu[0.1] / mu[1.0]), 0.0, 1.0)
        r2_ = np.clip(np.nan_to_num(mu[0.02] / mu[0.1]), 0.0, 1.0)
    i_bb1 = [i for i, w in enumerate(windows) if tuple(w) == (w0, w1)]
    zmax_cuts, zmax_all, z_bb1_2, z_bb1_10 = [], [], [], []
    for _ in range(n_toys):
        n1 = rng.poisson(mu[1.0])
        n10 = rng.binomial(n1, r10)
        n2 = rng.binomial(n10, r2_)
        z = {eff: scan_z(c, ff, params[eff], windows) for eff, c in ((1.0, n1), (0.1, n10), (0.02, n2))}
        zmax_cuts.append(float(max(z[0.1].max(), z[0.02].max())))
        zmax_all.append(float(max(v.max() for v in z.values())))
        if i_bb1:
            z_bb1_2.append(float(z[0.02][i_bb1[0]]))
            z_bb1_10.append(float(z[0.1][i_bb1[0]]))
    return {
        "n_toys": n_toys,
        "zmax_scan_cuts": zmax_cuts,  # largest local Z over the windows x {10 %, 2 %}
        "zmax_scan_all": zmax_all,  # ... over the windows x {no cut, 10 %, 2 %}
        "z_bb1_window_2pct": z_bb1_2,
        "z_bb1_window_10pct": z_bb1_10,
        "params": {str(k): [float(x) for x in v] for k, v in params.items()},
    }


def _scan(mjj: np.ndarray, score: np.ndarray, n_toys: int, seed: int) -> dict[str, Any]:
    """The window scan of one sample (every window x cut, with the analysis fit of ``bumphunt.search`` and with
    the converged fit of the toys started at the analysis optimum; the analysis fit, Nelder-Mead then Powell, can
    stop short of the minimum) and its toy experiments."""
    ff = _FastFit()
    in_range = (mjj >= DEFAULT_EDGES[0]) & (mjj < DEFAULT_EDGES[-1])
    rows = []
    for w in SCAN_WINDOWS:
        sbm = in_range & ~((mjj >= w[0]) & (mjj < w[1]))
        sr = _sr_bins(w)
        for eff in SCAN_EFFS:
            r = {"window": list(w), **search(mjj, score, eff, sr=w)}
            thr = -np.inf if eff >= 1 else float(np.quantile(score[sbm], 1 - eff))
            counts, _ = np.histogram(mjj[score >= thr], bins=DEFAULT_EDGES)
            p_ref = fit_sidebands(counts, DEFAULT_EDGES, ~sr).params
            b = float(ff.expectation(ff.fit(counts, ~sr, p_ref))[sr].sum())
            r["z_converged"] = poisson_z(float(counts[sr].sum()), b)
            rows.append(r)
    dz = np.array([r["z_converged"] - r["z"] for r in rows])
    zc = [r["z_converged"] for r in rows if r["eff"] < 1]
    za = [r["z"] for r in rows if r["eff"] < 1]
    best = max((r for r in rows if r["eff"] < 1), key=lambda r: r["z_converged"])
    return {
        "windows": [list(w) for w in SCAN_WINDOWS],
        "effs": list(SCAN_EFFS),
        "rows": rows,
        "zmax_data_cuts": float(max(zc)),
        "zmax_data_all": float(max(r["z_converged"] for r in rows)),
        "zmax_data_cuts_analysis_fit": float(max(za)),
        "largest_excess": {"window": best["window"], "eff": best["eff"], "z": best["z_converged"]},
        "fit_dz_max_abs": float(np.abs(dz).max()),
        "fit_dz_mean": float(dz.mean()),
        "fit_dz_sd": float(dz.std(ddof=1)),
        "toys": null_toys(mjj, score, n_toys, np.random.default_rng(seed)) if n_toys > 0 else None,
    }


def _meta() -> dict[str, Any]:
    return {k: v for k, v in run_metadata().items() if k in ("git", "time", "splits_sha256", "torch")}


def rnd_scan(run: str, n_toys: int = 1000, data=None, log=print, chunk: int = 20000) -> dict[str, Any]:
    """The BB1 train-on-data bump hunt without signal (module docstring).  Jet scores of all R&D background
    events are cached in ``<run>/scores_rnd_all.npz`` (ignored by git)."""
    from gaehep.eval.evaluate import aggregate, jet_reference
    from gaehep.train.run import Data

    model, cfg = _load_model(run)
    data = data or Data.from_config(cfg.get("data"))
    rdir = results_dir() / "runs" / run
    cache = rdir / "scores_rnd_all.npz"
    splits = ("bkg_train", "bkg_val", "bkg_test")
    have = dict(np.load(cache)) if cache.exists() else {}
    for sp in splits:
        if sp not in have:
            log(f"[rnd-scan] {run}: scoring {sp}")
            have[sp] = _jet_scores(model, data, data.events(sp), chunk)
            np.savez(cache, **have)
    ref = jet_reference({"bkg_val": have["bkg_val"]})
    mjj = np.concatenate([np.asarray(data.store("rnd")["mjj"][data.events(sp)], np.float64) for sp in splits])
    score = aggregate(np.concatenate([have[sp] for sp in splits]), "prod", ref)
    res = _scan(mjj, score, n_toys, zlib.crc32(run.encode()))
    out = {
        "name": run,
        "sample": "rnd background (bkg_train + bkg_val + bkg_test)",
        "n_events": len(mjj),
        "agg": "prod",
        **res,
        **_meta(),
    }
    save_json(out, rdir / "rnd_scan.json")
    toys = res["toys"] or {"zmax_scan_cuts": [float("nan")]}
    log(
        f"[rnd-scan] {run}: data max Z (10 %, 2 % cuts) {out['zmax_data_cuts']:+.2f}; toys: median max Z "
        f"{np.median(toys['zmax_scan_cuts']):.2f}; converged - analysis fit |dZ| <= {out['fit_dz_max_abs']:.3f}"
    )
    return out


# ------------------------------------------------------------------------- black-box commands (session F2)
def _reference(run: str, model=None, data=None, n_events: int | None = None) -> np.ndarray:
    """bkg_val jet scores of the run's training sample (F_B of the ``prod`` score): from ``<run>/scores.npz``
    when it has them, else scored now (all of bkg_val, or ``n_events`` of it)."""
    from gaehep.train.run import Data

    p = results_dir() / "runs" / run / "scores.npz"
    if p.exists():
        with np.load(p) as f:
            if "bkg_val" in f.files:
                return f["bkg_val"]
    if model is None:
        model, cfg = _load_model(run)
        data = data or Data.from_config(cfg.get("data"))
    ev = data.events("bkg_val", n_events)
    return _jet_scores(model, data, ev, store=data.store(data.sample_of("bkg_val")))


def prepare_run(run: str, log=print) -> None:
    """Make a committed run scoreable on the black boxes in a fresh session: copy ``results/models/<run>/model.pt``
    into the run directory and write ``<run>/scores.npz`` with the bkg_val jet scores (all that
    ``gaehep score-sample`` reads from it), unless they are already there."""
    import shutil

    rdir = results_dir() / "runs" / run
    src = results_dir() / "models" / run / "model.pt"
    if not (rdir / "model.pt").exists():
        if not src.exists():
            raise SystemExit(f"prepare-run: no weights for {run} in {src.parent}")
        shutil.copy2(src, rdir / "model.pt")
    p = rdir / "scores.npz"
    if p.exists() and "bkg_val" in np.load(p).files:
        log(f"[prepare-run] {run}: ready")
        return
    np.savez(p, bkg_val=_reference(run))
    log(f"[prepare-run] {run}: wrote {p}")


def bb_scan(run: str, sample: str, n_toys: int = 1000, log=print) -> dict[str, Any]:
    """The window scan + toys of ``rnd_scan`` on a black box, from ``<run>/<sample>_scores.npz`` of all its events
    (``gaehep score-sample RUN --sample S`` without ``--subsample``).  For a BB1-trained run on BB1 this is the
    blind version of the E3 hunt (does its largest excess sit at the true window?); for a BB2-trained run on BB2
    it is the procedure-matched null test.  Writes ``<run>/<sample>_scan.json``."""
    from gaehep.eval.evaluate import aggregate, jet_reference

    rdir = results_dir() / "runs" / run
    with np.load(rdir / f"{sample}_scores.npz") as f:
        js, mjj, label, sub = f["jet_scores"], f["mjj"].astype(np.float64), f["label"], bool(f["subsample"])
    if sub:
        raise SystemExit(
            f"bb-scan: {run}/{sample}_scores.npz is the subsample; score all events (no --subsample)"
        )
    score = aggregate(js, "prod", jet_reference({"bkg_val": _reference(run)}))
    res = _scan(mjj, score, n_toys, zlib.crc32(f"{run}:{sample}".encode()))
    out = {"name": run, "sample": sample, "n_events": len(mjj), "agg": "prod", **res, **_meta()}
    if sample in BLACKBOX_WINDOWS:  # truth, for the report only: where the signal is and how much a cut keeps
        w = BLACKBOX_WINDOWS[sample]
        out["true_window"] = list(w)
        out["true_window_rows"] = [r for r in res["rows"] if r["window"] == list(w)]
        out["n_signal_in_true_window"] = int(np.sum((mjj >= w[0]) & (mjj < w[1]) & (label > 0)))
    save_json(out, rdir / f"{sample}_scan.json")
    log(f"[bb-scan] {run} on {sample}: largest excess {res['largest_excess']}")
    return out


MASS_WINDOWS = ((650.0, 820.0), (320.0, 440.0))  # heavier and lighter jet, around m_X = 732 and m_Y = 378 GeV


def bb1_mass_check(run: str, n_ref_events: int = 20000, log=print) -> dict[str, Any]:
    """BB1 AUC at fixed jet masses: all BB1 events in the 3.6-4.0 TeV window are scored, and the event AUC is
    computed in the window and among the events whose heavier and lighter jets both fall in ``MASS_WINDOWS``
    (where the masses are nearly fixed, so a mass tagger is ~0.5).  F_B from ``n_ref_events`` events of the run's
    bkg_val (or ``<run>/scores.npz``).  References in the same events: the summed jet mass and minus the summed
    tau21 (computed tau, when the store has it).  Writes ``<run>/bb1_mass.json``."""
    from gaehep.eval.evaluate import aggregate, jet_reference
    from gaehep.eval.metrics import auc as _auc
    from gaehep.train.run import Data

    def auc(sig: np.ndarray, bkg: np.ndarray) -> float | None:  # None when a class has no events
        return _auc(sig, bkg) if len(sig) and len(bkg) else None

    model, cfg = _load_model(run)
    data = Data.from_config(cfg.get("data"))
    ref = jet_reference({"bkg_val": _reference(run, model, data, n_events=n_ref_events)})
    st = data.store("bb1")
    ev = data.splits["bb1_test"]
    m = np.asarray(st["mjj"][ev])
    w0, w1 = BLACKBOX_WINDOWS["bb1"]
    ev = ev[(m >= w0) & (m < w1)]
    score = aggregate(_jet_scores(model, data, ev, store=st), "prod", ref)
    sig = np.asarray(st["label"][ev]) > 0
    jm = np.asarray(st["m"][ev], np.float64)
    hi, lo = jm.max(1), jm.min(1)
    (a0, a1), (b0, b1) = MASS_WINDOWS
    inm = (hi >= a0) & (hi < a1) & (lo >= b0) & (lo < b1)
    refs = {"m_J sum": jm.sum(1)}
    if "tau1c" in st.table:
        t21 = np.asarray(st["tau2c"][ev], np.float64) / np.maximum(
            np.asarray(st["tau1c"][ev], np.float64), 1e-12
        )
        refs["-tau21 sum"] = -t21.sum(1)
    out = {
        "name": run,
        "window": [w0, w1],
        "mass_windows": [list(x) for x in MASS_WINDOWS],
        "n_sig_window": int(sig.sum()),
        "n_bkg_window": int((~sig).sum()),
        "n_sig_mass_window": int((sig & inm).sum()),
        "n_bkg_mass_window": int((~sig & inm).sum()),
        "auc_window": auc(score[sig], score[~sig]),
        "auc_mass_window": auc(score[sig & inm], score[~sig & inm]),
        "ref_auc_window": {k: auc(v[sig], v[~sig]) for k, v in refs.items()},
        "ref_auc_mass_window": {k: auc(v[sig & inm], v[~sig & inm]) for k, v in refs.items()},
        "n_ref_events": n_ref_events,
        **_meta(),
    }
    save_json(out, results_dir() / "runs" / run / "bb1_mass.json")
    log(
        f"[bb1-mass] {run}: AUC window {_f(out['auc_window'], '{:.3f}')}, at fixed jet masses "
        f"{_f(out['auc_mass_window'], '{:.3f}')}"
    )
    return out


# -------------------------------------------------------------------------------- 6 look-elsewhere
def _global(zmax: np.ndarray, z: float, zfix: np.ndarray | None = None) -> dict[str, float]:
    """Global p (and Z) of a largest local excess ``z``: the fraction of toy scans with a larger maximum.  With
    ``zfix`` (the toys' local Z at the fixed window) also the calibrated local p: the quoted local Z ignores the
    uncertainty of the fitted background, so its null distribution is wider than N(0, 1)."""
    p = float((np.sum(zmax >= z) + 1) / (len(zmax) + 1))
    out = {"z_local": float(z), "p_global": p, "z_global": float(stats.norm.isf(p))}
    if zfix is not None:
        pl = float((np.sum(zfix >= z) + 1) / (len(zfix) + 1))
        out.update({"p_local_calibrated": pl, "z_local_calibrated": float(stats.norm.isf(pl))})
    return out


def lee_summary(root: Path, null_runs: Sequence[str]) -> dict[str, Any]:
    scans = {}
    for r in null_runs:
        p = root / "runs" / r / "rnd_scan.json"
        if p.exists():
            scans[r] = load_json(p)
    if not scans:
        return {}
    zmax = np.concatenate([np.asarray(s["toys"]["zmax_scan_cuts"]) for s in scans.values()])
    zmax_all = np.concatenate([np.asarray(s["toys"]["zmax_scan_all"]) for s in scans.values()])
    zfix = np.concatenate([np.asarray(s["toys"]["z_bb1_window_2pct"]) for s in scans.values()])
    bb1 = {}
    for enc in ("edgeconv", "ewmp", "efn"):
        for sd in (1, 2, 3):
            p = root / "runs" / f"study_bb1data_{enc}_s{sd}" / "bb1_bumphunt.json"
            if p.exists():
                rows = load_json(p)["rows"]
                bb1[f"{enc}_s{sd}"] = {f"{r['eff']:g}": float(r["z"]) for r in rows}
    per_enc = {}
    for enc in ("edgeconv", "ewmp", "efn"):
        z2 = [v["0.02"] for k, v in bb1.items() if k.startswith(enc + "_")]
        if z2:
            per_enc[enc] = _global(zmax, float(np.mean(z2)), zfix)
    return {
        "null_runs": list(scans),
        "n_toys": len(zmax),
        "data": {
            r: {
                "zmax_cuts": s["zmax_data_cuts"],
                "zmax_all": s["zmax_data_all"],
                "zmax_cuts_analysis_fit": s["zmax_data_cuts_analysis_fit"],
                "fit_dz_max_abs": s["fit_dz_max_abs"],
                "fit_dz_mean": s["fit_dz_mean"],
                "fit_dz_sd": s["fit_dz_sd"],
            }
            for r, s in scans.items()
        },
        "toy_quantiles_zmax_cuts": {q: float(np.quantile(zmax, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "toy_quantiles_zmax_all": {q: float(np.quantile(zmax_all, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "toy_fixed_window_2pct": {"mean": float(zfix.mean()), "sd": float(zfix.std(ddof=1))},
        "local_to_global": [_global(zmax, z, zfix) for z in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)],
        "bb1_runs": {k: {**v, **{"global_2pct": _global(zmax, v["0.02"], zfix)}} for k, v in bb1.items()},
        "bb1_per_encoder_mean_2pct": per_enc,
    }


def bb_scan_summary(root: Path) -> dict[str, Any]:
    """BB1 scans of the BB1-trained runs and the BB2 scans of BB2-trained runs (session F2), if present."""
    out: dict[str, Any] = {}
    for key, pattern, sample in (("bb1", "study_bb1data_", "bb1"), ("bb2_null", "study_bb2data_", "bb2")):
        rows = {}
        for d in sorted((root / "runs").glob(f"{pattern}*")):
            p = d / f"{sample}_scan.json"
            if not p.exists():
                continue
            s = load_json(p)
            e = {"largest_excess": s["largest_excess"], "zmax_cuts": s["zmax_data_cuts"]}
            if s.get("toys"):
                e["toy_median_zmax"] = float(np.median(s["toys"]["zmax_scan_cuts"]))
                e["global"] = _global(np.asarray(s["toys"]["zmax_scan_cuts"]), s["zmax_data_cuts"])
            if s.get("true_window_rows"):
                e["z_true_window"] = {f"{r['eff']:g}": r["z_converged"] for r in s["true_window_rows"]}
            rows[d.name] = e
        if rows:
            out[key] = rows
    return out


def bb1_mass_summary(main: dict[str, Config], curves: dict[str, Config]) -> dict[str, Any]:
    rows = []
    for c in [*main.values(), *curves.values()]:
        for r in c.runs:
            p = r.dir / "bb1_mass.json"
            if p.exists() and r.best_val_loss is not None:
                j = load_json(p)
                rows.append(
                    {
                        "run": r.name,
                        "encoder": c.label.split("_e")[0],
                        "loss": float(r.best_val_loss),
                        **{
                            k: j[k]
                            for k in (
                                "auc_window",
                                "auc_mass_window",
                                "n_sig_mass_window",
                                "n_bkg_mass_window",
                                "ref_auc_mass_window",
                                "ref_auc_window",
                            )
                        },
                    }
                )
    if len(rows) < 3:
        return {"n_runs": len(rows)}
    enc = np.array([r["encoder"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    out: dict[str, Any] = {"n_runs": len(rows), "runs": rows}
    for k in ("auc_window", "auc_mass_window"):
        v = np.array([r[k] for r in rows])
        rho, pv = stats.spearmanr(v, loss)
        out[f"{k}_vs_loss"] = {"spearman": float(rho), "p": float(pv), "n": len(v)}
        out[f"{k}_vs_loss_within_encoder"] = within_group_test(loss, v, enc)
    out["per_encoder"] = {
        e: {
            k: float(np.mean([r[k] for r in rows if r["encoder"] == e]))
            for k in ("auc_window", "auc_mass_window", "loss")
        }
        | {"n": int(np.sum(enc == e))}
        for e in ENCODERS
        if np.any(enc == e)
    }
    out["references"] = {
        "window": rows[0]["ref_auc_window"],
        "mass_window": rows[0]["ref_auc_mass_window"],
        "n_sig_mass_window": rows[0]["n_sig_mass_window"],
        "n_bkg_mass_window": rows[0]["n_bkg_mass_window"],
    }
    return out


# ------------------------------------------------------------------------------------------- report
def _f(x: float | None, fmt: str = "{:+.4f}") -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def _m(b: bool | None) -> str:
    return {True: "yes", False: "no", None: "–"}[b]


def render(res: dict[str, Any]) -> str:
    L = [
        "# Bias-audit follow-up (post hoc, Deviations D9)",
        "",
        "Generated by `gaehep audit-followup` (src/gaehep/eval/followup.py). Every analysis here was added after "
        "all study results were seen, to test the claims of the results draft; none replaces a pre-registered "
        "verdict. Signals: R&D 3-prong (rd3), BB1, BB3.",
        "",
        "## 1. How much the fit-quality curve explains",
        "",
        "Curve: quadratic in the background validation loss, fitted to the E1 + E2 runs (seven encoders, 1, 2, 3 "
        "and 5 epochs). `R² curve`: share of the AUC variance between these runs that the curve explains; `R² "
        "configuration`: share explained by knowing only each run's configuration (encoder × training length, the "
        "mean of its seeds); the leave-one-out versions predict each run by the curve refitted without it and by "
        "the mean of the other seeds of its configuration (the fair comparison: the configuration means are 28 "
        "parameters); `configuration on residual`: share of the curve's residual variance that the "
        "configuration still explains. Within-configuration test: loss and AUC with every configuration's mean "
        "removed (seeds change the fit at fixed architecture and training length). Out of sample: the E4 ablation "
        "runs, which the curve was not fitted to; R² < 0 means the curve predicts them worse than their mean.",
        "",
        "| signal | runs | AUC sd | R² curve | R² configuration | R² curve, leave one out | R² configuration, leave "
        "one out | configuration on residual | within-config ρ (p) "
        "| without Deep Sets ρ (p) | E4 runs | E4 AUC sd | E4 RMSE | E4 R² |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s, e in res["curve"].items():
        w, w2, o = (
            e["within_configuration"],
            e["within_configuration_no_deepsets"],
            e.get("out_of_sample", {}),
        )
        L.append(
            f"| {s} | {e['n_runs']} | {e['auc_sd']:.4f} | {e['r2_curve']:.2f} | {e['r2_configuration']:.2f} | "
            f"{e['r2_curve_loo']:.2f} | {e['r2_configuration_loo']:.2f} | "
            f"{e['r2_configuration_on_residual']:.2f} | {w['spearman']:+.2f} ({w['p']:.1g}) | "
            f"{w2['spearman']:+.2f} ({w2['p']:.1g}) | {o.get('n_runs', '–')} | {_f(o.get('auc_sd'), '{:.4f}')} | "
            f"{_f(o.get('rmse'), '{:.4f}')} | {_f(o.get('r2'), '{:+.2f}')} |"
        )
    L += [
        "",
        "## 2. Contrasts at equal fit: functional form and curve uncertainty",
        "",
        "Δ = mean AUC residual of A − B over the E1 seeds. `*`: |Δ| > 2 SE (Welch, curve fixed). `SE boot`: "
        "bootstrap over runs resampled within configurations, refitting the curve each time (2000 replicas). The "
        "95 % interval uses the larger of the two SEs.",
        "",
        "| signal | contrast | " + " | ".join(CURVE_FORMS) + " | SE Welch | SE boot | 95 % interval |",
        "|---|---|" + "---|" * len(CURVE_FORMS) + "---|---|---|",
    ]
    for s, rows in res["equal_fit"].items():
        for r in rows:
            cells = " | ".join(
                f"{r['forms'][f]['delta']:+.4f}{'*' if r['forms'][f]['sig2se'] else ''}" for f in CURVE_FORMS
            )
            tag = "" if r["role"] == "primary" else " (sec.)"
            L.append(
                f"| {s} | {r['hypothesis']}{tag} {r['a']} − {r['b']} | {cells} | {r['se_welch']:.4f} | "
                f"{r['se_boot']:.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] |"
            )
    L += [
        "",
        "## 3. Inputs × pooling at equal fit",
        "",
        "Set family: Deep Sets (unsafe inputs, mean + max), Deep Sets with pT-weighted pooling (E4), EFN with mean "
        "+ max pooling (E5), EFN (unsafe = the 7 standard node features; safe = (Δη, Δφ, ΔR) only). Graph family: "
        "EdgeConv, EdgeConv with weighted pooling (E4), EWMP with mean + max pooling (E5), EWMP; EWMP also differs "
        "from EdgeConv in the graph (radius vs k-NN) and the aggregation (energy-weighted sum vs max). Residual = "
        "AUC minus the E1 + E2 curve at the run's loss, mean ± SE over seeds. ρ: score stability under the "
        "collinear split of the hardest constituent / ten soft emissions of 1 % pT each (1.000 = exactly invariant; splitting every constituent is no test of mean + max pooling, which it leaves unchanged).",
        "",
        "| signal | family | cell (inputs/pooling) | runs | loss | AUC | residual | ρ split top-1 | ρ soft 1 % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s, fams in res["decomposition"].items():
        for fam, e in fams.items():
            for cell, c in e["cells"].items():
                L.append(
                    f"| {s} | {fam} | {cell} ({c['run_prefix']}_{c['config']}) | {c['n']} | {c['loss']:.3f} | "
                    f"{c['auc']:.4f} ± {_f(c['auc_sd'], '{:.4f}')} | {c['resid']:+.4f} ± "
                    f"{_f(c['resid_se'], '{:.4f}')} | {_f(c['irc_rho'].get('split top-1'), '{:.3f}')} | "
                    f"{_f(c['irc_rho'].get('10 soft (1% pT each)'), '{:.3f}')} |"
                )
    L += [
        "",
        "Effects at equal fit (A − B of the cell residuals, Welch SE); the last columns give Δ for the other curve "
        "forms (`*`: |Δ| > 2 SE).",
        "",
        "| signal | family | effect at equal fit | Δ | SE | significant | "
        + " | ".join(CURVE_FORMS[1:])
        + " |",
        "|---|---|---|---|---|---|" + "---|" * (len(CURVE_FORMS) - 1),
    ]
    for s, fams in res["decomposition"].items():
        for fam, e in fams.items():
            bf = e.get("effects_by_form", {})
            for k, v in e["effects_at_equal_fit"].items():
                alt = " | ".join(
                    f"{bf[f][k]['delta']:+.4f}{'*' if bf[f][k]['sig2se'] else ''}"
                    if f in bf and k in bf[f]
                    else "–"
                    for f in CURVE_FORMS[1:]
                )
                L.append(
                    f"| {s} | {fam} | {k} | {v['delta']:+.4f} | {_f(v['se'], '{:.4f}')} | {_m(v['sig2se'])} | {alt} |"
                )
    irc = res["irc"]
    L += [
        "",
        "## 4. IRC robustness: IRC-safe minus IRC-unsafe",
        "",
        "Spearman ρ of the background jet scores before and after the perturbation (bkg_val jets in the signal "
        "region), mean over the five E1 seeds; Δ = safe − unsafe, Welch SE.",
        "",
        "| contrast | probe | ρ safe | ρ unsafe | Δ | SE | significant |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in irc["contrasts"]:
        ea, eb = irc["encoders"][r["a"]][r["probe"]], irc["encoders"][r["b"]][r["probe"]]
        L.append(
            f"| {r['a']} − {r['b']} | {r['probe']} | {ea['mean']:.3f} ± {_f(ea['sd'], '{:.3f}')} | "
            f"{eb['mean']:.3f} ± {_f(eb['sd'], '{:.3f}')} | {r['delta']:+.3f} | {_f(r['se'], '{:.3f}')} | "
            f"{_m(r['sig2se'])} |"
        )
    mm = res["mass"]
    L += ["", "## 5. Does a worse fit make the score more mass-like?", ""]
    if mm.get("n_runs", 0) >= 5:
        a, b = mm["rho_mass_vs_loss"], mm["rho_mass_vs_loss_within_encoder"]
        c, d = mm["auc_bb1_vs_rho_mass"], mm["auc_bb1_vs_rho_mass_within_encoder"]
        L += [
            f"ρ_mass = Spearman correlation of the jet score with m_J on 10 000 R&D bkg_test jets per run, for "
            f"{mm['n_runs']} E1 + E2 runs. If a poorly fitting model behaved like a mass tagger, ρ_mass would rise "
            "with the validation loss and predict the BB1 AUC.",
            "",
            "| test | Spearman | p | runs |",
            "|---|---|---|---|",
            f"| ρ_mass vs loss, all runs | {a['spearman']:+.2f} | {a['p']:.2g} | {a['n']} |",
            f"| ρ_mass vs loss, within encoder | {b['spearman']:+.2f} | {b['p']:.2g} | {b['n']} |",
            f"| BB1 AUC vs ρ_mass, all runs | {c['spearman']:+.2f} | {c['p']:.2g} | {c['n']} |",
            f"| BB1 AUC vs ρ_mass, within encoder | {d['spearman']:+.2f} | {d['p']:.2g} | {d['n']} |",
            "",
            "| encoder | runs | mean loss | mean ρ_mass |",
            "|---|---|---|---|",
        ]
        L += [
            f"| {e} | {v['n']} | {v['loss']:.3f} | {v['rho_mass']:+.3f} |"
            for e, v in mm["per_encoder"].items()
        ]
    else:
        L.append("No `mass_corr.json` files yet (`gaehep mass-corr RUN...`).")
    lee = res["lee"]
    L += ["", "## 6. Look-elsewhere calibration of the BB1 bump hunt", ""]
    if lee:
        q = lee["toy_quantiles_zmax_cuts"]
        L += [
            "The BB1 hunt (E3) trains on BB1's own events and tests the known window (3.6–4.0 TeV) with cuts keeping "
            "10 % and 2 % of the sideband events. Here the same procedure is run without signal: R&D-trained runs "
            f"({', '.join(lee['null_runs'])}) scan their own training sample, the whole R&D background "
            f"(1 000 000 events), over {len(SCAN_WINDOWS)} windows of 400 GeV (lower edges 3.0–4.3 TeV) × the "
            f"same cuts. {lee['n_toys']} toy experiments from smooth fits of the selected spectra give the "
            "distribution of the largest local Z of the scan (windows × {10 %, 2 %}). The toys and the "
            "`converged` columns minimise the likelihood fully (L-BFGS-B then BFGS, started at the analysis "
            "optimum); the analysis fit of `bumphunt.search` (Nelder–Mead then Powell) can stop short of the "
            "minimum, and the last columns give how much that moves the local Z of these spectra (42 tests per run).",
            "",
            "| null run | largest local Z, data scan (10 %, 2 %), converged | (all three cuts) | (10 %, 2 %), "
            "analysis fit | converged − analysis ΔZ: mean ± sd | max abs |",
            "|---|---|---|---|---|---|",
        ]
        L += [
            f"| {r} | {v['zmax_cuts']:+.2f} | {v['zmax_all']:+.2f} | {v['zmax_cuts_analysis_fit']:+.2f} | "
            f"{v['fit_dz_mean']:+.3f} ± {v['fit_dz_sd']:.3f} | {v['fit_dz_max_abs']:.3f} |"
            for r, v in lee["data"].items()
        ]
        fx = lee["toy_fixed_window_2pct"]
        L += [
            "",
            f"Toys: largest local Z of the scan, median {q[0.5]:.2f}, 90 % {q[0.9]:.2f}, 95 % {q[0.95]:.2f}, 99 % "
            f"{q[0.99]:.2f}. At the fixed BB1 window alone (2 % cut) the toys give Z = {fx['mean']:+.2f} ± "
            f"{fx['sd']:.2f}: without signal the quoted local Z is wider than N(0, 1), because it ignores the "
            "uncertainty of the background extrapolated from the sideband fit. The calibrated local p is the "
            "fraction of toys with a larger Z at that fixed window.",
            "",
            "| quoted local Z | calibrated local p (fixed window) | calibrated local Z | global p (scan) | global Z |",
            "|---|---|---|---|---|",
        ]
        L += [
            f"| {g['z_local']:.1f} | {g['p_local_calibrated']:.3g} | {g['z_local_calibrated']:.2f} | "
            f"{g['p_global']:.3g} | {g['z_global']:.2f} |"
            for g in lee["local_to_global"]
        ]
        L += [
            "",
            "| BB1-trained run | quoted local Z (2 % cut) | calibrated local Z | global p | global Z |",
            "|---|---|---|---|---|",
        ]
        L += [
            f"| {k} | {v['0.02']:+.2f} | {v['global_2pct']['z_local_calibrated']:.2f} | "
            f"{v['global_2pct']['p_global']:.3g} | {v['global_2pct']['z_global']:.2f} |"
            for k, v in lee["bb1_runs"].items()
        ]
        L += [
            "",
            "| encoder | mean quoted local Z (2 % cut) | calibrated local Z | global p | global Z |",
            "|---|---|---|---|---|",
        ]
        L += [
            f"| {k} | {v['z_local']:+.2f} | {v['z_local_calibrated']:.2f} | {v['p_global']:.3g} | {v['z_global']:.2f} |"
            for k, v in lee["bb1_per_encoder_mean_2pct"].items()
        ]
        L += [
            "",
            "The global values assume that BB1's own scan would peak at the true window; a larger excess elsewhere "
            "would make the hunt point to the wrong mass, so they are an upper bound on what a blind hunt achieves.",
        ]
    else:
        L.append("No `rnd_scan.json` files yet (`gaehep rnd-scan RUN...`).")
    bbs = res.get("bb_scans") or {}
    if bbs:
        L += [
            "",
            "### Black-box scans (session F2)",
            "",
            "BB1-trained runs scanned blind on BB1, and BB2-trained runs on BB2 (the procedure-matched null), with "
            "the windows, cuts and converged fit above; global p from each run's own toys.",
            "",
            "| sample | run | largest excess (window, cut, Z) | Z at the true window (10 %, 2 %) | toys: median largest Z "
            "| global p | global Z |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, rows in bbs.items():
            for r, e in rows.items():
                le = e["largest_excess"]
                zt = e.get("z_true_window", {})
                g = e.get("global", {})
                L.append(
                    f"| {key} | {r} | {le['window'][0] / 1000:.1f}–{le['window'][1] / 1000:.1f} TeV, {100 * le['eff']:g} %, "
                    f"{le['z']:+.2f} | {_f(zt.get('0.1'), '{:+.2f}')}, {_f(zt.get('0.02'), '{:+.2f}')} | "
                    f"{_f(e.get('toy_median_zmax'), '{:.2f}')} | {_f(g.get('p_global'), '{:.3g}')} | "
                    f"{_f(g.get('z_global'), '{:.2f}')} |"
                )
    bm = res.get("bb1_mass") or {}
    L += ["", "## 7. BB1 at fixed jet masses (session F2)", ""]
    if bm.get("n_runs", 0) >= 3:
        rf = bm["references"]
        L += [
            f"Event AUC in the BB1 window (3.6–4.0 TeV) and among the events whose heavier jet has 650–820 GeV and "
            f"lighter jet 320–440 GeV ({rf['n_sig_mass_window']} signal, {rf['n_bkg_mass_window']} background "
            "events), where a mass tagger is near 0.5. References in the same events: "
            + ", ".join(f"{k} {v:.3f}" for k, v in rf["mass_window"].items())
            + ".",
            "",
            "| encoder | runs | mean loss | AUC in the window | AUC at fixed jet masses |",
            "|---|---|---|---|---|",
        ]
        L += [
            f"| {e} | {v['n']} | {v['loss']:.3f} | {v['auc_window']:.3f} | {v['auc_mass_window']:.3f} |"
            for e, v in bm["per_encoder"].items()
        ]
        a, b = bm["auc_mass_window_vs_loss"], bm["auc_mass_window_vs_loss_within_encoder"]
        L += [
            "",
            f"AUC at fixed jet masses vs validation loss: Spearman {a['spearman']:+.2f} (p {a['p']:.2g}) over all runs, "
            f"{b['spearman']:+.2f} (p {b['p']:.2g}) within encoder.",
        ]
    else:
        L.append("No `bb1_mass.json` files yet (`gaehep bb1-mass-check RUN...`, needs the black boxes).")
    return "\n".join(L) + "\n"


def followup(
    root: Path | None = None,
    log=print,
    null_runs: Sequence[str] = ("study_efn_s1", "study_efn_s2", "study_efn_s3"),
) -> dict[str, Path]:
    root = Path(root) if root else results_dir()
    main = _main_configs(root)
    curves = _cfgs("stct", root)
    abl = _cfgs("stat", root)
    e5 = _cfgs("e5", root)
    res = {
        "curve": curve_checks(main, curves, abl),
        "equal_fit": equal_fit_robust(main, curves),
        "decomposition": decomposition(main, curves, {"study": main, "stat": abl, "e5": e5}),
        "irc": irc_contrasts(main),
        "mass": mass_mechanism(main, curves),
        "lee": lee_summary(root, null_runs),
        "bb_scans": bb_scan_summary(root),
        "bb1_mass": bb1_mass_summary(main, curves),
    }
    out = root / "study"
    paths = {"md": out / "audit_followup.md", "json": out / "audit_followup.json"}
    paths["md"].write_text(render(res))
    save_json({k: v for k, v in res.items()}, paths["json"])
    for p in paths.values():
        log(f"[audit-followup] wrote {p}")
    return paths
