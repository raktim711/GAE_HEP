"""Standard figures: ROC (as rejection vs efficiency), SIC curves, score histograms, sculpting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from gaehep.eval import style
from gaehep.eval.metrics import DEFAULT_MIN_BKG, roc, sic_curve

style.apply()
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rejection(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str = "") -> Path:
    """curves: label -> (sig_scores, bkg_scores).  Plots 1/eps_B vs eps_S (log y)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for label, (s, b) in curves.items():
        fpr, tpr, _ = roc(s, b)
        ok = fpr > 0
        ax.plot(tpr[ok], 1.0 / fpr[ok], label=label)
    ax.plot([0.01, 1], [100, 1], ls=":", color="grey", lw=1, label="random")
    ax.set_yscale("log")
    ax.set_xlabel(r"signal efficiency $\epsilon_S$")
    ax.set_ylabel(r"background rejection $1/\epsilon_B$")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_sic(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    title: str = "",
    min_bkg: int = DEFAULT_MIN_BKG,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for label, (s, b) in curves.items():
        es, _, sic = sic_curve(s, b, min_bkg)
        ax.plot(es, sic, label=label)
    ax.axhline(1.0, ls=":", color="grey", lw=1)
    ax.set_xlabel(r"signal efficiency $\epsilon_S$")
    ax.set_ylabel(r"SIC $=\epsilon_S/\sqrt{\epsilon_B}$")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_score_hist(
    series: Mapping[str, np.ndarray],
    path: Path,
    xlabel: str = "anomaly score",
    title: str = "",
    logy: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    allv = np.concatenate([np.asarray(v).ravel() for v in series.values()])
    lo, hi = np.quantile(allv, [0.001, 0.999])
    bins = np.linspace(lo, hi, 80)
    for label, v in series.items():
        ax.hist(
            np.clip(np.asarray(v).ravel(), lo, hi),
            bins=bins,
            density=True,
            histtype="step",
            lw=1.5,
            label=label,
        )
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_sculpting(
    x_bkg: np.ndarray,
    score_bkg: np.ndarray,
    path: Path,
    effs: tuple[float, ...] = (1.0, 0.1, 0.01),
    xlabel: str = r"$m_{jj}$ [GeV]",
    bins: np.ndarray | None = None,
    title: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ok = np.isfinite(x_bkg)
    x_bkg, score_bkg = x_bkg[ok], score_bkg[ok]
    bins = np.linspace(*np.quantile(x_bkg, [0.001, 0.999]), 60) if bins is None else bins
    for eff in effs:
        sel = score_bkg >= np.quantile(score_bkg, 1 - eff) if eff < 1 else np.ones(len(x_bkg), bool)
        ax.hist(
            x_bkg[sel], bins=bins, density=True, histtype="step", lw=1.5, label=rf"bkg, $\epsilon_B$={eff:g}"
        )
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_history(history: list[dict], path: Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_loss"] for h in history], label="train")
    ax.plot(ep, [h["val_loss"] for h in history], label="val (background)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_bumphunt(rows: list[dict], path: Path, title: str = "") -> Path:
    """Mean local significance (over seeds) vs number of injected signal events, one line per cut."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    effs = sorted({r["eff"] for r in rows}, reverse=True)
    for eff in effs:
        ns = sorted({r["n_inj"] for r in rows})
        z = [np.mean([r["z"] for r in rows if r["eff"] == eff and r["n_inj"] == n]) for n in ns]
        zs = [np.std([r["z"] for r in rows if r["eff"] == eff and r["n_inj"] == n]) for n in ns]
        lab = "no cut" if eff >= 1 else rf"top {100 * eff:g}% (sideband-defined)"
        ax.errorbar(ns, z, yerr=zs, marker="o", capsize=3, label=lab)
    ax.axhline(0, color="grey", lw=1)
    ax.axhline(3, color="grey", lw=1, ls=":")
    ax.set_xlabel("injected signal events (on 300k background)")
    ax.set_ylabel(r"local significance $Z$ in SR")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_weak(res: dict, path: Path, title: str = "") -> Path:
    """IAD max SIC (mean +- std over repetitions) vs injected S/B, one line per representation; the
    supervised classifier on the same features is shown as a dashed horizontal line."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for rep, r in res.items():
        rows = r["iad"]
        sb = sorted({x["s_over_b"] for x in rows})
        mean = [np.mean([x["max_sic"] for x in rows if x["s_over_b"] == v]) for v in sb]
        std = [np.std([x["max_sic"] for x in rows if x["s_over_b"] == v]) for v in sb]
        line = ax.errorbar(100 * np.asarray(sb), mean, yerr=std, marker="o", capsize=3, label=f"IAD: {rep}")
        ax.axhline(r["supervised"]["max_sic"], color=line[0].get_color(), ls="--", lw=1)
    ax.axhline(1, color="grey", lw=1, ls=":")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    sb_all = sorted({100 * x["s_over_b"] for r in res.values() for x in r["iad"]})
    ax.xaxis.set_major_locator(FixedLocator(sb_all))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("injected S/B in the signal region [%]")
    ax.set_ylabel("max SIC (test)")
    ax.set_title(title)
    ax.legend(
        title="dashed: supervised,\nsame features", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10
    )
    return _save(fig, path)
