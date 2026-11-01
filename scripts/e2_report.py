"""Task E2 report (docs/STUDY_PROTOCOL.md, sections 2 and 8): training-length curves and ablations.

Writes to ``results/study/``

* ``curves.png``    SR AUC (``prod``) against the background validation loss (``best_val_loss``), one panel per
                    signal (R&D 2-prong from the dev run's validation split, R&D 3-prong, BB1 window), one colour
                    per encoder, one marker per training length, one point per run;
* ``curves.csv``    the points of the figure (one row per run);
* ``ablations.md``  mean +- standard deviation over seeds of every ablation configuration (validation splits
                    only), with the number of flagged runs.

Runs used: ``stc_<enc>_e<E>_s<s>`` (dev) with their promotions ``stct_...``; E1's ``std_<enc>_s<s>`` (dev, 2
epochs) with ``study_<enc>_s<s>`` when they exist; the ablation dev runs ``stw_``, ``stp_`` and ``stn_``.
Nothing here feeds a design choice: the figure and the table report.

    python scripts/e2_report.py
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

import numpy as np

from gaehep.config import load_json, results_dir
from gaehep.eval.study import FLAG_FACTOR

ENCODERS = ("deepsets", "edgeconv", "transformer", "gcn_complete", "efn", "ewmp", "ew_attention")
COLORS = {  # colour-blind-safe, one per encoder
    "deepsets": "#0072B2",
    "edgeconv": "#D55E00",
    "transformer": "#009E73",
    "gcn_complete": "#7F7F7F",
    "efn": "#CC79A7",
    "ewmp": "#E69F00",
    "ew_attention": "#56B4E9",
}
MARKERS = {1: "v", 2: "o", 3: "s", 5: "^"}
CURVE = re.compile(r"^stc_(?P<enc>[a-z_]+?)_e(?P<ep>\d+)_s(?P<seed>\d+)$")
E1 = re.compile(r"^std_(?P<enc>[a-z_]+?)_s(?P<seed>\d+)$")
SEED = re.compile(r"^(?P<cfg>.+)_s(?P<seed>\d+)$")


def _get(d, *keys):
    for k in keys:
        try:
            d = d[k]
        except (KeyError, TypeError):
            return None
    return d if isinstance(d, (int, float)) and np.isfinite(d) else None


def _json(p: Path):
    try:
        return load_json(p)
    except (OSError, ValueError):
        return None


def curve_points(root: Path) -> list[dict]:
    pts = []
    for d in sorted((root / "dev").glob("*")):
        m = CURVE.match(d.name)
        e1 = E1.match(d.name)
        if not (m or e1) or "+" in d.name:
            continue
        dev = _json(d / "dev_metrics.json")
        if dev is None:
            continue
        if m:
            enc, ep, seed = m["enc"], int(m["ep"]), int(m["seed"])
            prom = root / "runs" / f"stct_{enc}_e{ep}_s{seed}"
        else:
            enc, seed = e1["enc"], int(e1["seed"])
            ep = int(dev.get("epochs") or 2)
            prom = root / "runs" / f"study_{enc}_s{seed}"
            if enc not in ENCODERS or ep != 2:
                continue
        met = _json(prom / "metrics.json")
        bb1 = _json(prom / "bb1_metrics.json")
        pts.append(
            {
                "run": d.name,
                "encoder": enc,
                "epochs": ep,
                "seed": seed,
                "val_loss": _get(dev, "best_val_loss"),
                "auc_rd2_dev": _get(dev, "metrics", "SR/prod", "auc"),
                "auc_rd3": _get(met, "metrics", "event", "prod", "sig3_test/SR", "auc"),
                "auc_bb1": _get(bb1, "metrics", "event", "prod", "bb1/window", "auc"),
            }
        )
    return pts


def plot_curves(pts: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("auc_rd2_dev", "R&D 2-prong (dev: $\\mathrm{sig\\_val}$, SR)"),
        ("auc_rd3", "R&D 3-prong (test, SR)"),
        ("auc_bb1", "BB1 (test subsample, window)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (key, title) in zip(axes, panels, strict=True):
        for enc in ENCODERS:
            for ep, mk in MARKERS.items():
                sel = [p for p in pts if p["encoder"] == enc and p["epochs"] == ep and p[key] is not None]
                sel = [p for p in sel if p["val_loss"] is not None]
                if sel:
                    ax.scatter(
                        [p["val_loss"] for p in sel],
                        [p[key] for p in sel],
                        c=COLORS[enc],
                        marker=mk,
                        s=36,
                        alpha=0.85,
                        edgecolors="k",
                        linewidths=0.4,
                    )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("background validation loss (best epoch)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("event AUC, $\\mathtt{prod}$ score")
    enc_h = [
        plt.Line2D([], [], marker="o", ls="", color=COLORS[e], label=e)
        for e in ENCODERS
        if any(p["encoder"] == e for p in pts)
    ]
    ep_h = [
        plt.Line2D(
            [], [], marker=m, ls="", color="0.4", mec="k", mew=0.4, label=f"{e} epoch{'s' if e > 1 else ''}"
        )
        for e, m in MARKERS.items()
        if any(p["epochs"] == e for p in pts)
    ]
    fig.legend(handles=enc_h + ep_h, loc="center right", fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def ablation_runs(root: Path) -> dict[str, list[dict]]:
    """Dev runs of the ablations, grouped by configuration (the name without the seed)."""
    groups: dict[str, list[dict]] = {}
    for d in sorted((root / "dev").glob("*")):
        if "+" in d.name or not d.name.startswith(("stw_", "stp_", "stn_")):
            continue
        m = SEED.match(d.name)
        dev = _json(d / "dev_metrics.json")
        if not m or dev is None:
            continue
        groups.setdefault(m["cfg"], []).append(
            {
                "run": d.name,
                "loss": _get(dev, "best_val_loss"),
                "auc": _get(dev, "metrics", "SR/prod", "auc"),
                "sic": _get(dev, "metrics", "SR/prod", "max_sic"),
                "rej": _get(dev, "metrics", "SR/prod", "rej@0.3"),
                "full_auc": _get(dev, "metrics", "full/prod", "auc"),
            }
        )
    return groups


def default_runs(root: Path) -> dict[str, list[dict]]:
    """E1's 2-epoch default configuration per encoder (latent 4, mean+max, 50 constituents), if merged."""
    groups: dict[str, list[dict]] = {}
    for d in sorted((root / "dev").glob("std_*")):
        m = E1.match(d.name)
        dev = _json(d / "dev_metrics.json")
        if not m or "+" in d.name or dev is None or m["enc"] not in ENCODERS:
            continue
        groups.setdefault(m["enc"], []).append(
            {
                "run": d.name,
                "loss": _get(dev, "best_val_loss"),
                "auc": _get(dev, "metrics", "SR/prod", "auc"),
                "sic": _get(dev, "metrics", "SR/prod", "max_sic"),
                "rej": _get(dev, "metrics", "SR/prod", "rej@0.3"),
                "full_auc": _get(dev, "metrics", "full/prod", "auc"),
            }
        )
    return groups


def ms(vals: list[float | None], fmt: str) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    sd = statistics.stdev(v) if len(v) > 1 else float("nan")
    return f"{fmt.format(statistics.fmean(v))} ± {fmt.format(sd)}" if len(v) > 1 else fmt.format(v[0])


def flagged(runs: list[dict]) -> list[str]:
    vals = [(r["run"], r["loss"]) for r in runs if r["loss"] is not None]
    if len(vals) < 2:
        return []
    med = statistics.median(v for _, v in vals)
    return [n for n, v in vals if med > 0 and v > FLAG_FACTOR * med]


def _label(cfg: str) -> tuple[str, str, str]:
    """(encoder, ablation, setting) of an ablation configuration name."""
    kind, rest = cfg.split("_", 1)
    for enc in sorted(ENCODERS, key=len, reverse=True):
        if rest.startswith(enc + "_"):
            tail = rest[len(enc) + 1 :]
            if kind == "stw":
                return enc, "latent width", f"d = {tail[1:]}"
            if kind == "stp":
                return enc, "pooling", tail
            return enc, "constituents", f"n = {tail[1:]}"
    return rest, kind, ""


def ablations_md(root: Path) -> str:
    abl = ablation_runs(root)
    dflt = default_runs(root)
    rows = []
    for enc, runs in dflt.items():
        if enc in {"deepsets", "edgeconv", "ewmp"}:
            rows.append((enc, "default", "latent 4, mean+max (EWMP: weighted), n=50", runs))
    for cfg, runs in abl.items():
        enc, kind, setting = _label(cfg)
        rows.append((enc, kind, setting, runs))
    order = {"default": 0, "latent width": 1, "pooling": 2, "constituents": 3}
    rows.sort(key=lambda r: (ENCODERS.index(r[0]) if r[0] in ENCODERS else 99, order[r[1]], _num(r[2]), r[2]))
    lines = [
        "# E2 ablations (validation splits only)",
        "",
        "Descriptive only (protocol H5: no verdict). All runs are 2 epochs of 120 000 jets, seeds 1-3, evaluated with the",
        "development protocol: background `bkg_val` against the 2-prong `sig_val`, `prod` event score, SR. The 2-prong",
        "signal is the tuned one, so these numbers are never evidence for a hypothesis. mean ± std over seeds (sample",
        "standard deviation); `flagged` = runs whose `best_val_loss` exceeds 1.5 × the median of their configuration",
        "(kept in every mean). `default` rows are E1's `std_<enc>_s<seed>` runs (merged by E1; absent if E1 is not merged).",
        "",
        "| encoder | ablation | setting | seeds | best val loss | AUC (SR) | max SIC (SR) | 1/ε_B @ ε_S=0.3 | AUC (full m_jj) | flagged |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for enc, kind, setting, runs in rows:
        fl = flagged(runs)
        lines.append(
            f"| {enc} | {kind} | {setting} | {len(runs)} | {ms([r['loss'] for r in runs], '{:.4f}')} "
            f"| {ms([r['auc'] for r in runs], '{:.3f}')} | {ms([r['sic'] for r in runs], '{:.2f}')} "
            f"| {ms([r['rej'] for r in runs], '{:.1f}')} | {ms([r['full_auc'] for r in runs], '{:.3f}')} "
            f"| {', '.join(fl) if fl else '0'} |"
        )
    return "\n".join(lines) + "\n"


def _num(s: str) -> float:
    m = re.search(r"\d+", s)
    return float(m.group()) if m else 0.0


def main() -> int:
    root = results_dir()
    out = root / "study"
    out.mkdir(parents=True, exist_ok=True)
    pts = curve_points(root)
    if pts:
        plot_curves(pts, out / "curves.png")
        with open(out / "curves.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pts[0]))
            w.writeheader()
            w.writerows(pts)
    (out / "ablations.md").write_text(ablations_md(root))
    print(f"{len(pts)} curve points; wrote {out}/curves.png, curves.csv, ablations.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
