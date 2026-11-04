"""Minimal figures for the results overview deck, read from the committed GAE_HEP results."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gaehep.eval import posthoc as P
from gaehep.eval.study import collect

matplotlib.rcdefaults()

ROOT = Path("/home/claude/gae_hep/results")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, BG = "#1d2433", "#6b7280", "#e5e7eb", "#fbfbf8"
BLUE, ORANGE = "#2563eb", "#ea7a19"
ENC = ["deepsets", "edgeconv", "transformer", "gcn_complete", "efn", "ewmp", "ew_attention"]
LABEL = {
    "deepsets": "Deep Sets",
    "edgeconv": "EdgeConv",
    "transformer": "Transformer",
    "gcn_complete": "GCN complete",
    "efn": "EFN (IRC-safe)",
    "ewmp": "EWMP (IRC-safe)",
    "ew_attention": "EW attention (IRC-safe)",
}
SAFE = {"efn", "ewmp", "ew_attention"}
COL = {e: (ORANGE if e in SAFE else BLUE) for e in ENC}
SIG = {"rd3": "R&D 3-prong", "bb1": "Black box 1", "bb3": "Black box 3"}

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "font.family": "DejaVu Sans",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
    }
)

study = {c.label: c for c in collect("study", ROOT) if c.runs}
curves = {c.label: c for c in collect("stct", ROOT) if c.runs}
abl = [c for c in collect("stat", ROOT) if c.runs]


def vals(cfg, sig, metric="auc"):
    return np.array(list(P.seed_values(cfg, sig, metric).values()))


# 1. main comparison ------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
base = {
    "rd3": {"−τ32": 0.805, "m_J": 0.646},
    "bb1": {"m_J": 0.918, "−τ21": 0.917},
    "bb3": {"jet-obs flow": 0.735},
}
for ax, s in zip(axes, ["rd3", "bb1", "bb3"], strict=True):
    for i, e in enumerate(ENC):
        v = vals(study[e], s)
        ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), fmt="o", color=COL[e], ms=9, capsize=4, lw=2)
    for j, (name, x) in enumerate(base[s].items()):
        ax.axvline(x, color=MUTED, ls="--", lw=1.3)
        ax.text(
            x + (0.0012 if (s == "bb1" and j == 0) else -0.0012 if s == "bb1" else 0),
            -0.75 - 0.45 * j,
            name,
            color=MUTED,
            fontsize=12,
            ha="center",
            va="bottom",
        )
    if s == "bb3":
        ax.axvline(0.5, color=INK, lw=0.8, alpha=0.4)
        ax.text(0.5, -0.75, "random", color=MUTED, fontsize=12, ha="center", va="bottom")
    ax.set_ylim(len(ENC) - 0.5, -1.4)
    ax.set_title(SIG[s], color=INK, fontsize=17, pad=12)
    ax.set_xlabel("AUC in the signal window")
    ax.grid(axis="x", color=GRID)
axes[0].set_yticks(range(len(ENC)), [LABEL[e] for e in ENC])
fig.tight_layout()
fig.savefig(OUT / "main_auc.png", dpi=150)
plt.close(fig)

# 2. fit-quality curve ----------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, s in zip(axes, ["rd3", "bb1"], strict=True):
    xs_all, ys_all = [], []
    for _group, cs, mk, alpha in (("E1", study.values(), "o", 0.9), ("E2", curves.values(), "^", 0.75)):
        for c in cs:
            enc = c.label.split("_e")[0]
            if enc not in ENC:
                continue
            v = P.seed_values(c, s, "auc")
            for r in c.runs:
                sd = P._seed_of(r)
                if sd in v and r.best_val_loss is not None:
                    ax.scatter(r.best_val_loss, v[sd], color=COL[enc], marker=mk, s=40, alpha=alpha, lw=0)
                    xs_all.append(r.best_val_loss)
                    ys_all.append(v[sd])
    for c in abl:
        v = P.seed_values(c, s, "auc")
        for r in c.runs:
            sd = P._seed_of(r)
            if sd in v and r.best_val_loss is not None:
                ax.scatter(r.best_val_loss, v[sd], facecolor="none", edgecolor=INK, marker="s", s=40, lw=1.2)
    coef = np.polyfit(xs_all, ys_all, 2)
    xx = np.linspace(min(xs_all), max(xs_all), 100)
    ax.plot(xx, np.polyval(coef, xx), color=INK, lw=1.5, alpha=0.6)
    ax.set_xlabel("background validation loss  (← better fit)")
    ax.set_ylabel(f"AUC, {SIG[s]}")
    ax.grid(color=GRID)
h = [
    plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=9, label="IRC-unsafe encoders"),
    plt.Line2D([], [], color=ORANGE, marker="o", ls="", ms=9, label="IRC-safe encoders"),
    plt.Line2D([], [], color=MUTED, marker="^", ls="", ms=9, label="1, 3, 5 epochs (E2)"),
    plt.Line2D(
        [],
        [],
        markerfacecolor="none",
        markeredgecolor=INK,
        marker="s",
        ls="",
        ms=9,
        label="ablations, not fitted (E4)",
    ),
]
axes[1].legend(handles=h, fontsize=12, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fit_curve.png", dpi=150)
plt.close(fig)

# 3. training length ------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
for ax, s in zip(axes, ["rd3", "bb1"], strict=True):
    for e in ENC:
        xs, ms, ss = [], [], []
        for ep in (1, 2, 3, 5):
            c = study.get(e) if ep == 2 else curves.get(f"{e}_e{ep}")
            if c is None:
                continue
            v = vals(c, s)
            xs.append(ep)
            ms.append(v.mean())
            ss.append(v.std(ddof=1))
        pal = {
            "deepsets": "#1f77b4",
            "edgeconv": "#17becf",
            "transformer": "#7f7f7f",
            "gcn_complete": "#9467bd",
            "efn": "#ff7f0e",
            "ewmp": "#d62728",
            "ew_attention": "#8c564b",
        }
        ls = "-" if e in SAFE else "--"
        ax.errorbar(xs, ms, yerr=ss, color=pal[e], ls=ls, marker="o", capsize=3, lw=2, label=LABEL[e])
    ax.set_xticks([1, 2, 3, 5])
    ax.set_xlabel("training length (epochs)")
    ax.set_ylabel(f"AUC, {SIG[s]}")
    ax.grid(color=GRID)
fig.legend(*axes[1].get_legend_handles_labels(), fontsize=12, frameon=False, ncol=4, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT / "training_length.png", dpi=150)
plt.close(fig)

# 4. equal-fit contrasts --------------------------------------------------------------------------------------
import json  # noqa: E402

ph = json.loads((ROOT / "study" / "study_posthoc.json").read_text())
pairs = [
    ("edgeconv", "deepsets", "graph vs none (unsafe)"),
    ("ewmp", "efn", "graph vs none (safe)"),
    ("ewmp", "edgeconv", "safe vs unsafe (graph)"),
    ("efn", "deepsets", "safe vs unsafe (none)"),
    ("ew_attention", "transformer", "safe vs unsafe (attention)"),
]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
for ax, s in zip(axes, ["rd3", "bb1"], strict=True):
    raw = {(r["a"], r["b"]): r for r in ph["contrasts"] if r["signal_key"] == s and r["metric"] == "auc"}
    adj = {(c["a"], c["b"]): c for c in ph["fit_adjusted"][s]["contrasts"]}
    y = np.arange(len(pairs))
    for k, (a, b, _) in enumerate(pairs):
        r, q = raw[(a, b)], adj[(a, b)]
        ax.barh(k + 0.18, r["delta"], height=0.34, color="#c7cdd6", label="as measured" if k == 0 else None)
        ax.barh(
            k - 0.18,
            q["delta"],
            height=0.34,
            color=INK if q["sig2se"] else "#8b93a1",
            label="at equal background fit" if k == 0 else None,
        )
        ax.errorbar(q["delta"], k - 0.18, xerr=2 * q["se"], color=ORANGE, capsize=3, lw=1.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [p[2] for p in pairs])
    ax.invert_yaxis()
    ax.set_title(SIG[s], fontsize=17, color=INK)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel("Δ AUC  (first − second)")
    ax.grid(axis="x", color=GRID)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=13, ncol=2, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "equal_fit.png", dpi=150)
plt.close(fig)

# 5. IRC robustness -------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.4))
keys = [
    ("split all", "split every constituent in two"),
    ("10 soft (1% pT each)", "add 10 soft particles (1 % pT each)"),
]
w = 0.38
for i, e in enumerate(ENC):
    for j, (k, _) in enumerate(keys):
        rho = [r.irc[k]["rho"] for r in study[e].runs if r.irc and k in r.irc]
        ax.bar(i + (j - 0.5) * w, np.mean(rho), width=w, color=COL[e], alpha=1.0 if j == 0 else 0.45)
ax.set_xticks(range(len(ENC)), [LABEL[e].replace(" (IRC-safe)", "\n(IRC-safe)") for e in ENC], fontsize=12)
ax.set_ylabel("rank correlation of scores\nbefore vs after (1 = unchanged)")
ax.set_ylim(0.4, 1.02)
ax.grid(axis="y", color=GRID)
ax.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=1.0, label=keys[0][1]),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.45, label=keys[1][1]),
    ],
    frameon=False,
    fontsize=13,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
)
fig.tight_layout()
fig.savefig(OUT / "irc.png", dpi=150)
plt.close(fig)

# 6. BB3 by mode ----------------------------------------------------------------------------------------------
diag = json.loads((ROOT / "study" / "bb3_diagnostics.json").read_text())
fig, ax = plt.subplots(figsize=(12, 5.4))
modes = [
    ("bb3_dijet/window", "dijet mode\n(KK gluon → q q̄)"),
    ("bb3_trijet/window", "trijet mode\n(KK gluon → g + radion → ggg)"),
]
items = [
    ("encoders (mean of 7)", None),
    ("jet mass", "auc_m"),
    ("multiplicity", "auc_n_const"),
    ("jet pT", "auc_pt"),
]
for i, (key, _) in enumerate(modes):
    enc = np.mean([diag["runs"][r]["score"][key] for r in diag["runs"] if r.startswith("study_")])
    vals_ = [enc] + [diag["observables"][key][k] for _, k in items[1:]]
    for j, v in enumerate(vals_):
        ax.bar(i * 5 + j, v, color=INK if j == 0 else "#9aa3b2")
        ax.text(i * 5 + j, v + 0.01, f"{v:.2f}", ha="center", fontsize=12, color=INK)
ax.axhline(0.5, color=ORANGE, lw=1.5, ls="--")
ax.text(9.4, 0.505, "random", color=ORANGE, fontsize=12, ha="right", va="bottom")
ax.set_xticks(
    [j + 5 * i for i in range(2) for j in range(4)],
    [n for _ in modes for n, _ in items],
    fontsize=11,
    rotation=20,
)
for i, (_, name) in enumerate(modes):
    ax.text(i * 5 + 1.5, 0.98, name, ha="center", fontsize=13, color=INK, va="top")
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("jet-level AUC, 4.0–4.4 TeV window")
ax.grid(axis="y", color=GRID)
fig.tight_layout()
fig.savefig(OUT / "bb3.png", dpi=150)
plt.close(fig)

print("ok", sorted(p.name for p in OUT.iterdir()))
