"""Publication figures for the GAE_HEP paper, read from the committed results.

    python scripts/make_paper_figs.py [OUT]        # default OUT = paper/figures

Writes figN_<name>.pdf (vector, for the journal) and .png (300 dpi) at final print width. The style is in
``gaehep.eval.paper_style``. Error bars are the standard deviation over seeds unless stated otherwise.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gaehep.eval import paper_style as S  # noqa: E402
from gaehep.eval import posthoc as P  # noqa: E402
from gaehep.eval.study import collect  # noqa: E402

ROOT = REPO / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper" / "figures"
S.apply()

study = {c.label: c for c in collect("study", ROOT) if c.runs}
curves = {c.label: c for c in collect("stct", ROOT) if c.runs}
fix = {c.label: c for c in collect("fix", ROOT) if c.runs}
abl = [c for c in collect("stat", ROOT) if c.runs]
ph = json.loads((ROOT / "study" / "study_posthoc.json").read_text())
bb3 = json.loads((ROOT / "study" / "bb3_diagnostics.json").read_text())


def vals(cfg, sig, metric="auc"):
    return np.array(list(P.seed_values(cfg, sig, metric).values()))


def baseline(label, sig):
    """Mean AUC of an F1 baseline (computed τ, D3). Single-observable runs carry no seed."""
    col = P._col(sig, "auc")
    v = [col.get(r) for r in fix[label].runs]
    return float(np.mean([x for x in v if x is not None]))


def points(cs, sig):
    """(validation loss, AUC) of every seed run of the configurations ``cs``."""
    xy = []
    for c in cs:
        v = P.seed_values(c, sig, "auc")
        for r in c.runs:
            sd = P._seed_of(r)
            if sd in v and r.best_val_loss is not None:
                xy.append((r.best_val_loss, v[sd]))
    return np.array(xy).reshape(-1, 2)


written = []

# Figure 1: AUC per encoder -----------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
base = {
    "rd3": [(r"$-\tau_{32}$", baseline("obs_tau32", "rd3")), (r"$m_J$", baseline("obs_mj", "rd3"))],
    "bb1": [(r"$m_J$, $-\tau_{21}$", 0.5 * (baseline("obs_mj", "bb1") + baseline("obs_tau21", "bb1")))],
    "bb3": [
        (r"$m_J$", baseline("obs_mj", "bb3")),
        ("flow", baseline("hldensity", "bb3")),
        ("random", 0.5),
    ],
}
for k, (ax, s) in enumerate(zip(axes, ["rd3", "bb1", "bb3"], strict=True)):
    for i, e in enumerate(S.ENCODERS):
        v = vals(study[e], s)
        ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), elinewidth=0.8, **S.mstyle(e))
    for name, x in base[s]:
        ax.axvline(x, color=S.GREY, ls=(0, (3, 2)), lw=0.7, zorder=0)
        ax.text(x, -0.75, name, color="0.3", fontsize=6.5, ha="center", va="bottom")
    ax.set_ylim(len(S.ENCODERS) - 0.4, -1.35)
    ax.set_xlabel("AUC in the signal window")
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_title(f"({'abc'[k]}) {S.SIGNAL[s]}", loc="left", pad=3)
    for y in (1.5, 3.5, 5.5):
        ax.axhline(y, color="0.85", lw=0.5, zorder=0)
axes[0].set_yticks(range(len(S.ENCODERS)), [S.LABEL[e] for e in S.ENCODERS])
axes[0].set_xlim(0.63, None)
axes[2].set_xlim(0.44, 0.76)
written += S.save(fig, OUT, "fig1_auc_by_encoder")

# Figure 2: AUC against background fit quality ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 2.35), gridspec_kw={"wspace": 0.32})
fitted = {e: [study[e]] + [c for lbl, c in curves.items() if lbl.split("_e")[0] == e] for e in S.ENCODERS}
for k, (ax, s) in enumerate(zip(axes, ["rd3", "bb1", "bb3"], strict=True)):
    allxy = []
    for e in S.ENCODERS:
        xy = points(fitted[e], s)
        allxy.append(xy)
        ax.scatter(xy[:, 0], xy[:, 1], s=9, zorder=3, **S.sstyle(e))
    xy = points(abl, s)
    ax.scatter(xy[:, 0], xy[:, 1], s=9, marker="x", color=S.GREY, linewidths=0.6, zorder=2)
    allxy = np.vstack(allxy)
    coef = np.polyfit(allxy[:, 0], allxy[:, 1], 2)
    xx = np.linspace(allxy[:, 0].min(), allxy[:, 0].max(), 200)
    ax.plot(xx, np.polyval(coef, xx), color="0.2", lw=0.8, zorder=1)
    ax.set_xlabel("Background validation loss")
    ax.set_ylabel("AUC" if k == 0 else None)
    ax.set_title(f"({'abc'[k]}) {S.SIGNAL[s]}", loc="left", pad=3)
handles = [
    *S.encoder_handles(),
    plt.Line2D([], [], marker="x", color=S.GREY, ls="", label="ablations (not fitted)"),
    plt.Line2D([], [], color="0.2", lw=0.8, label="quadratic fit"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.17))
written += S.save(fig, OUT, "fig2_auc_vs_fit")

# Figure 3: training length -----------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.45), gridspec_kw={"wspace": 0.25})
for k, (ax, s) in enumerate(zip(axes, ["rd3", "bb1"], strict=True)):
    for i, e in enumerate(S.ENCODERS):
        xs, ms, ss = [], [], []
        for ep in (1, 2, 3, 5):
            c = study.get(e) if ep == 2 else curves.get(f"{e}_e{ep}")
            if c is None:
                continue
            v = vals(c, s)
            xs.append(ep + 0.05 * (i - 3))
            ms.append(v.mean())
            ss.append(v.std(ddof=1))
        st = S.mstyle(e, linestyle="-" if e in S.IRC_SAFE else (0, (3, 1.5)), lw=0.8)
        ax.errorbar(xs, ms, yerr=ss, elinewidth=0.6, **st)
    ax.set_xticks([1, 2, 3, 5])
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlabel("Training length (epochs)")
    ax.set_ylabel("AUC")
    ax.set_title(f"({'ab'[k]}) {S.SIGNAL[s]}", loc="left", pad=3)
fig.legend(handles=S.encoder_handles(), loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.1))
written += S.save(fig, OUT, "fig3_training_length")

# Figure 4: contrasts as measured and at equal background fit -------------------------------------------------
# raw: the protocol contrasts (study_posthoc.json), +-2 Welch SE; at equal fit: the residual contrasts of the bias-audit
# follow-up (audit_followup.json), with 95 % intervals from the larger of the Welch and the bootstrap SE (the
# bootstrap refits the curve, so it includes the curve's own uncertainty)
fu = json.loads((ROOT / "study" / "audit_followup.json").read_text())
pairs = [
    ("edgeconv", "deepsets"),
    ("ewmp", "efn"),
    ("transformer", "deepsets"),
    ("ew_attention", "efn"),
    ("ewmp", "edgeconv"),
    ("efn", "deepsets"),
    ("ew_attention", "transformer"),
]
fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.4), sharey=True, gridspec_kw={"wspace": 0.08})
for k, (ax, s) in enumerate(zip(axes, ["rd3", "bb1"], strict=True)):
    raw = {(r["a"], r["b"]): r for r in ph["contrasts"] if r["signal_key"] == s and r["metric"] == "auc"}
    adj = {(c["a"], c["b"]): c for c in fu["equal_fit"][s]}
    for i, ab in enumerate(pairs):
        r, q = raw[ab], adj[ab]
        ax.errorbar(
            r["delta"],
            i - 0.14,
            xerr=2 * r["se"],
            marker="s",
            color=S.GREY,
            mfc="white",
            ls="",
            elinewidth=0.7,
        )
        lo, hi = q["ci95"]
        ax.errorbar(
            q["delta"],
            i + 0.14,
            xerr=[[q["delta"] - lo], [hi - q["delta"]]],
            marker="o",
            color=S.BLACK,
            ls="",
            elinewidth=0.7,
        )
    ax.axvline(0, color="0.5", lw=0.6, zorder=0)
    ax.axhline(3.5, color="0.85", lw=0.5, zorder=0)
    ax.set_yticks(range(len(pairs)), [f"{S.LABEL[a]} $-$ {S.LABEL[b]}" for a, b in pairs])
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_xlabel(r"$\Delta$AUC")
    ax.set_title(f"({'ab'[k]}) {S.SIGNAL[s]}", loc="left", pad=3)
axes[1].text(
    1.02, 1 - 2.0 / len(pairs), "interaction", transform=axes[1].transAxes, rotation=90, va="center", fontsize=7
)
axes[1].text(
    1.02,
    1.5 / len(pairs),
    "IRC-safe\ndesign",
    transform=axes[1].transAxes,
    rotation=90,
    va="center",
    fontsize=7,
)
fig.legend(
    handles=[
        plt.Line2D([], [], marker="s", color=S.GREY, mfc="white", ls="", label="as measured ($\\pm 2$ SE)"),
        plt.Line2D([], [], marker="o", color=S.BLACK, ls="", label="at equal background fit (95 % interval)"),
    ],
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.1),
)
written += S.save(fig, OUT, "fig4_equal_fit")

# Figure 5: IRC robustness ------------------------------------------------------------------------------------
probes = [
    ("split all", "(a) Collinear splits", (0.78, 1.01)),
    ("10 soft (0.2% pT each)", "(b) Soft emissions, 0.2 % each", (0.84, 1.005)),
    ("10 soft (1% pT each)", "(c) Soft emissions, 1 % each", (0.4, 1.02)),
]
fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 2.0), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (key, title, xlim) in zip(axes, probes, strict=True):
    for i, e in enumerate(S.ENCODERS):
        rho = np.array([r.irc[key]["rho"] for r in study[e].runs if r.irc and key in r.irc])
        sd = rho.std(ddof=1) if len(rho) > 1 else 0.0
        ax.errorbar(rho.mean(), i, xerr=sd, elinewidth=0.7, **S.mstyle(e))
    ax.set_xlim(*xlim)
    ax.set_ylim(len(S.ENCODERS) - 0.5, -0.5)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_title(title, loc="left", pad=3)
    for y in (1.5, 3.5, 5.5):
        ax.axhline(y, color="0.85", lw=0.5, zorder=0)
axes[0].set_yticks(range(len(S.ENCODERS)), [S.LABEL[e].replace(" (complete graph)", "") for e in S.ENCODERS])
fig.supxlabel(r"Spearman $\rho$ of the background scores before and after", fontsize=8, y=-0.05)
written += S.save(fig, OUT, "fig5_irc_robustness")

# Figure 6: black box 3 by decay mode -------------------------------------------------------------------------
items = [
    ("Encoders", None),
    (r"$m_J$", "auc_m"),
    (r"$n_\mathrm{const}$", "auc_n_const"),
    (r"$p_T$", "auc_pt"),
]
modes = [
    ("bb3_dijet/window", r"dijet mode ($q\bar{q}$)", "0.35"),
    ("bb3_trijet/window", r"trijet mode ($ggg$)", "0.8"),
]
fig, ax = plt.subplots(figsize=(S.SINGLE, 2.0))
w = 0.36
for j, (key, name, shade) in enumerate(modes):
    enc = np.array([bb3["runs"][r]["score"][key] for r in bb3["runs"] if r.startswith("study_")])
    ys = [enc.mean()] + [bb3["observables"][key][f] for _, f in items[1:]]
    xs = np.arange(len(items)) + (j - 0.5) * w
    ax.bar(xs, np.array(ys) - 0.5, bottom=0.5, width=w, color=shade, edgecolor="black", lw=0.5, label=name)
    ax.errorbar(xs[0], ys[0], yerr=enc.std(ddof=1), color="black", elinewidth=0.7, capsize=1.5)
ax.axhline(0.5, color="black", lw=0.6)
ax.set_xticks(range(len(items)), [n for n, _ in items])
ax.tick_params(axis="x", which="both", top=False, bottom=False)
ax.set_xlim(-0.6, len(items) - 0.4)
ax.set_ylim(0.35, 0.9)
ax.set_ylabel("Jet-level AUC")
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
written += S.save(fig, OUT, "fig6_bb3_modes")

print("\n".join(str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in written))
