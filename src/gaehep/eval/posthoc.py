"""Post-hoc robustness of the hypothesis verdicts (docs/STUDY_PROTOCOL.md, Deviations D2).

The protocol's decision rules (section 2) are applied as frozen, and three analyses added after E1 show how far
the verdicts depend on choices that the rules leave open.  None of them replaces a pre-registered verdict.

* ``protocol``  - the frozen rule: |delta| > 2 SE with the unpaired (Welch) SE over seeds; on BB1 and BB3 also
  the paired event bootstrap of the 5-seed ensembles (read from ``<prefix>_contrasts.json``).
* ``paired``    - seed s of every encoder sees the same training jets (``default_rng([seed, epoch])`` in
  train/loop.py), so the seeds are paired: delta and SE from the per-seed differences A_s - B_s.
* ``holm``      - Welch t-test p-values with a Holm-Bonferroni correction over all tests that enter the
  H1-H3 verdicts (alpha = 0.05), plus the black-box bootstrap.
* ``fit``       - the protocol rule after removing low-fit runs: background validation loss above
  ``LOW_FIT`` x the median of the configuration (the protocol flag uses 1.5 and caught none in E1).

H4 (training length) is read from the training-length runs (``stct_<enc>_e<E>_s<s>``) when they exist: every
contrast that is significant under the protocol rule must keep its sign at 1, 3 and 5 epochs.

``gaehep posthoc [--prefix study] [--curves stct]`` writes ``results/study/<prefix>_posthoc.{md,json}`` and
``results/study/figures/<prefix>_auc_vs_valloss.png``.  It needs only the committed metrics and history files.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from gaehep.config import load_json, results_dir, save_json
from gaehep.eval.study import COLUMNS, SIGNALS, Config, Run, collect

LOW_FIT = 1.25
ALPHA = 0.05
UNTUNED = ("rd3", "bb1", "bb3")
ALL_SIGNALS = ("rd2", "rd3", "bb1", "bb3")
VARIANTS = ("protocol", "paired", "holm", "fit")
EPOCHS = (1, 3, 5)

# (hypothesis, role, A, B, metric, signals, better): better = +1 if a larger metric favours A
HYPOTHESES: list[tuple[str, str, str, str, str, tuple[str, ...], int]] = [
    ("H1", "primary", "edgeconv", "deepsets", "auc", UNTUNED, +1),
    ("H1", "primary", "ewmp", "efn", "auc", UNTUNED, +1),
    ("H1", "secondary", "transformer", "deepsets", "auc", UNTUNED, +1),
    ("H1", "secondary", "ew_attention", "efn", "auc", UNTUNED, +1),
    ("H2a", "primary", "ewmp", "edgeconv", "auc", UNTUNED, +1),
    ("H2a", "primary", "efn", "deepsets", "auc", UNTUNED, +1),
    ("H2a", "primary", "ew_attention", "transformer", "auc", UNTUNED, +1),
    ("H2b", "primary", "ewmp", "edgeconv", "jsd", ("rd_bkg",), -1),
    ("H2b", "primary", "efn", "deepsets", "jsd", ("rd_bkg",), -1),
    ("H2b", "primary", "ew_attention", "transformer", "jsd", ("rd_bkg",), -1),
    ("H3", "primary", "gcn_complete", "deepsets", "auc", ALL_SIGNALS, +1),
]
_SIG = {s.short: s for s in SIGNALS}


# ------------------------------------------------------------------------------------------- values
def _col(signal: str, metric: str):
    if metric == "jsd":
        return COLUMNS["jsd"]
    sig = _SIG[signal]
    return COLUMNS[f"{sig.short}_{sig.primary}_{metric}"]


def _seed_of(run: Run) -> int | None:
    tail = run.name.rsplit("_s", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None


def seed_values(
    cfg: Config, signal: str, metric: str, keep: Callable[[Run], bool] | None = None
) -> dict[int, float]:
    """{seed: value} of the configuration's seed runs (runs without a value, or rejected by ``keep``, are left out)."""
    col = _col(signal, metric)
    out = {}
    for r in cfg.runs:
        s = _seed_of(r)
        v = col.get(r)
        if s is not None and v is not None and (keep is None or keep(r)):
            out[s] = v
    return out


def low_fit_runs(cfg: Config, factor: float = LOW_FIT) -> list[str]:
    vals = [(r.name, r.best_val_loss) for r in cfg.runs if r.best_val_loss is not None]
    if len(vals) < 3:
        return []
    med = float(np.median([v for _, v in vals]))
    return [n for n, v in vals if v > factor * med]


# ------------------------------------------------------------------------------------------- tests
def welch(a: list[float], b: list[float]) -> dict[str, Any]:
    """delta, unpaired SE, the 2-SE rule and the Welch two-sided p-value."""
    from scipy import stats

    out: dict[str, Any] = {"n_a": len(a), "n_b": len(b), "delta": None, "se": None, "sig2se": None, "p": None}
    if not a or not b:
        return out
    out["delta"] = float(np.mean(a) - np.mean(b))
    if len(a) > 1 and len(b) > 1:
        va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
        out["se"] = float(math.sqrt(va + vb))
        out["sig2se"] = bool(abs(out["delta"]) > 2 * out["se"])
        if out["se"] > 0:
            out["p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        else:
            out["p"] = 0.0 if out["delta"] != 0 else 1.0
    return out


def paired(a: dict[int, float], b: dict[int, float]) -> dict[str, Any]:
    """delta and SE of the per-seed differences over the seeds both configurations have."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    out: dict[str, Any] = {"seeds": seeds, "delta": None, "se": None, "sig2se": None}
    if len(d):
        out["delta"] = float(d.mean())
    if len(d) > 1:
        out["se"] = float(d.std(ddof=1) / math.sqrt(len(d)))
        out["sig2se"] = bool(abs(out["delta"]) > 2 * out["se"]) if out["se"] > 0 else bool(out["delta"] != 0)
    return out


def holm(pvalues: list[float | None], alpha: float = ALPHA) -> list[bool | None]:
    """Holm-Bonferroni step-down: which hypotheses are rejected (None where p is missing)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    rejected: list[bool | None] = [None] * len(pvalues)
    m, stop = len(order), False
    for rank, i in enumerate(order):
        if not stop and pvalues[i] <= alpha / (m - rank):
            rejected[i] = True
        else:
            stop = True
            rejected[i] = False
    return rejected


def _bootstrap(contrasts: dict | None, a: str, b: str, signal: str) -> bool | None:
    """excludes_zero of the black-box ensemble bootstrap for A - B on the AUC (None if unavailable)."""
    sig = _SIG.get(signal)
    if contrasts is None or sig is None or sig.source is None:
        return None
    try:
        boot = contrasts[f"{a} - {b}"][f"{sig.key}/{sig.primary}"]["auc"]["bootstrap"]
        return None if boot is None else bool(boot["excludes_zero"])
    except (KeyError, TypeError):
        return None


def _and(x: bool | None, boot: bool | None, needs_boot: bool) -> bool | None:
    if not needs_boot:
        return x
    if x is False or boot is False:
        return False
    if x is True and boot is True:
        return True
    return None


# ------------------------------------------------------------------------------------------- grid
def contrast_grid(
    configs: list[Config], contrasts: dict | None, low_fit: float = LOW_FIT
) -> list[dict[str, Any]]:
    """One row per hypothesis contrast x signal with the delta and the significance under every variant."""
    by = {c.label: c for c in configs}
    rows: list[dict[str, Any]] = []
    for hyp, role, a, b, metric, signals, better in HYPOTHESES:
        if a not in by or b not in by:
            continue
        ca, cb = by[a], by[b]
        lf = set(low_fit_runs(ca, low_fit)) | set(low_fit_runs(cb, low_fit))
        for sig in signals:
            s = "rd3" if sig == "rd_bkg" else sig  # the JSD column ignores the signal
            va, vb = seed_values(ca, s, metric), seed_values(cb, s, metric)
            if not va or not vb:
                continue
            w = welch(list(va.values()), list(vb.values()))
            p = paired(va, vb)
            keep = lambda r, lf=lf: r.name not in lf  # noqa: E731
            fa, fb = seed_values(ca, s, metric, keep), seed_values(cb, s, metric, keep)
            f = welch(list(fa.values()), list(fb.values()))
            needs_boot = metric == "auc" and _SIG[s].source is not None
            boot = _bootstrap(contrasts, a, b, s) if needs_boot else None
            rows.append(
                {
                    "hypothesis": hyp,
                    "role": role,
                    "a": a,
                    "b": b,
                    "metric": metric,
                    "signal": "R&D background" if sig == "rd_bkg" else _SIG[s].label,
                    "signal_key": sig,
                    "better": better,
                    "delta": w["delta"],
                    "favour": None if w["delta"] is None else int(np.sign(w["delta"] * better)),
                    "se": w["se"],
                    "p": w["p"],
                    "paired_delta": p["delta"],
                    "paired_se": p["se"],
                    "fit_delta": f["delta"],
                    "fit_se": f["se"],
                    "low_fit_removed": sorted(lf),
                    "bootstrap_excludes_zero": boot,
                    "needs_bootstrap": needs_boot,
                    "sig": {
                        "protocol": _and(w["sig2se"], boot, needs_boot),
                        "paired": _and(p["sig2se"], boot, needs_boot),
                        "fit": _and(f["sig2se"], boot, needs_boot),
                    },
                }
            )
    rejected = holm([r["p"] for r in rows if r["role"] == "primary"])
    it = iter(rejected)
    for r in rows:
        if r["role"] == "primary":
            r["sig"]["holm"] = _and(next(it), r["bootstrap_excludes_zero"], r["needs_bootstrap"])
        else:
            r["sig"]["holm"] = None
    return rows


def _signed(r: dict[str, Any], variant: str) -> int:
    """+1: significantly favours A in the hypothesis' direction, -1: significantly the other way, 0: neither."""
    s = r["sig"].get(variant)
    return 0 if not s or r["favour"] is None else r["favour"]


def verdicts(rows: list[dict[str, Any]], variant: str) -> dict[str, str]:
    """The section-2 rules; 'contradicted' is their mirror image (the significant results go the other way)."""
    out: dict[str, str] = {}

    def grid(hyp: str) -> dict[tuple[str, str], list[int]]:
        g: dict[tuple[str, str], list[int]] = {}
        for r in rows:
            if r["hypothesis"] == hyp and r["role"] == "primary":
                g.setdefault((r["a"], r["b"]), []).append(_signed(r, variant))
        return g

    g = grid("H1")
    if g:
        pos = [sum(x > 0 for x in v) for v in g.values()]
        neg = [sum(x < 0 for x in v) for v in g.values()]
        if all(p >= 2 for p in pos) and not any(neg):
            out["H1"] = "supported"
        elif all(n >= 2 for n in neg) and not any(pos):
            out["H1"] = "contradicted"
        else:
            out["H1"] = "inconclusive"
    for hyp in ("H2a", "H2b"):
        g = grid(hyp)
        if not g:
            continue
        need = 2 if hyp == "H2a" else 1  # H2b has one quantity (JSD) per contrast
        good = sum(sum(x > 0 for x in v) >= need for v in g.values())
        bad = sum(sum(x < 0 for x in v) >= need for v in g.values())
        any_good, any_bad = (
            any(x > 0 for v in g.values() for x in v),
            any(x < 0 for v in g.values() for x in v),
        )
        if good >= 2 and not any_bad:
            out[hyp] = "supported"
        elif bad >= 2 and not any_good:
            out[hyp] = "contradicted"
        else:
            out[hyp] = "inconclusive"
    g = grid("H3")
    if g:
        v = next(iter(g.values()))
        out["H3"] = "contradicted" if any(x > 0 for x in v) else "supported"
        # H3 says gcn_complete is NOT better than deepsets: a significant positive delta contradicts it
    return out


# ------------------------------------------------------------------------------------------- H4
def h4(rows: list[dict[str, Any]], curves: list[Config]) -> list[dict[str, Any]]:
    """Sign at 1, 3 and 5 epochs (3-seed means) of every contrast that is significant under the protocol rule."""
    by = {c.label: c for c in curves}  # labels "<enc>_e<E>"
    out = []
    for r in rows:
        if not r["sig"]["protocol"]:
            continue
        s = "rd3" if r["signal_key"] == "rd_bkg" else r["signal_key"]
        signs, missing = {}, []
        for e in EPOCHS:
            ca, cb = by.get(f"{r['a']}_e{e}"), by.get(f"{r['b']}_e{e}")
            va = list(seed_values(ca, s, r["metric"]).values()) if ca else []
            vb = list(seed_values(cb, s, r["metric"]).values()) if cb else []
            if not va or not vb:
                missing.append(e)
                continue
            signs[e] = int(np.sign(np.mean(va) - np.mean(vb)))
        ref = int(np.sign(r["delta"]))
        status = "pending" if missing else ("robust" if all(v == ref for v in signs.values()) else "flips")
        out.append(
            {
                "hypothesis": r["hypothesis"],
                "contrast": f"{r['a']} - {r['b']}",
                "signal": r["signal"],
                "metric": r["metric"],
                "sign_at_2": ref,
                "signs": signs,
                "missing_epochs": missing,
                "status": status,
            }
        )
    return out


# ------------------------------------------------------------------------------------------- fit quality
def fit_quality(configs: list[Config], signals: tuple[str, ...] = ("rd3", "bb1")) -> dict[str, Any]:
    """Per encoder: low-fit runs and the Spearman correlation of AUC with the validation loss over its seeds."""
    from scipy import stats

    out: dict[str, Any] = {}
    for c in configs:
        loss = {_seed_of(r): r.best_val_loss for r in c.runs if r.best_val_loss is not None}
        if len(loss) < 3:
            continue
        e: dict[str, Any] = {
            "low_fit": low_fit_runs(c),
            "val_loss": {str(k): v for k, v in sorted(loss.items())},
        }
        for s in signals:
            v = seed_values(c, s, "auc")
            seeds = sorted(set(v) & set(loss))
            if len(seeds) >= 3:
                rho = stats.spearmanr([loss[k] for k in seeds], [v[k] for k in seeds]).statistic
                e[f"spearman_auc_{s}"] = None if not np.isfinite(rho) else float(rho)
        out[c.label] = e
    return out


def fit_adjusted(
    configs: list[Config], extra: list[Config] | None = None, signals: tuple[str, ...] = UNTUNED
) -> dict[str, Any]:
    """Encoder contrasts at equal background fit: a quadratic in the validation loss is fitted to the AUC of all
    autoencoder runs (E1, plus the training-length runs when given), and the hypothesis contrasts are recomputed
    on the residuals (unpaired 2-SE rule on the E1 seeds).  A contrast that vanishes here is explained by how well
    the encoders fit the background, not by the encoder at a given fit."""
    pool = [c for c in configs + (extra or []) if c.kind not in ("", "hl_density")]
    out: dict[str, Any] = {}
    for s in signals:
        xs, ys = [], []
        for c in pool:
            v = seed_values(c, s, "auc")
            for r in c.runs:
                if _seed_of(r) in v and r.best_val_loss is not None:
                    xs.append(r.best_val_loss)
                    ys.append(v[_seed_of(r)])
        if len(xs) < 6:
            continue
        coef = np.polyfit(xs, ys, 2)
        resid_sd = float(np.std(np.asarray(ys) - np.polyval(coef, xs), ddof=3))
        by = {c.label: c for c in configs}
        rows = []
        for hyp, role, a, b, metric, sigs, _ in HYPOTHESES:
            if metric != "auc" or s not in sigs or a not in by or b not in by:
                continue

            def res(c: Config, s: str = s, coef: np.ndarray = coef) -> list[float]:
                v = seed_values(c, s, "auc")
                return [
                    v[_seed_of(r)] - float(np.polyval(coef, r.best_val_loss))
                    for r in c.runs
                    if _seed_of(r) in v and r.best_val_loss is not None
                ]

            w = welch(res(by[a]), res(by[b]))
            rows.append(
                {
                    "hypothesis": hyp,
                    "role": role,
                    "a": a,
                    "b": b,
                    "delta": w["delta"],
                    "se": w["se"],
                    "sig2se": w["sig2se"],
                }
            )
        out[s] = {
            "coef": [float(x) for x in coef],
            "n_runs": len(xs),
            "resid_sd": resid_sd,
            "contrasts": rows,
        }
    return out


def plot_auc_vs_loss(
    groups: dict[str, list[Config]], path: Path, signals: tuple[str, ...] = ("rd3", "bb1")
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(signals), figsize=(5.2 * len(signals), 4.2), squeeze=False)
    groups = {g: [c for c in cs if c.kind not in ("", "hl_density")] for g, cs in groups.items()}  # AE losses
    encs = sorted({c.label.split("_e")[0] for cs in groups.values() for c in cs if c.runs})
    colours = {e: f"C{i}" for i, e in enumerate(encs)}
    markers = {"study": "o", "curves": "^"}
    for ax, s in zip(axes[0], signals, strict=True):
        for gname, cs in groups.items():
            for c in cs:
                enc = c.label.split("_e")[0] if gname == "curves" else c.label
                if enc not in colours:
                    continue
                v = seed_values(c, s, "auc")
                xs = [r.best_val_loss for r in c.runs if _seed_of(r) in v and r.best_val_loss is not None]
                ys = [v[_seed_of(r)] for r in c.runs if _seed_of(r) in v and r.best_val_loss is not None]
                ax.scatter(xs, ys, s=18, color=colours[enc], marker=markers.get(gname, "o"), alpha=0.8)
        ax.set_xlabel("background validation loss")
        ax.set_ylabel(f"AUC, {_SIG[s].label} ({_SIG[s].primary})")
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=colours[e], marker="o", ls="", label=e) for e in encs]
    names = {"study": "2 epochs (E1)", "curves": "1, 3, 5 epochs (E2)"}
    handles += [
        plt.Line2D([], [], color="k", marker=m, ls="", label=names[g])
        for g, m in markers.items()
        if groups.get(g)
    ]
    axes[0][-1].legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- report
def _fmt(x: float | None, f: str = "{:+.4f}") -> str:
    return "–" if x is None else f.format(x)


def _mark(s: bool | None) -> str:
    return {True: "yes", False: "no", None: "–"}[s]


def render(
    prefix: str,
    rows: list[dict],
    verd: dict[str, dict[str, str]],
    h4rows: list[dict],
    fq: dict,
    fadj: dict | None = None,
) -> str:
    lines = [
        f"# Post-hoc robustness of the hypothesis verdicts (`{prefix}`)",
        "",
        "Generated by `gaehep posthoc` (src/gaehep/eval/posthoc.py). The pre-registered verdicts are the "
        "`protocol` column; the other columns are analyses added after E1 (protocol, Deviations D2) and do not "
        "replace them.",
        "",
        "## Verdicts",
        "",
        "| hypothesis | " + " | ".join(VARIANTS) + " |",
        "|---|" + "---|" * len(VARIANTS),
    ]
    for hyp in ("H1", "H2a", "H2b", "H3"):
        if any(hyp in v for v in verd.values()):
            lines.append(f"| {hyp} | " + " | ".join(verd[v].get(hyp, "–") for v in VARIANTS) + " |")
    lines += [
        "",
        "## Contrasts",
        "",
        "Δ = A − B over seeds (AUC: higher is better; JSD: lower is better). Significant: protocol = 2 SE unpaired "
        "(+ ensemble bootstrap on BB1/BB3); paired = 2 SE of per-seed differences; holm = Welch p, Holm over the "
        f"primary tests, α = {ALPHA}; fit = protocol rule without runs above {LOW_FIT} × the configuration's "
        "median validation loss.",
        "",
        "| hyp | contrast | signal | metric | Δ | SE | paired Δ ± SE | Δ without low-fit runs | protocol | paired "
        "| holm | fit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tag = "" if r["role"] == "primary" else " (secondary)"
        lines.append(
            f"| {r['hypothesis']}{tag} | {r['a']} − {r['b']} | {r['signal']} | {r['metric']} | {_fmt(r['delta'])} "
            f"| {_fmt(r['se'], '{:.4f}')} | {_fmt(r['paired_delta'])} ± {_fmt(r['paired_se'], '{:.4f}')} "
            f"| {_fmt(r['fit_delta'])} | " + " | ".join(_mark(r["sig"].get(v)) for v in VARIANTS) + " |"
        )
    removed = sorted({n for r in rows for n in r["low_fit_removed"]})
    lines += ["", f"Low-fit runs removed in `fit`: {', '.join(removed) or 'none'}.", ""]
    lines += [
        "## Fit quality per encoder",
        "",
        "| encoder | low-fit runs | ρ(AUC 3-prong, loss) | ρ(AUC BB1, loss) |",
        "|---|---|---|---|",
    ]
    for enc, e in fq.items():
        lines.append(
            f"| {enc} | {', '.join(e['low_fit']) or '–'} | {_fmt(e.get('spearman_auc_rd3'), '{:+.2f}')} "
            f"| {_fmt(e.get('spearman_auc_bb1'), '{:+.2f}')} |"
        )
    if fadj:
        lines += [
            "",
            "## Contrasts at equal background fit",
            "",
            "AUC residuals from a quadratic in the background validation loss fitted to all autoencoder runs; "
            "Δ = mean residual of A − B (E1 seeds), significant if |Δ| > 2 SE.",
            "",
            "| signal | runs in the fit | residual sd | contrast | Δ at equal fit | SE | significant |",
            "|---|---|---|---|---|---|---|",
        ]
        for s, e in fadj.items():
            for c in e["contrasts"]:
                tag = "" if c["role"] == "primary" else " (secondary)"
                lines.append(
                    f"| {_SIG[s].label} | {e['n_runs']} | {e['resid_sd']:.4f} | {c['hypothesis']}{tag}: {c['a']} − "
                    f"{c['b']} | {_fmt(c['delta'])} | {_fmt(c['se'], '{:.4f}')} | {_mark(c['sig2se'])} |"
                )
    lines += ["", "## H4: sign at 1, 3 and 5 epochs", ""]
    if not h4rows:
        lines.append(
            "No contrast is significant under the protocol rule, or no training-length runs were found."
        )
    else:
        lines += [
            "| hyp | contrast | signal | sign at 2 | 1 | 3 | 5 | status |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for h in h4rows:
            sg = [str(h["signs"].get(e, "–")) for e in EPOCHS]
            lines.append(
                f"| {h['hypothesis']} | {h['contrast']} | {h['signal']} | {h['sign_at_2']} | "
                + " | ".join(sg)
                + f" | {h['status']} |"
            )
    return "\n".join(lines) + "\n"


def posthoc(
    prefix: str = "study", curves: str = "stct", root: Path | None = None, log=print
) -> dict[str, Path]:
    root = Path(root) if root else results_dir()
    configs = [c for c in collect(prefix, root) if c.runs and len(c.runs) > 1]
    if not configs:
        raise ValueError(f"no seeded runs named {prefix}* under {root / 'runs'}")
    curve_cfgs = [c for c in collect(curves, root) if c.runs] if curves else []
    cpath = root / "study" / f"{prefix}_contrasts.json"
    contrasts = None
    if cpath.exists():
        contrasts = load_json(cpath).get("contrasts")
    else:
        log(f"[posthoc] {cpath} missing: the black-box bootstrap requirement is undetermined")
    rows = contrast_grid(configs, contrasts)
    verd = {v: verdicts(rows, v) for v in VARIANTS}
    h4rows = h4(rows, curve_cfgs)
    fq = fit_quality(configs)
    fadj = fit_adjusted(configs, curve_cfgs)
    out_dir = root / "study"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"md": out_dir / f"{prefix}_posthoc.md", "json": out_dir / f"{prefix}_posthoc.json"}
    paths["md"].write_text(render(prefix, rows, verd, h4rows, fq, fadj))
    save_json(
        {
            "prefix": prefix,
            "curves": curves,
            "low_fit_factor": LOW_FIT,
            "alpha": ALPHA,
            "verdicts": verd,
            "contrasts": rows,
            "h4": h4rows,
            "fit_quality": fq,
            "fit_adjusted": fadj,
        },
        paths["json"],
    )
    groups = {"study": configs}
    if curve_cfgs:
        groups["curves"] = curve_cfgs
    paths["png"] = out_dir / "figures" / f"{prefix}_auc_vs_valloss.png"
    plot_auc_vs_loss(groups, paths["png"])
    for p in paths.values():
        log(f"[posthoc] wrote {p}")
    return paths
