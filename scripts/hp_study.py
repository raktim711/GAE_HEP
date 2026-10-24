"""Tables and figures of the CPU hyper-parameter study (dev protocol) from ``results/dev``.

    python scripts/hp_study.py            # -> results/dev/hp_study.md, results/figures/hp_*.png

The study manifest below lists which dev runs belong to which variant (several runs = several seeds).
Round 1 varies one setting of the previous final model (``configs/ef_vae.yaml``: latent-4 VAE, β = 10⁻³,
kernels 0.02-0.4, 5 epochs × 120k jets); round 2 varies one setting of the new base ("e2": the same model
trained for 2 epochs); round 3 gives the round-2 gains more seeds.  Every number is a 2-prong *validation*
number (SR, prod aggregation of the reconstruction score unless stated otherwise).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEV = ROOT / "results" / "dev"
FIG = ROOT / "results" / "figures"

BASE = ["ef_ae_d11_l4_vae", "ef_ae_d11_l4_vae_s2", "ef_ae_d11_l4_vae_s3"]
E2 = ["hp_ep2", "hp_ep2_s2", "hp_ep2_s3", "hp_ep2_s4", "hp_ep2_s5"]

ROUND1 = {  # one change to the previous final model
    "β = 10⁻⁴": ["hp_beta1e-4"],
    "β = 3·10⁻³": ["hp_beta3e-3"],
    "β = 10⁻²": ["hp_beta1e-2", "hp_beta1e-2_s2", "hp_beta1e-2_s3"],
    "β = 3·10⁻²": ["hp_beta3e-2"],
    "β = 0.1": ["hp_beta1e-1"],
    "latent 3": ["hp_lat3"],
    "latent 6": ["hp_lat6"],
    "encoder width 32": ["hp_w32"],
    "encoder width 64": ["hp_w64"],
    "encoder depth 2": ["hp_d2"],
    "encoder depth 4": ["hp_d4"],
    "kernels 0.01-0.3": ["hp_sig_a"],
    "kernels 0.01-0.05": ["hp_sig_d"],
    "kernels 0.01-0.1": ["hp_sig_f"],
    "kernels 0.02-0.05": ["hp_sig_e"],
    "kernels 0.02-0.1": ["hp_sig_b", "fx_b3_sB_e5_s2", "fx_b3_sB_e5_s3"],
    "kernels 0.02-0.1, latent 6": ["cd_l6"],
    "kernels 0.02-0.1, latent 8": ["cd_l8", "cd_l8_s2", "cd_l8_s3"],
    "2 epochs": E2,
    "3 epochs": ["hp_ep3", "hp_ep3_s2", "hp_ep3_s3"],
    "10 epochs": ["hp_ep10"],
}
PAIRS = {  # combinations of the three round-1 gains
    "β = 10⁻² + kernels 0.02-0.1": ["fx_b2_sB_e5"],
    "β = 10⁻² + 2 epochs": ["fx_b2_sD_e2"],
    "kernels 0.02-0.1 + 2 epochs": ["fx_b3_sB_e2"],
    "all three": ["fx_b2_sB_e2"],
}
ROUND2 = {  # one change to e2
    "1 epoch": ["e2_ep1"],
    "2 × 60k jets": ["e2_j60k"],
    "2 × 240k jets": ["e2_j240k"],
    "learning rate 3·10⁻⁴": ["e2_lr3e-4", "e2_lr3e-4_s2", "e2_lr3e-4_s3"],
    "learning rate 3·10⁻³": ["e2_lr3e-3"],
    "batch 128": ["e2_bs128"],
    "batch 512": ["e2_bs512"],
    "latent 3": ["e2_l3"],
    "latent 6": ["e2_l6"],
    "β = 10⁻⁴": ["e2_b1e-4"],
    "plain AE (no KL)": ["e2_ae"],
    "encoder width 32": ["e2_w32"],
    "encoder width 96": ["e2_w96"],
    "encoder depth 2": ["e2_d2"],
    "encoder depth 4": ["e2_d4", "e2_d4_s2", "e2_d4_s3"],
    "depth 4 + learning rate 3·10⁻⁴": ["e2_d4lr", "e2_d4lr_s2", "e2_d4lr_s3"],
    "k = 6 neighbours": ["e2_k6"],
    "k = 8 neighbours": ["e2_k8", "e2_k8_s2", "e2_k8_s3", "e2_k8_s4", "e2_k8_s5"],
    "k = 8 + learning rate 3·10⁻⁴": ["e2_k8lr"],
    "k = 16": ["e2_k16"],
    "k = 24": ["e2_k24"],
    "max aggregation": ["e2_aggrmax"],
    "static k-NN graph": ["e2_static"],
    "16 decoder points": ["e2_nout16"],
    "64 decoder points": ["e2_nout64"],
    "decoder 128 × 2": ["e2_dec128"],
    "decoder 512 × 2": ["e2_dec512"],
    "kernels 0.02-0.05": ["e2_sig_e"],
    "kernels 0.05-0.8": ["e2_sig_w"],
    "30 constituents": ["e2_n30"],
    "100 constituents": ["e2_n100"],
}
BUDGET = {  # base model, training budget (epochs x jets per epoch)
    "1 × 120k": ["e2_ep1"],
    "2 × 60k": ["e2_j60k"],
    "2 × 120k": E2,
    "3 × 120k": ["hp_ep3", "hp_ep3_s2", "hp_ep3_s3"],
    "2 × 240k": ["e2_j240k"],
    "5 × 120k": BASE,
    "10 × 120k": ["hp_ep10"],
}


def metric(run: str, key: str = "SR/prod", q: str = "auc") -> float | None:
    p = DEV / run / "dev_metrics.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["metrics"][key][q])


def val_loss(run: str) -> float | None:
    p = DEV / run / "history.json"
    return float(json.loads(p.read_text())["best_val_loss"]) if p.exists() else None


def stats(runs: list[str], key: str = "SR/prod", q: str = "auc", suffix: str = "") -> tuple[list[float], str]:
    vals = [v for r in runs if (v := metric(r + suffix, key, q)) is not None]
    if not vals:
        return vals, "–"
    if len(vals) == 1:
        return vals, f"{vals[0]:.3f}" if q == "auc" else f"{vals[0]:.2f}"
    fmt = ".3f" if q == "auc" else ".2f"
    return vals, f"{np.mean(vals):{fmt}} ± {np.std(vals, ddof=1):{fmt}} ({len(vals)})"


def table(title: str, groups: dict[str, list[str]], ref: list[str]) -> list[str]:
    ref_auc = np.mean(stats(ref)[0])
    out = [
        f"### {title}",
        "",
        "| variant | dev AUC (seeds) | Δ vs reference | max SIC | ρ(score, m_J) |",
        "|---|---|---:|---:|---:|",
    ]
    for lab, runs in groups.items():
        vals, txt = stats(runs)
        if not vals:
            continue
        _, sic = stats(runs, q="max_sic")
        rho = np.mean([v for r in runs if (v := _rho(r)) is not None])
        out.append(f"| {lab} | {txt} | {np.mean(vals) - ref_auc:+.3f} | {sic} | {rho:+.2f} |")
    return [*out, ""]


def _rho(run: str) -> float | None:
    p = DEV / run / "dev_metrics.json"
    return float(json.loads(p.read_text())["metrics"]["bkg_spearman"]["m"]) if p.exists() else None


def effects_figure(path: Path) -> None:
    from gaehep.eval import style

    style.apply()
    import matplotlib.pyplot as plt

    panels = [
        ("round 1: one change to the previous model", ROUND1 | PAIRS, BASE),
        ("round 2: one change to e2", ROUND2, E2),
    ]
    heights = [sum(1 for r in g.values() if stats(r)[0]) for _, g, _ in panels]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.3 * max(heights) + 1.8))
    for ax, (title, groups, ref) in zip(axes, panels, strict=True):
        ref_vals = stats(ref)[0]
        mu, sd = np.mean(ref_vals), np.std(ref_vals, ddof=1)
        rows = [(lab, stats(runs)[0]) for lab, runs in groups.items() if stats(runs)[0]]
        rows.sort(key=lambda r: np.mean(r[1]))
        ax.axvspan(-2 * sd, 2 * sd, color="0.88", zorder=0, label="reference ± 2σ (seeds)")
        ax.axvline(0, color="0.5", lw=1)
        for i, (_, vals) in enumerate(rows):
            d = np.asarray(vals) - mu
            ax.scatter(d, np.full(len(d), i), s=18, color="C0", alpha=0.6, zorder=3)
            ax.scatter(
                [d.mean()],
                [i],
                s=46,
                marker="D" if len(d) > 1 else "o",
                color="C3" if len(d) > 1 else "C0",
                zorder=4,
            )
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10)
        ax.tick_params(axis="y", which="minor", left=False, right=False)
        ax.set_xlabel("Δ dev AUC (SR, prod) vs reference")
        ax.set_title(title, fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def budget_figure(path: Path) -> None:
    from gaehep.eval import style

    style.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, (lab, runs) in enumerate(BUDGET.items()):
        pts = [(val_loss(r), metric(r)) for r in runs if metric(r) is not None]
        if not pts:
            continue
        x, y = np.array(pts).T
        ax.scatter(x, y, color=f"C{i}", s=24, alpha=0.7)
        ax.errorbar(
            [x.mean()],
            [y.mean()],
            yerr=[y.std()] if len(y) > 1 else None,
            fmt="D",
            color=f"C{i}",
            ms=7,
            capsize=3,
            label=f"{lab} jets",
        )
    ax.set_xlabel("background validation loss (best epoch)")
    ax.set_ylabel("dev AUC (SR, prod)")
    ax.set_title("training budget of the latent-4 VAE", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def seed_ensemble_table() -> list[str]:
    """Rank ensembles over seeds (built with gaehep dev-ensemble as seeds<n>_<tag>[+ens|+kl])."""
    rows = [
        ("previous model (5 epochs)", "seeds3_base"),
        ("kernels 0.02-0.1", "seeds3_sB"),
        ("β = 10⁻²", "seeds3_b1e-2"),
        ("3 epochs", "seeds3_e3"),
        ("2 epochs (e2), 3 seeds", "seeds3_e2"),
        ("2 epochs (e2), 5 seeds", "seeds5_e2"),
        ("2 epochs, k = 8 (final), 5 seeds", "seeds5_e2k8"),
        ("latent 8, 5 epochs", "seeds3_d6"),
    ]
    out = [
        "### Seed ensembles and score kinds",
        "",
        "Rank ensembles over seeds of the reconstruction score (rec), of the KL score, and of the per-seed rank "
        "ensemble of the two (rec+KL); SR AUC / max SIC (prod) and ρ(score, m_J).",
        "",
        "| configuration | rec | KL | rec+KL |",
        "|---|---|---|---|",
    ]
    for lab, tag in rows:
        cells = []
        for suf in ("", "+kl", "+ens"):
            a, s, r = metric(tag + suf), metric(tag + suf, q="max_sic"), _rho(tag + suf)
            cells.append("–" if a is None else f"{a:.3f} / {s:.2f} (ρ {r:+.2f})")
        out.append(f"| {lab} | {' | '.join(cells)} |")
    return [*out, ""]


def main() -> None:
    lines = [
        "# Hyper-parameter study (CPU, dev protocol)",
        "",
        "2-prong validation split, SR, prod aggregation of the reconstruction score; dev AUC as mean ± sample std over "
        "seeds (number of seeds in brackets), single seeds otherwise.  Generated by `scripts/hp_study.py`.",
        "",
        f"Reference of round 1 (previous final model, {len(BASE)} seeds): {stats(BASE)[1]}; "
        f"reference of round 2 (e2 = 2 epochs, {len(E2)} seeds): {stats(E2)[1]}.",
        "",
        *table("Round 1: one change to the previous final model", ROUND1, BASE),
        *table("Pairs of the round-1 gains", PAIRS, BASE),
        *table("Training budget (latent-4 VAE)", BUDGET, BASE),
        *table("Round 2: one change to e2", ROUND2, E2),
        *seed_ensemble_table(),
    ]
    (DEV / "hp_study.md").write_text("\n".join(lines) + "\n")
    effects_figure(FIG / "hp_effects.png")
    budget_figure(FIG / "hp_budget.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
