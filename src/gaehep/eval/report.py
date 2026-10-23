"""Aggregate ``results/runs/*/metrics.json`` into a markdown summary table."""

from __future__ import annotations

from pathlib import Path

from gaehep.config import load_json, results_dir

COLUMNS = [
    ("auc_2p_SR", "AUC 2p SR", "{:.3f}"),
    ("maxsic_2p_SR", "maxSIC 2p SR", "{:.2f}"),
    ("rej30_2p_SR", "1/εB@εS=0.3 2p SR", "{:.1f}"),
    ("auc_3p_SR", "AUC 3p SR", "{:.3f}"),
    ("maxsic_3p_SR", "maxSIC 3p SR", "{:.2f}"),
    ("auc_2p_full", "AUC 2p full", "{:.3f}"),
    ("jsd_mjj@0.01", "JSD mjj @εB=1%", "{:.3f}"),
    ("dcor_mj_jet", "dCor(score, m_j)", "{:.2f}"),
]


def _fmt(v, f: str) -> str:
    if v is None:
        return "–"
    try:
        return f.format(v)
    except (TypeError, ValueError):
        return str(v)


def collect(root: Path | None = None) -> list[tuple[str, dict]]:
    root = Path(root) if root else results_dir() / "runs"
    rows = []
    for p in sorted(root.glob("*/metrics.json")):
        d = load_json(p)
        rows.append((d.get("name", p.parent.name), d["headline"]))
    return rows


def collect_full(root: Path | None = None) -> list[tuple[str, dict]]:
    root = Path(root) if root else results_dir() / "runs"
    return [
        (load_json(p).get("name", p.parent.name), load_json(p)["metrics"])
        for p in sorted(root.glob("*/metrics.json"))
    ]


def markdown_table(rows: list[tuple[str, dict]]) -> str:
    head = "| run | " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    lines = [head, sep]
    for name, h in rows:
        lines.append(f"| {name} | " + " | ".join(_fmt(h.get(k), f) for k, _, f in COLUMNS) + " |")
    return "\n".join(lines)


AGGS = ("max", "sum", "min", "prod")


def aggregation_table(rows: list[tuple[str, dict]], key: str, fmt: str, region: str = "SR") -> str:
    """One row per run: ``key`` (auc / max_sic / rej@0.3) for 2-prong and 3-prong, every aggregation."""
    head = (
        "| run | " + " | ".join(f"2p {a}" for a in AGGS) + " | " + " | ".join(f"3p {a}" for a in AGGS) + " |"
    )
    sep = "|---|" + "|".join("---:" for _ in range(2 * len(AGGS))) + "|"
    lines = [head, sep]
    for name, m in rows:
        cells = []
        for sk in ("sig_test", "sig3_test"):
            for a in AGGS:
                e = m["event"].get(a, {}).get(f"{sk}/{region}")
                cells.append(_fmt(None if e is None else e[key], fmt))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sculpting_table(rows: list[tuple[str, dict]]) -> str:
    lines = [
        "| run | JSD(m_jj) @εB=10% max | @1% max | @10% prod | @1% prod | dCor(jet score, m_j) | dCor(jet score, n) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in rows:
        sc = m["sculpting"]

        def get(a, e, sc=sc):
            return sc.get(a, {}).get("mjj", {}).get(f"jsd@{e}")

        lines.append(
            f"| {name} | {_fmt(get('max', 0.1), '{:.3f}')} | {_fmt(get('max', 0.01), '{:.3f}')} | "
            f"{_fmt(get('prod', 0.1), '{:.3f}')} | {_fmt(get('prod', 0.01), '{:.3f}')} | "
            f"{_fmt(m['correlation']['jet'].get('dcor_mj'), '{:.2f}')} | "
            f"{_fmt(m['correlation']['jet'].get('dcor_nconst'), '{:.2f}')} |"
        )
    return "\n".join(lines)


def main() -> str:
    md = markdown_table(collect())
    full = collect_full()
    out = results_dir() / "summary.md"
    out.write_text(
        "# Results summary (test splits)\n\n"
        "2p = 2-prong W'->XY(qq) signal, 3p = 3-prong signal (never used for training or tuning), "
        "SR = 3.3 < m_jj < 3.7 TeV. Event score from the two jets' scores: max / sum / min, or prod = "
        "log F_B(s_1) + log F_B(s_2) with F_B the background (validation) CDF of jet scores. "
        "maxSIC requires >= 50 background test events after the cut.\n\n"
        "## AUC in the SR\n\n" + aggregation_table(full, "auc", "{:.3f}") + "\n\n"
        "## max SIC in the SR\n\n" + aggregation_table(full, "max_sic", "{:.2f}") + "\n\n"
        "## Background rejection 1/εB at εS = 0.3 in the SR\n\n"
        + aggregation_table(full, "rej@0.3", "{:.1f}")
        + "\n\n## Sculpting and correlations (background test events)\n\n"
        + sculpting_table(full)
        + "\n\n## Headline (max aggregation)\n\n"
        + md
        + "\n"
    )
    return md


def compare(
    runs: list[str],
    out_dir: Path,
    agg: str = "max",
    region: str = "SR",
    labels: dict[str, str] | None = None,
) -> list[Path]:
    """Overlay rejection and SIC curves of several runs (from their saved ``scores.npz``).

    A run can be given as ``name:agg`` to use its own event aggregation instead of ``agg``.
    """
    import numpy as np

    from gaehep.eval import plots
    from gaehep.eval.evaluate import aggregate, jet_reference, region_mask
    from gaehep.train.run import Data

    data = Data(n_const=50)
    tabs = {k: data.table(k) for k in ("bkg_test", "sig_test", "sig3_test")}
    paths = []
    for sk, tag in (("sig_test", "2prong"), ("sig3_test", "3prong")):
        curves = {}
        for spec in runs:
            r, _, a = spec.partition(":")
            a = a or agg
            with np.load(results_dir() / "runs" / r / "scores.npz") as f:
                ref = jet_reference({k: f[k] for k in f.files})
                b = aggregate(f["bkg_test"], a, ref)[region_mask(tabs["bkg_test"]["mjj"], region)]
                s = aggregate(f[sk], a, ref)[region_mask(tabs[sk]["mjj"], region)]
            curves[(labels or {}).get(spec, f"{r} [{a}]")] = (s, b)
        out_dir = Path(out_dir)
        paths.append(
            plots.plot_rejection(
                curves, out_dir / f"rejection_{tag}_{region}_{agg}.png", title=f"{tag}, {region}, {agg}"
            )
        )
        paths.append(
            plots.plot_sic(curves, out_dir / f"sic_{tag}_{region}_{agg}.png", title=f"{tag}, {region}, {agg}")
        )
    return paths
