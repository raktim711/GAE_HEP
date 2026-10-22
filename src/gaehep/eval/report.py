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


def markdown_table(rows: list[tuple[str, dict]]) -> str:
    head = "| run | " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    lines = [head, sep]
    for name, h in rows:
        lines.append(f"| {name} | " + " | ".join(_fmt(h.get(k), f) for k, _, f in COLUMNS) + " |")
    return "\n".join(lines)


def main() -> str:
    md = markdown_table(collect())
    out = results_dir() / "summary.md"
    out.write_text(
        "# Results summary (event score = max over the two leading jets)\n\n"
        "2p = 2-prong W'->XY(qq) signal, 3p = 3-prong signal (never used for training or tuning), "
        "SR = 3.3 < m_jj < 3.7 TeV. maxSIC requires >= 50 background test events after the cut.\n\n"
        + md
        + "\n"
    )
    return md
