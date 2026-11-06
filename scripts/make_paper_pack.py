"""One Markdown file with every text source for writing the paper, for models that cannot open archives.

    python scripts/make_paper_pack.py OUT.md

PAPER_CONTEXT.md comes first; every other file follows under a banner with its repository path.  The figures are
not included: upload paper/figures/*.png (or the PDFs) next to the pack.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = [
    "results/PAPER_CONTEXT.md",
    "paper/figures/CAPTIONS.md",
    "paper/results_draft.md",
    "results/study/audit_followup.md",
    "docs/STUDY_PROTOCOL.md",
    "results/study/study_table.md",
    "results/study/fix_table.md",
    "results/study/study_posthoc.md",
    "results/study/stct_table.md",
    "results/study/ablations.md",
    "results/study/e3_summary.md",
    "results/study/study_n100_table.md",
    "results/study/study_bb1data_table.md",
    "results/study/bb1_diagnostics.md",
    "results/study/bb3_diagnostics.md",
    "docs/F2_SESSION.md",
    "RESULTS.md",
    "results/summary.md",
]
HEADER = """# GAE_HEP paper pack

This one file merges every text source for writing the GAE_HEP paper. Read the first part, PAPER_CONTEXT.md, first:
it explains the study, lists the claims the data support (section 7) and says where each number comes from. The other
files follow in full, each under a banner line with its repository path.

The paper figures (fig1_auc_by_encoder … fig6_bb3_modes) are uploaded separately. They are in journal style at final
print size; use the PDFs in the manuscript. Their draft captions and the colour/marker encoding are in the
paper/figures/CAPTIONS.md part below.
"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper_pack.md"
    parts = [HEADER, "Contents: " + ", ".join(f for f in FILES if (REPO / f).exists())]
    for f in FILES:
        p = REPO / f
        if p.exists():
            parts += ["", "", "=" * 64, f"FILE: {f}", "=" * 64, "", p.read_text()]
    out.write_text("\n".join(parts))
    print(f"{out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
