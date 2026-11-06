"""Publication style for the paper figures.

The figures are made at their final printed size, so that no scaling happens in the journal's production:
- single column 3.4 in (8.6 cm); double column 7.0 in (17.8 cm). These are the widths of PRD/revtex, JHEP,
  EPJC and SciPost.
- Text: Computer Modern (matplotlib's own cmr10 and cm mathtext, so no LaTeX installation is needed), 8 pt for
  labels and 7 pt for ticks and legends. This matches 10 pt body text at print size.
- Lines at least 0.6 pt. HEP tick convention: inward ticks on all four sides, with minor ticks.
- White background. Vector PDF with embedded TrueType fonts (``pdf.fonttype`` 42), plus a 300 dpi PNG for
  previews.

Encoder encoding, shared by every figure, so that it reads in colour, for colour-vision deficiencies and in
grey-scale:
- colour and marker shape = the interaction step. Colours are Okabe–Ito: none = blue circle, local graph =
  green triangle, attention = vermillion square, complete-graph GCN = black diamond. The palette is validated
  for CVD separation and contrast on white.
- fill = IRC safety: filled = IRC-safe, open = IRC-unsafe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE = 3.4  # in
DOUBLE = 7.0  # in

BLUE, GREEN, VERMILLION, BLACK, GREY = "#0072B2", "#009E73", "#D55E00", "#000000", "#7f7f7f"

ENCODERS = ["deepsets", "efn", "edgeconv", "ewmp", "transformer", "ew_attention", "gcn_complete"]
LABEL = {
    "deepsets": "Deep Sets",
    "efn": "EFN",
    "edgeconv": "EdgeConv",
    "ewmp": "EWMP",
    "transformer": "Transformer",
    "ew_attention": "EW attention",
    "gcn_complete": "GCN (complete graph)",
}
IRC_SAFE = {"efn", "ewmp", "ew_attention"}
_FAMILY = {
    "deepsets": "none",
    "efn": "none",
    "edgeconv": "graph",
    "ewmp": "graph",
    "transformer": "attention",
    "ew_attention": "attention",
    "gcn_complete": "gcn",
}
_COLOUR = {"none": BLUE, "graph": GREEN, "attention": VERMILLION, "gcn": BLACK}
_MARKER = {"none": "o", "graph": "^", "attention": "s", "gcn": "D"}

SIGNAL = {"rd3": "R&D 3-prong", "bb1": "Black box 1", "bb3": "Black box 3"}

RC = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.0,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.8,
    "errorbar.capsize": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply() -> None:
    plt.rcParams.update(RC)


def colour(enc: str) -> str:
    return _COLOUR[_FAMILY[enc]]


def marker(enc: str) -> str:
    return _MARKER[_FAMILY[enc]]


def mstyle(enc: str, **kw) -> dict:
    """Marker keywords for an encoder: its colour and shape, filled if IRC-safe and open otherwise."""
    c = colour(enc)
    out = {
        "marker": marker(enc),
        "color": c,
        "markeredgecolor": c,
        "markerfacecolor": c if enc in IRC_SAFE else "white",
        "linestyle": "",
    }
    out.update(kw)
    return out


def sstyle(enc: str, **kw) -> dict:
    """The same encoding as keywords for ``Axes.scatter``."""
    c = colour(enc)
    out = {
        "marker": marker(enc),
        "edgecolors": c,
        "facecolors": c if enc in IRC_SAFE else "white",
        "linewidths": 0.7,
    }
    out.update(kw)
    return out


def encoder_handles(encoders: list[str] | None = None) -> list:
    return [plt.Line2D([], [], label=LABEL[e], **mstyle(e)) for e in (encoders or ENCODERS)]


def panel_label(ax, text: str, loc: str = "upper left") -> None:
    """A panel tag such as ``(a) R&D 3-prong`` inside the axes."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.96, "top") if "upper" in loc else (0.04, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8)


def save(fig, out: Path, name: str) -> list[Path]:
    """Write ``name``.pdf (vector, for the journal) and ``name``.png (300 dpi preview)."""
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths
