"""Every figure function writes a non-empty PNG, also for degenerate input, and closes its figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gaehep.eval import plots

RNG = np.random.default_rng(0)
SIG, BKG = RNG.normal(1.5, 1, 200), RNG.normal(0, 1, 1000)
MJJ = RNG.uniform(2500, 5000, 1000)
ONE = np.array([0.7])  # a class with a single event
EQ_S, EQ_B = np.full(20, 0.5), np.full(80, 0.5)  # all scores equal


def _history(n_epochs: int, loss: float | None = None) -> list[dict]:
    return [
        {"epoch": e, "train_loss": loss or 1.0 / (e + 1), "val_loss": loss or 1.2 / (e + 1)}
        for e in range(n_epochs)
    ]


def _bumphunt_rows(effs, n_injs, seeds) -> list[dict]:
    # the keys plot_bumphunt reads from eval.bumphunt.run_bumphunt rows
    return [
        {"eff": e, "n_inj": n, "seed": s, "z": 0.01 * n * (1.5 - e) + 0.3 * s}
        for e in effs
        for n in n_injs
        for s in seeds
    ]


def _weak_res(reps, s_over_b, n_rep) -> dict:
    # {rep: {"iad": rows, "supervised": {...}}} as returned by eval.weak.run_iad
    return {
        rep: {
            "iad": [
                {"s_over_b": v, "rep": r, "max_sic": 1.0 + 10 * v + 0.1 * r + i, "auc": 0.6}
                for v in s_over_b
                for r in range(n_rep)
            ],
            "supervised": {"auc": 0.9, "max_sic": 5.0 + i},
        }
        for i, rep in enumerate(reps)
    }


CASES = {  # "+" marks degenerate input: all scores equal, a class with a single event, a single entry
    "rejection+": lambda p: plots.plot_rejection(
        {"A": (SIG, BKG), "equal": (EQ_S, EQ_B), "one sig": (ONE, BKG), "one bkg": (SIG, ONE)}, p, title="t"
    ),
    "sic+": lambda p: plots.plot_sic(
        {"A": (SIG, BKG), "equal": (EQ_S, EQ_B), "one sig": (ONE, BKG), "one bkg": (SIG, ONE)}, p, title="t"
    ),
    "score_hist+one": lambda p: plots.plot_score_hist({"background": BKG, "signal": SIG, "one": ONE}, p),
    "score_hist+equal": lambda p: plots.plot_score_hist({"background": EQ_B, "signal": EQ_S}, p, logy=False),
    "sculpting": lambda p: plots.plot_sculpting(MJJ, BKG + MJJ / 1000, p, title="t"),
    "sculpting+equal": lambda p: plots.plot_sculpting(MJJ, np.full(len(MJJ), 0.5), p),
    "sculpting+one": lambda p: plots.plot_sculpting(np.array([3500.0]), ONE, p),
    "history": lambda p: plots.plot_history(_history(5), p, title="t"),
    "history+one": lambda p: plots.plot_history(_history(1, loss=0.5), p),
    "bumphunt": lambda p: plots.plot_bumphunt(_bumphunt_rows((1.0, 0.1, 0.01), (0, 300, 1200), (0, 1)), p),
    "bumphunt+one": lambda p: plots.plot_bumphunt(_bumphunt_rows((1.0,), (0,), (0,)), p),
    "weak": lambda p: plots.plot_weak(_weak_res(("hl4", "latent"), (0.001, 0.003, 0.01), 3), p, title="t"),
    "weak+one": lambda p: plots.plot_weak(_weak_res(("hl4",), (0.002,), 1), p),
}


@pytest.fixture(autouse=True)
def _fast_png():
    """Start without open figures; low-resolution, untrimmed PNGs (speed only, ~2x faster)."""
    plt.close("all")
    with plt.rc_context({"savefig.dpi": 30, "savefig.bbox": "standard"}):
        yield


@pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore::UserWarning")
@pytest.mark.parametrize("case", list(CASES))
def test_plot_writes_png_and_closes_figure(case, tmp_path):
    """Each plot_* function writes a non-empty PNG for normal and degenerate input, leaving no open figure."""
    path = tmp_path / "sub" / f"{case}.png"
    out = CASES[case](path)
    assert out == path
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 1000
    assert plt.get_fignums() == []
