"""evaluate_score_arrays (metrics of arbitrary per-event scores) and the ``gaehep eval-scores`` command."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gaehep.cli import main
from gaehep.config import load_json
from gaehep.eval.metrics import bootstrap, summarize
from gaehep.eval.score_file import evaluate_score_arrays
from gaehep.eval.sculpting import sculpting_jsd


@pytest.fixture
def fast_png():
    """Low-resolution, untrimmed PNGs (speed only)."""
    import matplotlib.pyplot as plt

    with plt.rc_context({"savefig.dpi": 30, "savefig.bbox": "standard"}):
        yield


def _toy(n_sig: int = 500, n_bkg: int = 3000, d: float = 1.0, seed: int = 0):
    """Unit Gaussians at separation d; m_jj uniform in [2.5, 4.5] TeV (20% of events in the default SR)."""
    rng = np.random.default_rng(seed)
    score = np.r_[rng.normal(d, 1, n_sig), rng.normal(0, 1, n_bkg)]
    label = np.r_[np.ones(n_sig, int), np.zeros(n_bkg, int)]
    mjj = rng.uniform(2500, 4500, n_sig + n_bkg)
    return score, label, mjj


def test_gaussian_auc_inside_bootstrap_interval():
    """Unit Gaussians at separation d: AUC = Phi(d / sqrt 2) lies inside the 16-84% bootstrap interval."""
    d = 1.0
    expected = 0.5 * (1 + math.erf(d / 2))  # Phi(d / sqrt 2) = 0.760
    score, label, _ = _toy(n_sig=2000, n_bkg=2000, d=d, seed=6)
    res = evaluate_score_arrays(score, label, n_boot=20)
    bs = res["full"]["bootstrap"]["auc"]
    assert abs(res["full"]["auc"] - expected) < 0.01
    assert bs["q16"] + 0.002 < expected < bs["q84"] - 0.002  # with these seeds 0.756 < 0.760 < 0.769
    assert evaluate_score_arrays(score, label, n_boot=20) == res  # bootstrap seeded: deterministic


def test_perfect_and_random_scores():
    """A perfectly separating score gives AUC 1 (infinite rejection), a label-independent one AUC ~ 0.5."""
    label = np.r_[np.ones(500, int), np.zeros(1500, int)]
    perfect = evaluate_score_arrays(label + np.linspace(0, 0.5, len(label)), label, n_boot=0)["full"]
    assert perfect["auc"] == 1.0 and perfect["rej@0.3"] == math.inf
    rnd = evaluate_score_arrays(np.random.default_rng(1).normal(size=len(label)), label, n_boot=0)["full"]
    assert abs(rnd["auc"] - 0.5) < 0.05
    assert "bootstrap" not in rnd


def test_lower_is_signal_flips_score():
    """higher_is_signal=False evaluates -score: same metrics as the negated score, AUC -> 1 - AUC."""
    score, label, mjj = _toy()
    ref = evaluate_score_arrays(score, label, mjj, n_boot=10)
    flipped = evaluate_score_arrays(-score, label, mjj, n_boot=10, higher_is_signal=False)
    assert flipped == {**ref, "higher_is_signal": False}
    wrong_way = evaluate_score_arrays(score, label, n_boot=0, higher_is_signal=False)["full"]["auc"]
    assert wrong_way == pytest.approx(1 - ref["full"]["auc"], abs=1e-12)


def test_event_order_irrelevant():
    """Shuffling the events changes nothing, bootstrap and SR included."""
    score, label, mjj = _toy()
    perm = np.random.default_rng(3).permutation(len(score))
    ref = evaluate_score_arrays(score, label, mjj, n_boot=10)
    assert evaluate_score_arrays(score[perm], label[perm], mjj[perm], n_boot=10) == ref


def test_max_sic_respects_min_bkg():
    """max SIC only uses cuts keeping >= min_bkg background events (NaN if none does)."""
    rng = np.random.default_rng(1)
    s = np.r_[np.full(30, 5.0), rng.normal(0, 1, 270)]
    b = np.r_[np.full(3, 5.0), rng.normal(-1, 1, 1000)]  # 3 background events tie with the signal tail
    score, label = np.r_[s, b], np.r_[np.ones(len(s), int), np.zeros(len(b), int)]
    effs = {}
    for min_bkg in (1, 50):
        got = evaluate_score_arrays(score, label, n_boot=0, min_bkg=min_bkg)["full"]
        best, eff = 0.0, None
        for t in np.unique(score):  # brute force over all thresholds
            nb = np.sum(b >= t)
            if nb >= min_bkg and np.mean(s >= t) / math.sqrt(nb / len(b)) > best:
                best, eff = np.mean(s >= t) / math.sqrt(nb / len(b)), np.mean(s >= t)
        assert got["max_sic"] == pytest.approx(best, rel=1e-12)
        assert got["eff_s_at_max_sic"] == pytest.approx(eff, rel=1e-12)
        effs[min_bkg] = eff
    assert effs[1] < 0.11 < 0.4 < effs[50]  # min_bkg=1: the tail cut keeping 3 background events wins
    assert math.isnan(evaluate_score_arrays(score, label, n_boot=0, min_bkg=2000)["full"]["max_sic"])


def test_agrees_with_metrics_and_sculpting():
    """Full-range and SR entries equal metrics.summarize / bootstrap, sculpting equals sculpting_jsd."""
    score, label, mjj = _toy()
    res = evaluate_score_arrays(score, label, mjj, n_boot=10, sr=(3300.0, 3700.0))
    sig, bkg = label == 1, label == 0
    sr = (mjj >= 3300) & (mjj <= 3700)
    for region, s, b in (("full", score[sig], score[bkg]), ("SR", score[sig & sr], score[bkg & sr])):
        entry = dict(res[region])
        assert entry.pop("bootstrap") == bootstrap(np.sort(s), np.sort(b), n_boot=10)
        assert entry == pytest.approx(summarize(s, b), rel=1e-12)
    assert res["SR"]["n_sig"] == np.sum(sig & sr) and res["sr"] == [3300.0, 3700.0]
    assert res["sculpting"]["mjj"] == sculpting_jsd(score[bkg], mjj[bkg])


def test_degenerate_inputs_raise():
    """No signal / no background (overall or in the SR), bad labels or shapes, NaN scores: ValueError."""
    score, label, mjj = _toy(n_sig=50, n_bkg=200)
    with pytest.raises(ValueError, match="full range: 0 signal"):
        evaluate_score_arrays(score[label == 0], label[label == 0])
    with pytest.raises(ValueError, match=r"SR m_jj in \[3300, 3700\] GeV: 0 signal and 0 background"):
        evaluate_score_arrays(score, label, mjj / 1000)  # m_jj in TeV
    with pytest.raises(ValueError, match="labels"):
        evaluate_score_arrays(score, label + 1)
    with pytest.raises(ValueError, match="equal length"):
        evaluate_score_arrays(score, label[:-1])
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_score_arrays(np.r_[score[:-1], np.nan], label)


def test_cli_npz_round_trip(tmp_path, capsys, fast_png):
    """eval-scores on an .npz without m_jj writes metrics.json (= evaluate_score_arrays) and three PNGs."""
    score, label, _ = _toy(n_sig=300, n_bkg=1000)
    np.savez(tmp_path / "toy.npz", score=score.astype(np.float32), label=label.astype(np.int8))
    out = tmp_path / "out"
    assert main(["eval-scores", str(tmp_path / "toy.npz"), "--out", str(out), "--n-boot", "5"]) == 0
    assert "no 'mjj' column" in capsys.readouterr().out
    metrics = load_json(out / "metrics.json")
    assert metrics == evaluate_score_arrays(score.astype(np.float32), label, n_boot=5, name="toy")
    assert "SR" not in metrics and metrics["name"] == "toy"
    assert sorted(p.name for p in out.iterdir()) == [
        "metrics.json",
        "rejection.png",
        "score_hist.png",
        "sic.png",
    ]
    assert all((out / f).stat().st_size > 0 for f in ("rejection.png", "sic.png", "score_hist.png"))


def test_cli_csv_round_trip_with_mjj(tmp_path, fast_png):
    """eval-scores on a .csv with custom column names, m_jj, --sr and --name; bad input exits cleanly."""
    score, label, mjj = _toy(n_sig=300, n_bkg=1500)
    csv = tmp_path / "toy.csv"
    np.savetxt(csv, np.c_[label, score, mjj], delimiter=",", header="y, s, m", comments="", fmt="%.17g")
    out = tmp_path / "out"
    argv = ["eval-scores", str(csv), "--out", str(out), "--score-key", "s", "--label-key", "y"]
    assert main([*argv, "--mjj-key", "m", "--sr", "3000,4000", "--name", "my model", "--n-boot", "5"]) == 0
    metrics = load_json(out / "metrics.json")
    ref = evaluate_score_arrays(score, label, mjj, sr=(3000, 4000), n_boot=5, name="my model")
    assert metrics == ref and metrics["sr"] == [3000.0, 4000.0] and "jsd@0.1" in metrics["sculpting"]["mjj"]
    assert all((out / f).stat().st_size > 0 for f in ("rejection.png", "sic.png", "score_hist.png"))
    with pytest.raises(SystemExit, match="no column"):
        main(["eval-scores", str(csv), "--out", str(out)])  # default keys score / label are absent
    with pytest.raises(SystemExit, match="--sr"):
        main([*argv, "--sr", "3700"])
