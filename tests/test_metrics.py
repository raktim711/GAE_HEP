import math

import numpy as np

from gaehep.eval.metrics import at_signal_eff, auc, bootstrap, max_sic, summarize
from gaehep.eval.sculpting import distance_correlation, jsd, sculpting_jsd


def test_auc_matches_gaussian_formula():
    rng = np.random.default_rng(0)
    d = 1.0
    s, b = rng.normal(d, 1, 200_000), rng.normal(0, 1, 200_000)
    expected = 0.5 * (1 + math.erf(d / 2))  # Phi(d / sqrt 2)
    assert abs(auc(s, b) - expected) < 3e-3


def test_sic_and_rejection_on_separable_toy():
    s = np.r_[np.full(900, 10.0), np.full(100, -10.0)]  # 90% of signal perfectly separated
    b = np.r_[np.full(50, 10.0), np.linspace(-5, 5, 99_950)]  # 50 background events tie with signal
    sic, eff = max_sic(s, b, min_bkg=50)
    assert abs(eff - 0.9) < 1e-9
    assert abs(sic - 0.9 / math.sqrt(50 / 100_000)) < 1e-6
    rej, _ = at_signal_eff(s, b, 0.5)
    assert abs(rej - 100_000 / 50) < 1e-6


def test_max_sic_respects_min_background():
    rng = np.random.default_rng(1)
    s = np.r_[np.full(10, 5.0), rng.normal(0, 1, 990)]
    b = np.r_[np.full(3, 5.0), rng.normal(-1, 1, 10_000)]
    for min_bkg in (1, 50, 500):
        got, _ = max_sic(s, b, min_bkg=min_bkg)
        # brute force over all thresholds
        best = 0.0
        for t in np.unique(np.r_[s, b]):
            nb = np.sum(b >= t)
            if nb >= min_bkg:
                best = max(best, np.mean(s >= t) / math.sqrt(nb / len(b)))
        assert abs(got - best) < 1e-9


def test_summarize_and_bootstrap_keys():
    rng = np.random.default_rng(0)
    s, b = rng.normal(1, 1, 2000), rng.normal(0, 1, 5000)
    out = summarize(s, b)
    for k in ("auc", "max_sic", "rej@0.3", "sic@0.5", "n_sig", "n_bkg"):
        assert k in out
    bs = bootstrap(s, b, n_boot=5)
    assert 0.6 < bs["auc"]["mean"] < 0.85 and bs["auc"]["std"] > 0


def test_jsd_properties():
    p = np.array([1, 2, 3, 4.0])
    assert abs(jsd(p, p)) < 1e-12
    assert abs(jsd([1, 0], [0, 1]) - 1.0) < 1e-12
    assert abs(jsd(p, p[::-1]) - jsd(p[::-1], p)) < 1e-12


def test_sculpting_zero_for_independent_score():
    rng = np.random.default_rng(0)
    x = rng.exponential(1000, 200_000)
    s = rng.normal(size=200_000)
    out = sculpting_jsd(s, x, effs=(0.1,), bins=30)
    assert out["jsd@0.1"] < 5e-3
    s_corr = x + rng.normal(0, 10, len(x))
    assert sculpting_jsd(s_corr, x, effs=(0.1,), bins=30)["jsd@0.1"] > 0.3


def test_distance_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    assert distance_correlation(x, rng.normal(size=3000)) < 0.08
    assert distance_correlation(x, x**2) > 0.3  # nonlinear dependence detected
    assert abs(distance_correlation(x, 3 * x + 1) - 1) < 1e-9
