"""Bias-audit follow-up (src/gaehep/eval/followup.py, docs/STUDY_PROTOCOL.md Deviations D9): the statistics
helpers, the fast dijet fit used by the toys, and the calibration of the toy scan on a background-only sample."""

from __future__ import annotations

import numpy as np

from gaehep import cli
from gaehep.config import results_dir
from gaehep.eval import followup as FU
from gaehep.eval.bumphunt import DEFAULT_EDGES, _bin_expectation, fit_sidebands, poisson_z, search


def test_r2_and_group_helpers():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert FU.r2(y, y) == 1.0
    assert abs(FU.r2(y, np.full(4, y.mean()))) < 1e-12
    assert FU.r2(y, y[::-1]) < 0  # worse than predicting the mean
    g = np.array(["a", "a", "b", "b"])
    np.testing.assert_allclose(FU.group_centred(y, g), [-0.5, 0.5, -0.5, 0.5])


def test_within_group_test_ignores_between_group_trends():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(10), 5)
    # between groups: y rises with x; within groups: y falls with x
    xg, x_in = g.astype(float), rng.normal(size=50)
    x = xg + 0.1 * x_in
    y = 3.0 * xg - 0.1 * x_in + 0.001 * rng.normal(size=50)
    w = FU.within_group_test(x, y, g)
    assert w["spearman"] < -0.9 and w["slope"] < 0 and w["n"] == 50 and w["groups"] == 10
    single = FU.within_group_test(np.r_[x, 99.0], np.r_[y, 99.0], np.r_[g, 99])  # a one-run group is dropped
    assert single["n"] == 50


def _spectrum(rng, n_events: int = 400_000) -> np.ndarray:
    """m_jj draws from a falling dijet-like spectrum over the fit range (GeV)."""
    u = rng.random(n_events)
    lo, hi = DEFAULT_EDGES[0], DEFAULT_EDGES[-1]
    k = 8.0  # ~ m^-k
    return (lo ** (1 - k) + u * (hi ** (1 - k) - lo ** (1 - k))) ** (1 / (1 - k))


def test_fast_fit_converges_at_least_as_well_as_the_analysis_fit():
    """The toys' fit (analytic gradient, L-BFGS-B then BFGS) reaches a likelihood at least as good as the analysis
    fit (Nelder-Mead then Powell), from the full-range fit as the start; the local Z agree to a few tenths (the
    analysis fit can stop short of the minimum)."""
    rng = np.random.default_rng(2)
    counts, _ = np.histogram(_spectrum(rng), bins=DEFAULT_EDGES)
    ff = FU._FastFit()
    full = fit_sidebands(counts, DEFAULT_EDGES, np.ones(len(counts), bool)).params

    def nll(p, mask):
        mu = _bin_expectation(DEFAULT_EDGES, p)[mask]
        return float(np.sum(mu - counts[mask] * np.log(mu)))

    for w in ((3000.0, 3400.0), (3600.0, 4000.0), (4300.0, 4700.0)):
        sr = FU._sr_bins(w)
        ref = fit_sidebands(counts, DEFAULT_EDGES, ~sr).params
        p = ff.fit(counts, ~sr, full)
        assert nll(p, ~sr) <= nll(ref, ~sr) + 1e-6
        n = float(counts[sr].sum())
        z_ref = poisson_z(n, float(_bin_expectation(DEFAULT_EDGES, ref)[sr].sum()))
        z_fast = poisson_z(n, float(ff.expectation(p)[sr].sum()))
        assert abs(z_fast - z_ref) < 0.5, (w, z_fast, z_ref)
        np.testing.assert_allclose(ff.expectation(p), _bin_expectation(DEFAULT_EDGES, p), rtol=1e-10)


def test_null_toys_are_calibrated_and_the_scan_max_dominates():
    rng = np.random.default_rng(3)
    mjj = _spectrum(rng)
    score = rng.random(len(mjj))  # an m_jj-independent score: the cuts do not sculpt
    toys = FU.null_toys(mjj, score, n_toys=150, rng=np.random.default_rng(4))
    z = np.asarray(toys["z_bb1_window_2pct"])
    # the local Z of bumphunt.search ignores the uncertainty of the background extrapolated from the sideband
    # fit, so without signal it is centred on 0 but wider than N(0, 1); the toys include that uncertainty
    assert abs(z.mean()) < 0.35 and 1.0 < z.std(ddof=1) < 2.5
    zmax = np.asarray(toys["zmax_scan_cuts"])
    assert np.all(zmax >= z - 1e-9) and np.all(np.asarray(toys["zmax_scan_all"]) >= zmax - 1e-9)
    assert np.median(zmax) > 1.0  # 14 windows x 2 cuts: the largest local Z of a scan is well above 0
    # the data scan of this sample uses the analysis fit and agrees with the toys' fixed window in scale
    r = search(mjj, score, 0.02, sr=(3600.0, 4000.0))
    assert abs(r["z"]) < 4


def test_global_significance_counts_toys():
    zmax = np.arange(1000) / 250.0  # 0 ... 4
    g = FU._global(zmax, 3.0)
    assert abs(g["p_global"] - 251 / 1001) < 1e-12 and 0.6 < g["z_global"] < 0.7


# ------------------------------------------------------------------ black-box commands (session F2), synthetic data
from tests.test_blackbox import env, runs  # noqa: E402, F401  (synthetic black boxes, two trained runs)


def test_black_box_followup_commands(env, runs):  # noqa: F811
    """prepare-run -> score-sample (all events) -> bb-scan, and bb1-mass-check, on the synthetic black boxes."""
    rdir = results_dir() / "runs" / "rnd"
    # a fresh session has only the committed files: weights under results/models, no model.pt or scores.npz
    (results_dir() / "models" / "rnd").mkdir(parents=True, exist_ok=True)
    (rdir / "model.pt").replace(results_dir() / "models" / "rnd" / "model.pt")
    ref = np.load(rdir / "scores.npz")["bkg_val"]
    (rdir / "scores.npz").unlink()
    FU.prepare_run("rnd", log=lambda *a: None)
    assert (rdir / "model.pt").exists()
    np.testing.assert_allclose(np.load(rdir / "scores.npz")["bkg_val"], ref, rtol=1e-6)  # the same F_B scores
    assert cli.main(["score-sample", "rnd", "--sample", "bb1", "--n-boot", "0"]) == 0
    out = FU.bb_scan("rnd", "bb1", n_toys=2, log=lambda *a: None)
    assert out["true_window"] == [3600.0, 4000.0] and len(out["rows"]) == len(FU.SCAN_WINDOWS) * len(
        FU.SCAN_EFFS
    )
    assert out["toys"]["n_toys"] == 2 and (rdir / "bb1_scan.json").exists()
    m = FU.bb1_mass_check("rnd", n_ref_events=50, log=lambda *a: None)
    assert m["n_sig_window"] + m["n_bkg_window"] > 0 and "m_J sum" in m["ref_auc_window"]
    assert (rdir / "bb1_mass.json").exists()
