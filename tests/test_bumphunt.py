import numpy as np

from gaehep.eval.bumphunt import DEFAULT_EDGES, SQRT_S, dijet_fn, poisson_z, search


def _toy(n_bkg=300_000, n_sig=0, seed=0, sig_score_shift=0.0):
    """Falling dijet-like spectrum (sampled from the fit function) + optional Gaussian bump at 3.5 TeV."""
    rng = np.random.default_rng(seed)
    x = np.linspace(DEFAULT_EDGES[0], DEFAULT_EDGES[-1], 20_000)
    pdf = dijet_fn(x / SQRT_S, np.array([0.0, 10.0, 5.0, 0.0]))
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    mjj_b = np.interp(rng.random(n_bkg), cdf, x)
    mjj_s = rng.normal(3500, 100, n_sig)
    score_b = rng.normal(0, 1, n_bkg)  # independent of mjj -> no sculpting
    score_s = rng.normal(sig_score_shift, 1, n_sig)
    return np.r_[mjj_b, mjj_s], np.r_[score_b, score_s]


def test_poisson_z():
    assert poisson_z(100, 100) == 0
    assert abs(poisson_z(130, 100) - 2.8) < 0.15
    assert poisson_z(80, 100) < 0


def test_no_fake_bump_without_signal():
    mjj, score = _toy(seed=1)
    for eff in (1.0, 0.1):
        r = search(mjj, score, eff)
        assert abs(r["z"]) < 3.0, r


def test_injected_bump_found_and_enhanced_by_cut():
    mjj, score = _toy(n_sig=1500, seed=2, sig_score_shift=3.0)
    z_all = search(mjj, score, 1.0)["z"]
    z_cut = search(mjj, score, 0.05)["z"]
    assert z_all > 2.0
    assert z_cut > z_all
