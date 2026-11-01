"""N-subjettiness from the stored constituents (data/nsubjettiness.py) and the computed-tau baselines."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from gaehep.data.nsubjettiness import N_MAX, nsubjettiness


def jets(n: int = 40, n_nodes: int = 30, prongs: int = 2, seed: int = 0, spread: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.zeros((n, n_nodes, 3))
    for i in range(n):
        k = rng.integers(prongs + 3, n_nodes + 1)
        axes = rng.normal(0, 0.3, size=(prongs, 2)) + np.array([rng.uniform(-1, 1), rng.uniform(-3, 3)])
        which = rng.integers(0, prongs, size=k)
        d = axes[which] + rng.normal(0, spread, size=(k, 2))
        c[i, :k, 0] = np.sort(rng.exponential(50.0, size=k) + 1.0)[::-1]
        c[i, :k, 1], c[i, :k, 2] = d[:, 0], np.mod(d[:, 1] + np.pi, 2 * np.pi) - np.pi
    return c


def test_tau_n_vanishes_with_n_constituents_and_is_nan_below():
    c = np.zeros((1, 10, 3))
    c[0, :2] = [[100.0, 0.1, 0.2], [50.0, -0.3, 0.4]]
    t = nsubjettiness(c)
    assert t[0, 1] == pytest.approx(0.0, abs=1e-9) and np.isnan(t[0, 2]) and t[0, 0] > 0


def test_tau1_is_the_global_minimum_of_the_weighted_distance_sum():
    c = jets(8, prongs=3, spread=0.1)
    t = nsubjettiness(c)
    for i in range(len(c)):
        m = c[i, :, 0] > 0
        pt, eta, phi = c[i, m, 0], c[i, m, 1], c[i, m, 2]

        def f(x, pt=pt, eta=eta, phi=phi):
            dphi = np.mod(phi - x[1] + np.pi, 2 * np.pi) - np.pi
            return float((pt * np.hypot(eta - x[0], dphi)).sum())

        x0 = [np.average(eta, weights=pt), phi[0]]
        best = min(
            minimize(f, x0 + d, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-9}).fun
            for d in (np.zeros(2), np.array([0.05, -0.05]))
        )
        assert t[i, 0] <= best * (1 + 1e-5)
        assert t[i, 0] == pytest.approx(best, rel=1e-4)


def test_permutation_and_padding_invariance():
    c = jets(20)
    t = nsubjettiness(c)
    rng = np.random.default_rng(1)
    perm = c[:, rng.permutation(c.shape[1])]
    padded = np.concatenate([c, np.zeros((len(c), 7, 3))], axis=1)
    np.testing.assert_allclose(nsubjettiness(perm), t, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(nsubjettiness(padded), t, rtol=1e-12, atol=1e-12)


def test_collinear_split_leaves_tau_unchanged():
    c = jets(20)
    split = np.concatenate([c, np.zeros((len(c), 1, 3))], axis=1)
    split[:, -1] = split[:, 0] * [0.5, 1.0, 1.0]
    split[:, 0, 0] *= 0.5
    np.testing.assert_allclose(nsubjettiness(split), nsubjettiness(c), rtol=1e-6)


def test_two_prong_jets_have_small_tau21():
    t2 = nsubjettiness(jets(30, prongs=2, spread=0.01))
    t1 = nsubjettiness(jets(30, prongs=1, spread=0.1))
    assert np.median(t2[:, 1] / t2[:, 0]) < 0.3 < np.median(t1[:, 1] / t1[:, 0])
    assert t2.shape[1] == N_MAX


def test_tau_choice_in_the_observable_scores():
    from gaehep.eval.evaluate import observable_scores

    tab = {
        "m": np.ones(3),
        "n_const": np.ones(3),
        "tau1": np.array([1.0, 1.0, 1.0]),
        "tau2": np.array([0.5, 0.2, 0.9]),
        "tau3": np.ones(3),
        "tau1c": np.array([1.0, 1.0, 1.0]),
        "tau2c": np.array([0.1, 0.1, 0.1]),
        "tau3c": np.ones(3),
    }
    np.testing.assert_allclose(observable_scores(tab)["-tau21"], [-0.5, -0.2, -0.9])
    np.testing.assert_allclose(observable_scores(tab, "computed")["-tau21"], [-0.1, -0.1, -0.1])


def test_computed_tau_reaches_the_tables_and_the_baselines(tmp_path):
    from gaehep.baselines import run_obs_baselines
    from gaehep.config import load_json, processed_dir
    from gaehep.data.nsubjettiness import compute_sample
    from gaehep.train.run import Data
    from tests.study_env import fake_env

    with fake_env(tmp_path, n_bkg=300, n_sig=60):
        for sample in ("rnd", "qqq"):
            if (processed_dir() / sample).exists():
                compute_sample(processed_dir() / sample, workers=1, log=lambda *a: None)
        data = Data(n_const=50)
        tab = data.table("bkg_test")
        assert {"tau1c", "tau2c", "tau3c"} <= set(tab) and np.isfinite(tab["tau1c"]).mean() > 0.9
        res = run_obs_baselines(data, n_boot=2, log=lambda *a: None, prefix="fix", tau="computed")
        assert set(res) == {"fix_obs_mj", "fix_obs_nconst", "fix_obs_tau21", "fix_obs_tau32"}
        cfg = load_json(tmp_path / "results" / "runs" / "fix_obs_tau21" / "metrics.json")
        assert cfg["config"]["tau"] == "computed"
