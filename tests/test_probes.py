"""``gaehep probes``: R-squared of ridge / MLP regressions of jet observables on a run's latents."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from gaehep import cli
from gaehep.config import load_json
from gaehep.eval import efp
from gaehep.eval.probes import N_FOLDS, TARGETS, cv_r2, jet_targets, probe
from gaehep.train.run import SPLIT_SAMPLE, Data
from tests.conftest import synthetic_pool
from tests.study_env import fake_env

N_EV = 150  # events whose latents are probed


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    with fake_env(tmp_path_factory.mktemp("probes"), n_bkg=2000, n_sig=100) as root:
        yield root


@pytest.fixture(scope="module")
def tables(env):
    data = Data(n_const=50)
    events = data.events("bkg_val")[:N_EV]
    store = data.store(SPLIT_SAMPLE["bkg_val"])
    tab = {k: store[k][events] for k in ("m", "pt", "n_const", "tau1", "tau2", "tau3")}
    return jet_targets(tab, store.pool(events, 50).const)


@pytest.fixture(scope="module")
def probe_run(env, tables):
    """A run directory whose first latent dimension is the jet mass and whose others are noise."""
    d = env / "results" / "runs" / "latent_run"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    m = tables["m_j"]
    z = np.concatenate(
        [
            ((m - m.mean()) / m.std())[:, None] + 0.05 * rng.normal(size=(len(m), 1)),
            rng.normal(size=(len(m), 3)),
        ],
        axis=1,
    )
    np.savez(d / "latents.npz", bkg_val=z.reshape(N_EV, 2, 4).astype(np.float32))
    (d / "config.yaml").write_text("model: set_ae\ndata:\n  n_const: 50\n")
    return d


def test_targets_cover_the_protocol_and_undefined_taus_are_nan():
    pool = synthetic_pool(6, 12, seed=1)
    tab = {
        "m": np.arange(6.0).reshape(3, 2),
        "pt": np.full((3, 2), 1500.0),
        "n_const": np.arange(6).reshape(3, 2),
        "tau1": np.array([[0.4, 0.0], [0.5, 0.5], [0.3, 0.3]]),
        "tau2": np.array([[0.2, 0.1], [0.25, 0.25], [0.15, 0.0]]),
        "tau3": np.full((3, 2), 0.1),
    }
    t = jet_targets(tab, pool.const)
    assert set(t) == set(TARGETS)
    assert all(v.shape == (6,) for v in t.values())
    assert t["m_j"].tolist() == [0, 1, 2, 3, 4, 5]
    assert (
        np.isnan(t["tau21"][1]) and np.isnan(t["tau32"][5]) and np.isfinite(t["tau21"][[0, 2, 3, 4, 5]]).all()
    )
    assert np.allclose(t["efp_wedge"], efp.probe_efps(pool.const)["efp_wedge"])


def test_cv_r2_recovers_a_linear_target_and_ignores_scales():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(400, 3))
    y = 2 * z[:, 0] - z[:, 1] + 0.1 * rng.normal(size=400)
    folds = cv_r2(z, y, "ridge")
    assert len(folds) == N_FOLDS and min(folds) > 0.95
    # inputs and targets are standardised inside each fold: affine changes leave R^2 untouched
    assert np.allclose(cv_r2(1000 * z + 5, 100 * y - 7, "ridge"), folds, atol=1e-9)
    assert np.mean(cv_r2(z[:, 2:], y, "ridge")) < 0.05  # unrelated input
    nonlinear = np.abs(z[:, 0]) + 0.05 * rng.normal(size=400)
    assert np.mean(cv_r2(z, nonlinear, "ridge")) < 0.1 < 0.5 < np.mean(cv_r2(z, nonlinear, "mlp"))


def test_probe_writes_probes_json(env, probe_run, tables):
    out = probe("latent_run", n_events=N_EV, log=lambda *a: None)
    saved = load_json(probe_run / "probes.json")
    assert saved["name"] == "latent_run" and saved["n_events"] == N_EV and saved["n_jets"] == 2 * N_EV
    assert saved["latent_dim"] == 4 and saved["n_folds"] == 5 and set(saved["targets"]) == set(TARGETS)
    for t in TARGETS:
        n_ok = int(np.isfinite(tables[t]).sum())
        assert saved["targets"][t]["n_jets"] == n_ok
        for model in ("ridge", "mlp"):
            r = saved["targets"][t][model]
            assert len(r["r2_folds"]) == 5
            assert r["r2_mean"] == pytest.approx(np.mean(r["r2_folds"]))
            assert r["r2_std"] == pytest.approx(np.std(r["r2_folds"], ddof=1))
    assert out["targets"]["m_j"]["ridge"]["r2_mean"] > 0.95  # the latent holds the mass ...
    assert saved["targets"]["m_j"]["mlp"]["r2_mean"] > 0.9
    assert saved["targets"]["pt_j"]["ridge"]["r2_mean"] < 0.3  # ... and the noise dimensions hold nothing


def test_probe_cli_uses_the_first_10000_events_at_most(env, probe_run):
    (probe_run / "probes.json").unlink()
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(["probes", "latent_run"]) == 0
    assert load_json(probe_run / "probes.json")["n_events"] == N_EV  # only that many latents exist
