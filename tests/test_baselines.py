"""Non-graph baselines: single observables, the jet-observable flow and the EFP k-NN score; the EFPs they use;
the ``SCORERS`` registry.  A small synthetic dataset (module-scoped), no LHCO files."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import torch
import yaml

from gaehep import baselines, cli
from gaehep.baselines import (
    OBSERVABLES,
    SCORERS,
    draw_reference_jets,
    efp_features,
    hl_features,
    hl_nll,
    knn_mean_distance,
    run_efp_knn,
    run_hl_density,
    standardise_reference,
)
from gaehep.config import load_json
from gaehep.eval import efp
from gaehep.eval.irc_probe import add_soft, split_top
from gaehep.train.run import SCORE_SPLITS, Data
from tests.conftest import permute_nodes, synthetic_pool
from tests.study_env import fake_env

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    with fake_env(tmp_path_factory.mktemp("baselines")) as root:
        yield root


@pytest.fixture(scope="module")
def runs(env):
    return env / "results" / "runs"


@pytest.fixture(scope="module")
def data(env) -> Data:
    return Data(n_const=50)


def load_npz(p):
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def cli_quiet(argv: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return cli.main(argv)


# ------------------------------------------------------------------------------------------ registry
def test_scorers_registry_keys():
    assert set(SCORERS) == {"observable", "hl_density", "efp_knn"}
    assert all(callable(f) for f in SCORERS.values())


# ----------------------------------------------------------------------------------- obs-baselines
def test_obs_baselines_run_directories(env, runs, data):
    assert cli_quiet(["obs-baselines"]) == 0
    assert set(OBSERVABLES) == {"study_obs_mj", "study_obs_nconst", "study_obs_tau21", "study_obs_tau32"}
    for name, obs in OBSERVABLES.items():
        d = runs / name
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        assert cfg["model"] == "observable" and cfg["observable"] == obs
        sc = load_npz(d / "scores.npz")
        assert set(sc) == set(SCORE_SPLITS)
        for split, s in sc.items():
            assert s.shape == (len(data.events(split)), 2) and s.dtype == np.float32 and np.isfinite(s).all()
        m = load_json(d / "metrics.json")
        assert m["name"] == name and 0 <= m["metrics"]["event"]["prod"]["sig3_test/SR"]["auc"] <= 1
        # the registered scorer reproduces the stored scores and returns float64 (n_events, 2)
        s = SCORERS["observable"](d, data, "bkg_test")
        assert s.dtype == np.float64 and s.shape == sc["bkg_test"].shape
        assert np.array_equal(s.astype(np.float32), sc["bkg_test"])
    mj = load_npz(runs / "study_obs_mj" / "scores.npz")["sig_test"]
    assert np.array_equal(mj, data.table("sig_test")["m"].astype(np.float32))
    # the heavy 2-prong synthetic jets are found by the mass
    auc = load_json(runs / "study_obs_mj" / "metrics.json")["metrics"]["event"]["prod"]["sig_test/SR"]["auc"]
    assert auc > 0.9


# --------------------------------------------------------------------------------------- hl-density
def test_hl_features_define_undefined_taus():
    tab = {
        "m": np.array([[0.2, 80.0], [50.0, 90.0]]),
        "pt": np.array([[1500.0, 1400.0], [1300.0, 0.0]]),
        "n_const": np.array([[0, 20], [30, 40]]),
        "tau1": np.array([[0.4, 0.0], [0.5, 0.5]]),
        "tau2": np.array([[0.2, 0.1], [np.nan, 0.25]]),
        "tau3": np.array([[0.1, 0.05], [0.1, 0.1]]),
    }
    f = hl_features(tab)
    assert f.shape == (2, 2, 5)
    assert f[0, 0, 0] == 0.0 and f[0, 0, 4] == 0.0  # log(max(m, 1)), log(max(n, 1))
    assert f[0, 0, 2] == pytest.approx(0.5) and f[0, 0, 3] == pytest.approx(0.5)
    assert np.isnan(f[0, 1, 2])  # tau1 = 0
    assert np.isnan(f[1, 0, 2]) and np.isnan(f[1, 0, 3])  # tau2 missing
    assert np.isnan(f[1, 1, 1])  # pT = 0
    assert np.isfinite(f[0, 0]).all() and np.isfinite(f[1, 1, [0, 2, 3, 4]]).all()


def test_hl_density_run(env, runs, data):
    assert cli_quiet(["hl-density", "--seed", "1", "--epochs", "2"]) == 0
    d = runs / "study_hldensity_s1"
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    assert (
        cfg["model"] == "hl_density"
        and cfg["seed"] == 1
        and cfg["flow"] == {"transforms": 4, "hidden": [128, 128]}
    )
    hist = load_json(d / "history.json")
    assert hist["status"] == "done" and np.isfinite(hist["best_val_loss"]) and hist["epochs_done"] <= 2
    sc = load_npz(d / "scores.npz")
    assert set(sc) == set(SCORE_SPLITS)
    assert all(s.shape == (len(data.events(k)), 2) and np.isfinite(s).all() for k, s in sc.items())
    assert load_json(d / "metrics.json")["name"] == "study_hldensity_s1"
    # the registered scorer reloads model.pt and reproduces the stored scores
    for split in ("bkg_val", "sig3_test"):
        s = SCORERS["hl_density"](d, data, split)
        assert s.dtype == np.float64 and s.shape == sc[split].shape
        assert np.array_equal(s.astype(np.float32), sc[split])


def test_hl_density_trains_on_background_only_and_resumes(env, runs, data, monkeypatch):
    d = runs / "study_hldensity_s1"
    before = (d / "model.pt").read_bytes()
    monkeypatch.setattr(
        baselines.LatentFlow, "fit", lambda *a, **k: pytest.fail("a finished run is retrained")
    )
    run_hl_density(1, data=data, epochs=2, log=lambda *a: None)
    assert (d / "model.pt").read_bytes() == before
    monkeypatch.undo()
    # the flow sees only bkg_train/bkg_val jets: the fit arguments hold no signal-split rows
    seen = {}
    real_fit = baselines.LatentFlow.fit

    def spy(self, z_train, z_val, **kw):
        seen["n"] = (len(z_train), len(z_val))
        return real_fit(self, z_train, z_val, **kw)

    monkeypatch.setattr(baselines.LatentFlow, "fit", spy)
    run_hl_density(2, data=data, epochs=1, log=lambda *a: None)
    n_train, n_val = 2 * len(data.events("bkg_train")), 2 * len(data.events("bkg_val"))
    assert seen["n"] == (n_train, n_val)  # no jet of the synthetic data has an undefined tau


def test_hl_density_seeds_and_reproducibility(env, runs, data):
    run_hl_density(1, data=data, name="hl_again", epochs=2, log=lambda *a: None)
    a, b = load_npz(runs / "study_hldensity_s1" / "scores.npz"), load_npz(runs / "hl_again" / "scores.npz")
    assert all(np.array_equal(a[k], b[k]) for k in a)
    c = load_npz(runs / "study_hldensity_s2" / "scores.npz")
    assert not np.array_equal(a["bkg_test"], c["bkg_test"])


def test_hl_density_scores_outliers_and_imputes(env, runs, data):
    flow, fill = baselines._load_flow(runs / "study_hldensity_s1" / "model.pt")
    tab = data.table("bkg_test")
    base = hl_nll(flow, fill, tab)
    wild = {**tab, "n_const": tab["n_const"] * 1000}
    assert np.median(hl_nll(flow, fill, wild)) > np.median(base) + 10
    holes = {**tab, "tau2": np.full_like(tab["tau2"], np.nan)}  # undefined tau21, tau32
    s = hl_nll(flow, fill, holes)
    assert s.shape == base.shape and np.isfinite(s).all()


def test_hl_density_scorer_uses_only_the_jet_table(env, runs, data):
    """Permutation / padding invariance by construction: no constituent is read."""

    class NoConstituents(Data):
        def pool(self, *a, **k):
            raise AssertionError("hl_density must not read constituents")

    d = NoConstituents(n_const=50)
    s = SCORERS["hl_density"](runs / "study_hldensity_s1", d, "bkg_val")
    assert np.isfinite(s).all()
    s = SCORERS["observable"](runs / "study_obs_mj", d, "bkg_val")
    assert np.isfinite(s).all()


# ------------------------------------------------------------------------------------------ the EFPs
def _delta_r(const: np.ndarray) -> np.ndarray:
    eta, phi = const[..., 1], const[..., 2]
    dphi = (phi[:, :, None] - phi[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(eta[:, :, None] - eta[:, None, :], dphi)


def test_prime_efp_set_is_the_protocols():
    graphs = efp.prime_graphs()
    assert len(graphs) == 20 and all(1 <= len(g) <= 4 for g in graphs)
    assert [(0, 1)] in graphs and [(0, 1), (0, 2)] in graphs  # edge and wedge
    assert efp.prime_efps(synthetic_pool(5).const).shape == (5, 20)


def test_efps_match_the_energyflow_package():
    """Graph list and values equal energyflow 1.4.0's (the package is not a dependency: it does not import with
    numpy 2 on Python 3.11 / 3.12, see gaehep.eval.efp)."""
    try:
        ef = pytest.importorskip("energyflow")
    except (
        Exception
    ) as err:  # a wasserstein build for numpy 1.x raises ImportError subclasses, others may differ
        pytest.skip(f"energyflow is not importable here: {err}")
    pool = synthetic_pool(60, 40, seed=11)
    const = pool.const.copy()
    const[:3, 1:] = 0.0  # single-constituent jets: every EFP is 0
    const[3, :, 1:] = const[3, 0, 1:]  # all constituents at one point: every EFP is 0
    efpset = ef.EFPSet(*efp.PRIME_SET, measure="hadr", beta=1, normed=True)
    assert [[tuple(map(int, e)) for e in g] for g in efpset.graphs() if len(g)] == efp.prime_graphs()
    ref = efpset.batch_compute(const.astype(np.float64), n_jobs=1)[:, 1:]
    assert np.allclose(efp.prime_efps(const), ref, rtol=1e-9, atol=1e-15)
    for name, graph in efp.PROBE_GRAPHS.items():
        want = ef.EFP(graph, measure="hadr", beta=1, normed=True).batch_compute(
            const.astype(np.float64), n_jobs=1
        )
        assert np.allclose(efp.probe_efps(const)[name], want, rtol=1e-9, atol=1e-15)


def test_probe_efps_match_the_definitions():
    pool = synthetic_pool(12, 20, seed=3)
    const = pool.const.astype(np.float64)
    z = const[..., 0] / const[..., 0].sum(1, keepdims=True)
    th = _delta_r(const)
    got = efp.probe_efps(pool.const)
    zz = np.sqrt(z)[:, :, None] * th * np.sqrt(z)[:, None, :]  # sqrt(z_i) theta_ij sqrt(z_j)
    wedge = np.einsum("bj,bj->b", z, np.einsum("bi,bij->bj", z, th) ** 2)
    triangle = np.einsum("bij,bjk,bki->b", zz, zz, zz)
    square = np.einsum("bij,bjk,bkl,bli->b", zz, zz, zz, zz)
    assert np.allclose(got["efp_wedge"], wedge, rtol=1e-9, atol=1e-14)
    assert np.allclose(got["efp_triangle"], triangle, rtol=1e-9, atol=1e-14)
    assert np.allclose(got["efp_square"], square, rtol=1e-9, atol=1e-14)


def test_efps_are_permutation_and_padding_invariant():
    pool = synthetic_pool(40, 30, seed=5)
    base = efp.prime_efps(pool.const)
    other = permute_nodes(pool, seed=9, pad_extra=11)
    assert other.const.shape[1] == 41
    assert np.allclose(efp.prime_efps(other.const), base, rtol=1e-9, atol=1e-14)
    pe = efp.probe_efps(pool.const)
    po = efp.probe_efps(other.const)
    assert all(np.allclose(po[k], pe[k], rtol=1e-9, atol=1e-14) for k in pe)
    shifted = pool.const.astype(np.float64)
    shifted[..., 2] += 2 * np.pi  # the azimuth enters only through DeltaR
    assert np.allclose(efp.prime_efps(shifted), base, rtol=1e-9, atol=1e-14)


def test_efps_are_irc_safe():
    pool = synthetic_pool(30, 20, seed=2)
    base = efp.prime_efps(pool.const)
    assert np.allclose(efp.prime_efps(split_top(pool, None).const), base, rtol=1e-6, atol=1e-12)
    change = {z: np.max(np.abs(efp.prime_efps(add_soft(pool, 1, z).const) - base)) for z in (1e-6, 1e-8)}
    assert 30 < change[1e-6] / change[1e-8] < 300  # a soft emission of pT fraction z changes an EFP by O(z)


def test_efp_log_floor_keeps_single_constituent_jets_finite():
    one = np.zeros((1, 5, 3), dtype=np.float32)
    one[0, 0] = (100.0, 0.1, 0.2)
    x = efp.prime_efps(one)
    assert np.all(x == 0)
    assert np.isfinite(efp.log_efps(x)).all()


# -------------------------------------------------------------------------------------------- k-NN
@pytest.mark.parametrize("chunk", [512, 7])
def test_knn_mean_distance_matches_brute_force(chunk):
    rng = np.random.default_rng(0)
    ref, q = rng.normal(size=(60, 4)), rng.normal(size=(23, 4))
    d = np.sort(np.linalg.norm(q[:, None] - ref[None], axis=-1), axis=1)[:, :5].mean(1)
    got = knn_mean_distance(q, ref, 5, chunk=chunk)
    assert got.dtype == np.float64 and np.allclose(got, d, rtol=1e-10)
    with pytest.raises(ValueError, match="neighbours"):
        knn_mean_distance(q, ref[:3], 5)


def test_standardise_reference_is_zero_mean_unit_variance():
    x = np.random.default_rng(1).normal(3.0, 2.0, size=(500, 6))
    s = standardise_reference(x)
    assert np.allclose(s["ref"].mean(0), 0, atol=1e-12) and np.allclose(s["ref"].std(0), 1)


def test_knn_scores_are_permutation_and_padding_invariant():
    pool = synthetic_pool(60, 30, seed=7)
    ref = standardise_reference(efp_features(pool.const[:40]))

    def score(const):
        return knn_mean_distance((efp_features(const) - ref["mean"]) / ref["std"], ref["ref"], 3)

    other = permute_nodes(pool, seed=4, pad_extra=9)
    assert np.allclose(score(other.const), score(pool.const), rtol=1e-8, atol=1e-10)


# ------------------------------------------------------------------------------------------ efp-knn
def test_reference_jets_are_a_seeded_bkg_train_draw(env, data):
    ref = draw_reference_jets(data, 50, 20201)
    flat = np.sort(np.random.default_rng(20201).choice(2 * len(data.events("bkg_train")), 50, replace=False))
    assert ref.shape == (50, 50, 3)
    assert np.array_equal(ref, data.pool("bkg_train").const[flat])
    assert np.array_equal(ref, draw_reference_jets(data, 50, 20201))
    assert not np.array_equal(ref, draw_reference_jets(data, 50, 1))
    with pytest.raises(ValueError, match="bkg_train has"):
        draw_reference_jets(data, 10**9, 20201)


def test_efp_knn_run(env, runs, data):
    assert cli_quiet(["efp-knn", "-k", "3", "--n-ref", "200"]) == 0
    d = runs / "study_efpknn"
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    assert cfg["model"] == "efp_knn" and cfg["k"] == 3 and cfg["n_ref"] == 200 and cfg["ref_seed"] == 20201
    assert cfg["efp"]["set"] == ["d<=4", "p==1"] and cfg["data"]["n_const"] == 50
    ref = load_npz(d / "reference.npz")
    assert ref["ref"].shape == (200, 20)
    assert np.allclose(ref["ref"].mean(0), 0, atol=1e-9)
    sc = load_npz(d / "scores.npz")
    assert set(sc) == set(SCORE_SPLITS)
    assert all(
        s.shape == (len(data.events(k)), 2) and np.isfinite(s).all() and (s >= 0).all() for k, s in sc.items()
    )
    auc3 = load_json(d / "metrics.json")["metrics"]["event"]["prod"]["sig3_test/SR"]["auc"]
    assert auc3 > 0.8  # multi-prong synthetic jets sit far from the 1-prong background
    # the registered scorer reproduces the stored scores from the cached reference ...
    s = SCORERS["efp_knn"](d, data, "sig_test")
    assert s.dtype == np.float64 and np.array_equal(s.astype(np.float32), sc["sig_test"])
    # ... and rebuilds the same reference from bkg_train when the cache is gone
    (d / "reference.npz").unlink()
    assert np.array_equal(SCORERS["efp_knn"](d, data, "sig_test").astype(np.float32), sc["sig_test"])
    assert np.array_equal(load_npz(d / "reference.npz")["ref"], ref["ref"])


def test_efp_knn_scorer_requires_matching_constituents(env, runs):
    with pytest.raises(ValueError, match="constituents"):
        SCORERS["efp_knn"](runs / "study_efpknn", Data(n_const=30), "bkg_val")


def test_efp_knn_run_is_training_free_and_deterministic(env, runs, data):
    run_efp_knn(data=data, name="knn_again", k=3, n_ref=200, log=lambda *a: None)
    a, b = load_npz(runs / "study_efpknn" / "scores.npz"), load_npz(runs / "knn_again" / "scores.npz")
    assert all(np.array_equal(a[k], b[k]) for k in a)
