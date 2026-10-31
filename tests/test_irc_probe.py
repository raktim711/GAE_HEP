"""``gaehep irc-probe``: perturbations, the k-NN tie count and the run-directory output."""

from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
import pytest

from gaehep import cli
from gaehep.config import load_json
from gaehep.data.dataset import fit_scaler
from gaehep.eval import irc_probe
from gaehep.eval.irc_probe import PERTURBATIONS, add_soft, knn_tie_counts, reorder, split_top
from gaehep.train.run import Data, run
from tests.conftest import permute_nodes, synthetic_pool
from tests.study_env import fake_env

CFG = {
    "model": "ef_ae",
    "model_args": {"hidden": [8, 8], "k": 4, "latent_dim": 2, "n_out": 8, "dec_hidden": [16]},
    "data": {"n_const": 50},
    "train": {
        "epochs": 1,
        "jets_per_epoch": 200,
        "batch_size": 64,
        "val_jets": 100,
        "threads": 1,
        "device": "cpu",
    },
}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    with fake_env(tmp_path_factory.mktemp("irc"), n_bkg=1200, n_sig=300) as root:
        run(copy.deepcopy(CFG), "tiny", stages=("train",), log=lambda *a: None)
        yield root


def jets_by_pt(const: np.ndarray) -> np.ndarray:
    """Per-jet constituents as a sorted set of rows (padding removed), for order-free comparisons."""
    out = []
    for jet in const:
        real = jet[jet[:, 0] > 0]
        out.append(real[np.lexsort((real[:, 2], real[:, 1], real[:, 0]))])
    return out


# ------------------------------------------------------------------------------------- perturbations
def test_collinear_splitting_conserves_pt_and_doubles_constituents():
    pool = synthetic_pool(20, 15, seed=1)
    for k in (1, 5, None):
        out = split_top(pool, k)
        real0, real1 = (pool.const[..., 0] > 0).sum(1), (out.const[..., 0] > 0).sum(1)
        want = real0 + np.minimum(real0, 15 if k is None else k)
        assert np.array_equal(real1, want)
        assert np.allclose(out.const[..., 0].sum(1), pool.const[..., 0].sum(1), rtol=1e-6)
    top = split_top(pool, 1).const
    assert np.allclose(top[:, 0, 0], pool.const[:, 0, 0] / 2) and np.array_equal(
        top[:, 0, 1:], pool.const[:, 0, 1:]
    )


def test_soft_emission_adds_constituents_with_the_given_pt_fraction():
    pool = synthetic_pool(20, 15, seed=2)
    out = add_soft(pool, 10, 0.01)
    soft = out.const[:, 15:]
    assert np.allclose(soft[..., 0], 0.01 * pool.const[..., 0].sum(1)[:, None], rtol=1e-6)
    dr = np.hypot(soft[..., 1] - pool.eta[:, None], soft[..., 2] - pool.phi[:, None])
    assert (dr <= 0.8 + 1e-5).all()
    assert np.array_equal(out.const[:, :15], pool.const)


def test_reorder_permutes_the_stored_constituents():
    pool = synthetic_pool(30, 20, seed=3)
    out = reorder(pool, seed=5)
    assert not np.array_equal(out.const, pool.const)
    assert all(
        np.array_equal(a, b) for a, b in zip(jets_by_pt(out.const), jets_by_pt(pool.const), strict=True)
    )
    assert np.array_equal(reorder(pool, seed=5).const, out.const)  # seeded
    assert "random re-order" in PERTURBATIONS


# ------------------------------------------------------------------------------------------- k-NN ties
def test_knn_tie_counts_find_exactly_equidistant_neighbours():
    pool = synthetic_pool(16, 20, seed=4)
    scaler = fit_scaler(pool)
    base = knn_tie_counts(pool, scaler, 3)
    assert base["jets_with_tie"] == 0 and base["nodes_with_tie"] == 0 and base["n_jets"] == 16
    const = pool.const.copy()
    const[0, :10] = const[0, 0]  # ten constituents at the same point: all their distances to each other are 0
    const[0, 10:] = 0.0
    tied = type(pool)(**{**{k: getattr(pool, k) for k in pool.__dataclass_fields__}, "const": const})
    c = knn_tie_counts(tied, scaler, 3)
    assert c["jets_with_tie"] == 1 and c["nodes_with_tie"] == 10 and c["k"] == 3
    assert c["jet_fraction"] == pytest.approx(1 / 16)
    # k too large for the jet (needs k + 1 other real nodes): no tie can be counted
    assert (
        knn_tie_counts(tied, scaler, 9)["nodes_with_tie"] == 0
        or knn_tie_counts(tied, scaler, 9)["jets_with_tie"] == 1
    )
    # invariant under the order of the stored constituents
    shuffled = permute_nodes(tied, seed=8, pad_extra=4)
    assert knn_tie_counts(shuffled, scaler, 3) == {**c, "n_nodes": c["n_nodes"]}


# ------------------------------------------------------------------------------------------- the probe
def test_probe_on_a_trained_model_writes_irc_probe_json(env):
    out = irc_probe.probe("tiny", n_events=300, log=lambda *a: None)
    saved = load_json(env / "results" / "runs" / "tiny" / "irc_probe.json")
    assert saved["name"] == "tiny" and 0.0 <= saved["auc"] <= 1.0 and saved["sr"] == [3300.0, 3700.0]
    assert set(PERTURBATIONS) <= set(saved)
    for label in PERTURBATIONS:
        e = saved[label]
        assert (
            set(e) >= {"auc", "rho", "flip_pct_at_1pct"}
            and 0.0 <= e["auc"] <= 1.0
            and -1.0 <= e["rho"] <= 1.0
        )
    # a permutation-invariant model does not see the order of the stored constituents
    ro = saved["random re-order"]
    assert ro["rho"] > 0.9999 and ro["max_abs_diff"] < 1e-4 * max(1.0, abs(out["auc"]))
    assert saved["split all"]["rho"] < 1.0  # this (IRC-unsafe) model does see the splittings
    ties = saved["knn_ties"]
    assert ties["bkg"]["k"] == 4  # model_args.k of the run
    for side in ("bkg", "sig"):
        t = ties[side]
        assert (
            0 <= t["jets_with_tie"] <= t["n_jets"]
            and t["jets_with_tie"] <= t["nodes_with_tie"] <= t["n_nodes"]
        )
        assert t["jet_fraction"] == pytest.approx(t["jets_with_tie"] / t["n_jets"])
    assert saved["n_bkg_jets"] == ties["bkg"]["n_jets"] and saved["n_sig_jets"] == ties["sig"]["n_jets"]


def test_cli_irc_probe_takes_k_and_run_names(env):
    d = env / "results" / "runs" / "tiny" / "irc_probe.json"
    d.unlink()
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(["irc-probe", "tiny", "-k", "6"]) == 0
    assert load_json(d)["knn_ties"]["bkg"]["k"] == 6
    assert Data(n_const=50).has("bkg_val")  # the probe does not touch the data directory
