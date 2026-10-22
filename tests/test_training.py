"""End-to-end smoke tests of the training loop, resume, scoring and evaluation on synthetic jets."""

import numpy as np
import torch

from gaehep.eval.evaluate import evaluate_scores, headline
from gaehep.models import build_model
from gaehep.train.loop import TrainConfig, fit, score_pool
from gaehep.train.run import concat_pools
from tests.conftest import synthetic_pool


def _cfg(**kw):
    base = dict(
        epochs=2, jets_per_epoch=64, batch_size=16, val_jets=32, patience=5, threads=1, prepare_batches=2
    )
    base.update(kw)
    return TrainConfig(**base)


def test_fit_resume_and_score(tmp_path, scaler):
    train, val = synthetic_pool(96, seed=1), synthetic_pool(48, seed=2)
    torch.manual_seed(0)
    m = build_model("set_ae", hidden=(16, 16), k=4, latent_dim=4, n_out=30, dec_hidden=(32,))
    s = fit(m, train, val, scaler, _cfg(epochs=1), tmp_path)
    assert s["status"] == "done" and (tmp_path / "model.pt").exists()
    # resume to more epochs from the checkpoint
    m2 = build_model("set_ae", hidden=(16, 16), k=4, latent_dim=4, n_out=30, dec_hidden=(32,))
    s2 = fit(m2, train, val, scaler, _cfg(epochs=3), tmp_path)
    assert s2["epochs_done"] == 3 and len(s2["history"]) == 3
    out = score_pool(m2, val, scaler, batch_size=20, with_embed=True)
    assert out["score"].shape == (48,) and out["z"].shape == (48, 4)
    assert np.all(np.isfinite(out["score"]))


def test_time_budget_pauses(tmp_path, scaler):
    train, val = synthetic_pool(96, seed=1), synthetic_pool(48, seed=2)
    m = build_model("node_gae", enc_hidden=(8,), node_latent=2, dec_hidden=(8,), k=4)
    s = fit(m, train, val, scaler, _cfg(epochs=5, time_budget_min=0.0), tmp_path)
    assert s["status"] == "paused" and s["epochs_done"] == 1


def test_deep_svdd_and_classifier_train(tmp_path, scaler):
    train, val = synthetic_pool(96, seed=1), synthetic_pool(48, seed=2)
    m = build_model("deep_svdd", hidden=(8, 8), k=4, latent_dim=4)
    s = fit(m, train, val, scaler, _cfg(), tmp_path / "svdd")
    assert s["status"] == "done" and bool(m.center_ready)
    sig_tr, sig_va = (
        synthetic_pool(96, seed=3, prongs=2, label=1),
        synthetic_pool(48, seed=4, prongs=2, label=1),
    )
    c = build_model("classifier", hidden=(8, 8), k=4)
    s = fit(
        c,
        concat_pools([train, sig_tr]),
        concat_pools([val, sig_va]),
        scaler,
        _cfg(epochs=3),
        tmp_path / "clf",
    )
    assert s["history"][-1]["train_loss"] < s["history"][0]["train_loss"] + 1e-6


def test_evaluate_scores_structure():
    rng = np.random.default_rng(0)
    n_b, n_s = 4000, 600

    def tab(n, sig):
        mjj = rng.uniform(2500, 4500, n) if not sig else rng.normal(3500, 100, n)
        return {"mjj": mjj, "m": rng.uniform(20, 500, (n, 2)), "n_const": rng.integers(10, 80, (n, 2))}

    tables = {"bkg_test": tab(n_b, False), "sig_test": tab(n_s, True), "sig3_test": tab(n_s, True)}
    scores = {
        "bkg_test": rng.normal(0, 1, (n_b, 2)),
        "sig_test": rng.normal(1, 1, (n_s, 2)),
        "sig3_test": rng.normal(0.5, 1, (n_s, 2)),
    }
    met = evaluate_scores(scores, tables, n_boot=3)
    h = headline(met)
    assert 0.6 < h["auc_2p_full"] < 0.95 and h["auc_2p_full"] > h["auc_3p_full"]
    assert set(met["event"]) == {"max", "min", "sum"}
    assert "dcor_mj" in met["correlation"]["jet"]
