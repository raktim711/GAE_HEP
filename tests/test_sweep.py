"""End-to-end check of the dev-protocol sweep on a tiny synthetic dataset (CPU, seconds)."""

import json

import pytest
import yaml

from gaehep.data.fake import make_fake
from gaehep.train.sweep import point_config, point_name


def test_point_config_and_name():
    base = {"model": "ef_ae", "model_args": {"hidden": [48, 48, 48], "k": 12}, "extra_scores": ["kl"]}
    p = {"latent_dim": 4, "beta": 0.0, "width": 128, "jets_per_epoch": 60_000}
    cfg = point_config(base, p, {"train.device": "cpu", "data.n_const": 30})
    assert cfg["model_args"]["variational"] is False and cfg["extra_scores"] == []
    assert cfg["model_args"]["hidden"] == [128, 128, 128] and cfg["model_args"]["k"] == 16
    assert cfg["train"]["jets_per_epoch"] == 60_000 and cfg["train"]["epochs"] == 5
    assert cfg["train"]["device"] == "cpu" and cfg["data"]["n_const"] == 30
    assert point_name("gpu", p) == "gpu_l4_b0_w128_j60k"
    assert point_name("gpu", {**p, "beta": 1e-3}) == "gpu_l4_b0.001_w128_j60k"
    assert base["model_args"]["hidden"] == [48, 48, 48]  # base config untouched


@pytest.mark.slow
def test_sweep_end_to_end(tmp_path, monkeypatch):
    make_fake(tmp_path / "data", 2400, 600, 20)
    monkeypatch.setenv("GAEHEP_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("GAEHEP_RESULTS", str(tmp_path / "results"))
    base = {
        "model": "ef_ae",
        "model_args": {"hidden": [8, 8], "k": 4, "n_out": 6, "dec_hidden": [16], "latent_dim": 2},
        "data": {"n_const": 20, "latent_train_events": 200},
    }
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text(yaml.safe_dump(base))
    from gaehep.train.sweep import sweep

    grid = {"latent_dim": [2, 3], "beta": [0.0, 0.001], "width": [8], "jets_per_epoch": [400]}
    fixed = {"train.epochs": 1, "train.val_jets": 200, "train.batch_size": 64, "train.device": "cpu"}
    out = sweep(str(cfg_path), prefix="t", grid=grid, overrides=fixed, top_k=2, n_seeds=2, final_seeds=3,
                log=lambda *a: None)
    assert len(out["phase1"]) == 4 and len(out["seeds"]) == 2
    assert out["promoted"] == ["t_best_s1", "t_best_s2", "t_best_s3"]
    runs = tmp_path / "results" / "runs"
    for r in (*out["promoted"], "t_best_seeds"):
        assert (runs / r / "metrics.json").exists(), r
    summary = json.loads((tmp_path / "results" / "dev" / "t_sweep.json").read_text())
    assert summary["best"]["name"] in {d["name"] for d in summary["phase1"]}
    assert (tmp_path / "results" / "dev" / "t_sweep.md").exists()
