"""A full run (train + score) is reproducible from its config: same ``train.seed`` -> identical scores."""

import numpy as np

CFG = {
    "model": "set_ae",
    "model_args": {"hidden": [8, 8], "k": 4, "latent_dim": 2, "n_out": 20, "dec_hidden": [16]},
    "data": {"n_const": 20, "latent_train_events": 200},
    "train": {
        "epochs": 2,
        "jets_per_epoch": 300,
        "val_jets": 100,
        "batch_size": 64,
        "prepare_batches": 2,
        "threads": 1,
    },
}


def test_same_seed_same_scores(tmp_path, monkeypatch):
    from gaehep.config import results_dir
    from gaehep.data.fake import make_fake
    from gaehep.train.run import run

    make_fake(tmp_path / "data", 1200, 300, 20)
    monkeypatch.setenv("GAEHEP_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("GAEHEP_RESULTS", str(tmp_path / "results"))
    quiet = lambda *a: None  # noqa: E731
    scores = {}
    for name, seed in (("a", 1), ("b", 1), ("c", 2)):
        # fresh run names: a finished run directory would be reused instead of retrained
        run({**CFG, "train": {**CFG["train"], "seed": seed}}, name, stages=("train", "score"), log=quiet)
        with np.load(results_dir() / "runs" / name / "scores.npz") as f:
            scores[name] = {k: f[k] for k in f.files}
    assert set(scores["a"]) == {"bkg_val", "bkg_test", "sig_test", "sig3_test"}
    # bitwise equal: one thread, CPU, same data order (default_rng([seed, epoch])) and initial weights
    assert all(np.array_equal(scores["a"][k], scores["b"][k]) for k in scores["a"])
    assert not any(np.array_equal(scores["a"][k], scores["c"][k]) for k in scores["a"])
