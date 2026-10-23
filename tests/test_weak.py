import numpy as np

from gaehep.eval import weak


def _tab(n, rng, sig=False):
    m = np.stack([rng.normal(500 if sig else 150, 40, n), rng.normal(100 if sig else 120, 30, n)], 1)
    tau1 = rng.uniform(0.2, 1, (n, 2))
    tau2 = tau1 * (rng.uniform(0.1, 0.3, (n, 2)) if sig else rng.uniform(0.3, 0.9, (n, 2)))
    return {"mjj": rng.uniform(3300, 3700, n), "m": np.abs(m), "tau1": tau1, "tau2": tau2}


def test_hl4_and_latent_features_are_jet_order_invariant():
    rng = np.random.default_rng(0)
    t = _tab(50, rng)
    lat = rng.normal(size=(50, 2, 3))
    sw = {k: (v[:, ::-1] if v.ndim == 2 else v) for k, v in t.items()}
    assert np.allclose(weak.hl4(t), weak.hl4(sw))
    assert np.allclose(weak.latent_features(t, lat, True), weak.latent_features(sw, lat[:, ::-1], True))
    assert np.all(weak.hl4(t)[:, 1] >= 0)  # Δm with the lighter jet first


def test_iad_finds_injected_signal():
    rng = np.random.default_rng(1)
    feats = {
        "hl4": {
            "bkg_train": weak.hl4(_tab(6000, rng)),
            "sig_train": weak.hl4(_tab(2000, rng, sig=True)),
            "bkg_test": weak.hl4(_tab(4000, rng)),
            "sig_test": weak.hl4(_tab(1000, rng, sig=True)),
        }
    }
    res = weak.run_iad(feats, n_inj=(300,), reps=(0,), n_models=2, log=lambda *a: None)
    rows = {r["n_inj"]: r for r in res["hl4"]["iad"]}
    assert rows[300]["auc"] > 0.9
    assert res["hl4"]["supervised"]["auc"] > 0.95
