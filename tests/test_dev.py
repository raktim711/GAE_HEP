import numpy as np

from gaehep.train.dev import dev_metrics, headline


def _tab(n, rng, sig=False):
    mjj = rng.uniform(3000, 4000, n) if not sig else rng.normal(3500, 80, n)
    m = np.stack([rng.uniform(50, 600, n), rng.uniform(30, 300, n)], 1)
    tau1 = rng.uniform(0.1, 1, (n, 2))
    return {
        "mjj": mjj,
        "m": m,
        "n_const": rng.integers(10, 100, (n, 2)),
        "tau1": tau1,
        "tau2": tau1 * rng.uniform(0.1, 1, (n, 2)),
        "label": np.full(n, int(sig)),
    }


def test_dev_metrics_shapes_and_signal_direction():
    rng = np.random.default_rng(0)
    tb, ts = _tab(4000, rng), _tab(800, rng, sig=True)
    sb = rng.normal(0, 1, (4000, 2))
    ss = rng.normal(1, 1, (800, 2))
    m = dev_metrics(sb, ss, tb, ts)
    assert m["SR/max"]["auc"] > 0.7 and m["full/sum"]["auc"] > 0.7
    assert abs(m["bkg_spearman"]["m"]) < 0.1  # scores independent of mass here
    assert 0.45 < dev_metrics(sb, rng.normal(0, 1, (800, 2)), tb, ts)["SR/jet"]["auc"] < 0.55
    assert "SR:" in headline(m)


def test_dev_ensemble(tmp_path, monkeypatch):
    import pytest

    from gaehep.data.fake import make_fake
    from gaehep.eval.metrics import auc
    from gaehep.train.dev import dev_dir, dev_ensemble
    from gaehep.train.run import Data

    make_fake(tmp_path / "data", 2400, 600, 20)
    monkeypatch.setenv("GAEHEP_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("GAEHEP_RESULTS", str(tmp_path / "results"))
    d = Data(n_const=20)
    nb, ns = len(d.events("bkg_val")), len(d.events("sig_val"))
    rng = np.random.default_rng(1)
    runs = {
        "good": (rng.normal(0, 1, (nb, 2)), rng.normal(1.5, 1, (ns, 2))),
        "random": (rng.normal(0, 1, (nb, 2)), rng.normal(0, 1, (ns, 2))),
        "short": (rng.normal(0, 1, (nb - 1, 2)), rng.normal(0, 1, (ns, 2))),
    }
    for name, (b, s) in runs.items():
        dev_dir(name).mkdir(parents=True)
        np.savez(dev_dir(name) / "dev_scores.npz", bkg_val=b, sig_val=s)
    quiet = lambda *a: None  # noqa: E731
    raw = auc(runs["good"][1].ravel(), runs["good"][0].ravel())
    same = dev_ensemble(["good", "good"], "good2", log=quiet)["metrics"]["full/jet"]["auc"]
    assert abs(same - raw) < 0.01  # CDF transform of one score: same ranking up to ties
    mixed = dev_ensemble(["good", "random"], "mixed", log=quiet)["metrics"]["full/jet"]["auc"]
    assert 0.55 < mixed < raw
    assert (dev_dir("mixed") / "dev_metrics.json").exists() and (dev_dir("mixed") / "dev_scores.npz").exists()
    with pytest.raises(ValueError):
        dev_ensemble(["good", "short"], "bad", log=quiet)
