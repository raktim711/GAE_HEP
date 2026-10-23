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
