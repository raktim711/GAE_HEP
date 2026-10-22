"""Feature construction, clustering, tables, splits — on synthetic inputs (no LHCO data needed)."""

import numpy as np
import pytest

from gaehep.data.dataset import fit_scaler, make_batch, node_features, wrap_phi
from gaehep.data.preprocess import cluster_events, n_rows, read_events
from gaehep.data.splits import check_disjoint, make_splits
from gaehep.data.tables import dijet_mass, jet_observables, match_features


def test_wrap_phi():
    x = np.array([0.0, np.pi + 0.1, -np.pi - 0.1, 3 * np.pi])
    w = wrap_phi(x)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.cos(w), np.cos(x)) and np.allclose(np.sin(w), np.sin(x))


def test_node_features_hand_example():
    c = np.zeros((1, 3, 3), dtype=np.float32)
    c[0, 0] = [100.0, 0.5, 3.1]
    c[0, 1] = [50.0, 0.6, -3.1]  # across the phi = +-pi boundary
    f, mask = node_features(c, np.array([150.0]), np.array([0.5]), np.array([3.1]), np.array([170.0]))
    assert mask.tolist() == [[True, True, False]]
    assert np.allclose(f[0, 0, :2], 0.0, atol=1e-6)
    assert abs(f[0, 1, 1] - wrap_phi(np.array([-6.2]))[0]) < 1e-4 and abs(f[0, 1, 1]) < 0.1
    assert np.isclose(f[0, 0, 2], np.log(100.0))
    assert np.isclose(f[0, 0, 4], np.log(100.0 / 150.0))
    assert np.all(f[0, 2] == 0.0)


def test_scaler_and_batch_shapes(pool):
    sc = fit_scaler(pool)
    b = make_batch(pool, np.arange(5), sc)
    assert b.x.shape == (5, pool.n_nodes, 7) and b.coords.shape == (5, pool.n_nodes, 2)
    assert b.jet.shape == (5, 3) and b.raw.shape == (5, pool.n_nodes, 3)
    assert np.all(b.x.numpy()[~b.mask.numpy()] == 0)
    feats, mask = node_features(pool.const, pool.pt, pool.eta, pool.phi, pool.e)
    z = (feats[mask] - sc.node_mean) / sc.node_std
    assert np.allclose(z.mean(0), 0, atol=1e-3) and np.allclose(z.std(0), 1, atol=1e-2)


def _event(groups):
    """groups: list of (n, pt, eta, phi) -> one zero-padded (700, 3) event."""
    rows = []
    for n, pt, eta, phi in groups:
        for i in range(n):
            rows.append([pt * (1 + 0.01 * i), eta + 0.01 * i, phi - 0.01 * i])
    ev = np.zeros((700, 3))
    if rows:
        ev[: len(rows)] = rows
    return ev


def _p4(c):
    pt, eta, phi = c[..., 0].astype(float), c[..., 1].astype(float), c[..., 2].astype(float)
    return np.array(
        [
            np.sum(pt * np.cos(phi)),
            np.sum(pt * np.sin(phi)),
            np.sum(pt * np.sinh(eta)),
            np.sum(pt * np.cosh(eta)),
        ]
    )


def test_cluster_events_two_jets_and_constituents():
    parts = np.stack(
        [
            _event([(5, 100.0, 0.0, 0.0), (3, 300.0, 1.0, 3.0), (1, 5.0, -2.4, -1.5)]),
            _event([]),
            _event([(120, 10.0, 0.2, 1.0)]),  # one jet with more constituents than n_max
        ]
    )
    out = cluster_events(parts, n_max=100)
    assert out["n_jets"].tolist() == [2, 0, 1]
    # leading jet (by pT) is the 3 x 300 GeV group
    assert out["n_const"][0].tolist() == [3, 5]
    pts = out["constituents"][0, 0, :, 0]
    assert np.all(np.diff(pts[:3]) <= 0) and np.all(pts[3:] == 0)
    for k in (0, 1):
        assert np.allclose(_p4(out["constituents"][0, k]), out["jets_p4"][0, k], rtol=1e-6)
    assert np.all(np.isnan(out["jets_p4"][1])) and out["n_const"][1].tolist() == [0, 0]
    assert out["n_const"][2].tolist() == [120, 0]
    assert np.count_nonzero(out["constituents"][2, 0, :, 0]) == 100


def test_read_events_and_n_rows(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    rng = np.random.default_rng(0)
    arr = rng.random((25, 2101))
    arr[:, -1] = rng.integers(0, 2, 25)
    f = tmp_path / "toy.h5"
    pd.DataFrame(arr).to_hdf(f, key="df", mode="w", complevel=5, complib="blosc")
    assert n_rows(f) == 25
    parts, labels = read_events(f, 5, 15)
    assert parts.shape == (10, 700, 3)
    assert np.allclose(parts[0].ravel(), arr[5, :2100])
    assert labels.tolist() == arr[5:15, -1].astype(int).tolist()


def test_observables_and_matching():
    p4 = np.array([[[3.0, 4.0, 0.0, 13.0], [-3.0, -4.0, 0.0, 13.0]]])  # pt 5, m 12 each, back to back
    obs = jet_observables(p4)
    assert np.allclose(obs["pt"], 5.0) and np.allclose(obs["m"], 12.0) and np.allclose(obs["eta"], 0.0)
    assert np.allclose(dijet_mass(p4), 26.0)
    feat = {
        "pxj1": np.array([-3.0]),
        "pyj1": np.array([-4.0]),
        "pzj1": np.array([0.0]),
        "mj1": np.array([12.0]),
        "pxj2": np.array([3.0]),
        "pyj2": np.array([4.0]),
        "pzj2": np.array([0.0]),
        "mj2": np.array([12.0]),
    }
    m = match_features(p4, feat)
    assert m["matched"].tolist() == [True] and m["swapped"].tolist() == [True]
    feat["mj2"] = np.array([20.0])
    assert match_features(p4, feat)["matched"].tolist() == [False]


def test_splits_disjoint_and_deterministic():
    rng = np.random.default_rng(0)
    labels = (rng.random(10_000) < 0.1).astype(np.int8)
    n_jets = np.where(rng.random(10_000) < 0.01, 1, 2)
    s1 = make_splits(n_jets, labels, np.full(500, 2), seed=7)
    s2 = make_splits(n_jets, labels, np.full(500, 2), seed=7)
    check_disjoint(s1)
    for k in s1:
        assert np.array_equal(s1[k], s2[k])
    bkg = np.r_[s1["bkg_train"], s1["bkg_val"], s1["bkg_test"]]
    assert np.all(labels[bkg] == 0) and np.all(n_jets[bkg] >= 2)
    assert len(bkg) == np.sum((labels == 0) & (n_jets >= 2))
    sig = np.r_[s1["sig_train"], s1["sig_val"], s1["sig_test"]]
    assert np.all(labels[sig] == 1)
    assert abs(len(s1["bkg_train"]) / len(bkg) - 0.6) < 0.01
