"""Further permutation / padding invariance checks: batch composition, k-NN slot order, degenerate jets,
phi wrap-around and rigid shifts, event-level aggregation and training gradients."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import gaehep.models.edgeconv as edgeconv_mod
from gaehep.data.dataset import JetPool, make_batch, node_features, wrap_phi
from gaehep.eval.evaluate import aggregate
from gaehep.graphs.ops import masked_knn
from gaehep.models.edgeconv import EdgeConvBlock, EdgeConvEncoder
from gaehep.train.loop import score_pool
from tests.conftest import permute_nodes
from tests.test_invariance import MODELS, _model

NEEDS_PREPARE = ("deep_svdd", "rnd")
# the only encoder that does not build a k-NN graph: radius balls have no tie-breaking
KNN_FREE = ("ef_ae_ewmp",)


def _replace(pool: JetPool, **kw) -> JetPool:
    return JetPool(**{**{k: getattr(pool, k) for k in pool.__dataclass_fields__}, **kw})


def _prepared(name: str, pool: JetPool, scaler):
    m = _model(name)
    if name in NEEDS_PREPARE:
        m.prepare([make_batch(pool, np.arange(16), scaler)])
    return m


def _score(m, pool: JetPool, scaler, idx=None) -> torch.Tensor:
    idx = np.arange(len(pool)) if idx is None else idx
    with torch.no_grad():
        return m.score(make_batch(pool, idx, scaler))


def _with_jets(pool: JetPool, const: np.ndarray) -> JetPool:
    """Pool with new constituents; jet pT / E recomputed from them (kept positive for empty jets)."""
    pt = const[..., 0].sum(1)
    e = (const[..., 0] * np.cosh(const[..., 1])).sum(1)
    pt = np.where(pt > 0, pt, pool.pt).astype(np.float32)
    e = np.where(e > 0, np.maximum(e, pt), pool.e).astype(np.float32)
    return _replace(pool, const=const.astype(np.float32), pt=pt, e=e, n_const=(const[..., 0] > 0).sum(1))


# ------------------------------------------------------------------------------- 1. batch independence
@pytest.mark.parametrize("name", list(MODELS))
def test_score_pool_independent_of_batch_size(name, pool, scaler):
    """Catches cross-jet leakage in a batch (batch statistics, batch-wide normalisation, mis-indexed rows)."""
    m = _prepared(name, pool, scaler)
    ref = score_pool(m, pool, scaler, batch_size=len(pool))["score"]
    for bs in (1, 7, 64):
        s = score_pool(m, pool, scaler, batch_size=bs)["score"]
        assert s.shape == ref.shape
        np.testing.assert_allclose(s, ref, rtol=1e-4, atol=1e-5, err_msg=f"batch_size={bs}")


@pytest.mark.parametrize("name", list(MODELS))
def test_score_independent_of_padding_to_longest_jet(name, pool, scaler):
    """Catches a score that depends on the padded width (batch's longest jet) instead of the jet itself."""
    m = _prepared(name, pool, scaler)
    idx = np.arange(12)
    ref = _score(m, pool, scaler, idx)
    for i in idx:
        n = int((pool.const[i, :, 0] > 0).sum())
        single = _replace(pool.subset(np.array([i])), const=pool.const[i : i + 1, :n].copy())
        s = _score(m, single, scaler)
        assert torch.allclose(s, ref[i : i + 1], rtol=1e-4, atol=1e-5), (i, n, s, ref[i])


# --------------------------------------------------------------------------- 2. neighbour-slot order
def _shuffled_knn(seed: int):
    def knn(coords, mask, k, exclude_self=True):
        idx, valid = masked_knn(coords, mask, k, exclude_self)
        g = torch.Generator().manual_seed(seed)
        perm = torch.rand(idx.shape, generator=g).argsort(-1)  # an independent slot order per node
        # invalid slots are masked, so where they point must not matter either
        junk = torch.randint(0, coords.shape[1], idx.shape, generator=g)
        idx = torch.where(valid, idx, junk)
        return idx.gather(-1, perm), valid.gather(-1, perm)

    return knn


@pytest.mark.parametrize("aggr", ["mean", "max"])
def test_edgeconv_invariant_to_neighbour_slot_order(aggr, pool, scaler, monkeypatch):
    """Catches aggregation that depends on the order of the k neighbour slots (or ignores ``valid``)."""
    torch.manual_seed(0)
    blk = EdgeConvBlock(7, 8, k=5, aggr=aggr).eval()
    enc = EdgeConvEncoder(7, (8, 8), k=5, aggr=aggr).eval()
    # jets with fewer real nodes than k, so that invalid (self-pointing) slots get shuffled in too
    const = pool.const[:6].copy()
    const[0, 3:] = 0.0
    const[1, 1:] = 0.0
    b = make_batch(_with_jets(pool.subset(np.arange(6)), const), np.arange(6), scaler)
    with torch.no_grad():
        y0 = blk(b.x, b.mask, b.coords)
        h0 = enc(b.x, b.mask, b.coords)
        monkeypatch.setattr(edgeconv_mod, "masked_knn", _shuffled_knn(seed=7))
        y1 = blk(b.x, b.mask, b.coords)
        h1 = enc(b.x, b.mask, b.coords)
    assert torch.allclose(y0, y1, atol=1e-5), (y0 - y1).abs().max()
    assert torch.allclose(h0, h1, atol=1e-5), (h0 - h1).abs().max()
    assert torch.allclose(enc.pool(h0, b.mask), enc.pool(h1, b.mask), atol=1e-5)


# --------------------------------------------------------------------------------- 3. degenerate jets
def _degenerate_pool(pool: JetPool) -> JetPool:
    const = pool.const[:8].copy()
    const[0] = 0.0  # entirely padding
    const[1, 1:] = 0.0  # one real constituent
    const[2, 2:] = 0.0  # two real constituents
    const[3] = 0.0
    const[3, :6] = pool.const[3, 0]  # six exact duplicates (same pT, eta, phi)
    const[4, 2] = const[4, 0]  # one duplicated constituent inside a normal jet
    return _with_jets(pool.subset(np.arange(8)), const)


@pytest.mark.parametrize("name", list(MODELS))
def test_degenerate_jets_finite_and_permutation_invariant(name, pool, scaler):
    """Catches NaN/inf (0/0, empty max, log 0) and order dependence on empty, 1-2 node and duplicated jets."""
    m = _prepared(name, pool, scaler)
    deg = _degenerate_pool(pool)
    s0 = _score(m, deg, scaler)
    assert torch.isfinite(s0).all(), s0
    for seed, pad in ((11, 0), (12, 6)):
        s1 = _score(m, permute_nodes(deg, seed=seed, pad_extra=pad), scaler)
        assert torch.allclose(s0, s1, rtol=1e-4, atol=1e-5), (seed, pad, (s0 - s1).abs().max())


@pytest.mark.parametrize(
    "name",
    [
        n
        if n in KNN_FREE
        else pytest.param(
            n,
            marks=pytest.mark.xfail(
                strict=True,
                reason="masked_knn breaks distance ties by storage index: constituents at identical "
                "(eta, phi) but different pT give an order-dependent k-NN graph and score",
            ),
        )
        for n in MODELS
    ],
)
def test_coincident_constituents_with_distinct_pt(name, pool, scaler):
    """Catches k-NN tie-breaking by node index when more than k constituents share one (eta, phi)."""
    m = _prepared(name, pool, scaler)
    const = pool.const[:4].copy()
    const[0] = 0.0
    const[0, :6, 1:] = pool.const[0, 0, 1:]  # six constituents at one point (k = 4 neighbours)
    const[0, :6, 0] = [50.0, 40.0, 30.0, 20.0, 10.0, 5.0]
    deg = _with_jets(pool.subset(np.arange(4)), const)
    s0 = _score(m, deg, scaler)
    assert torch.isfinite(s0).all()
    diff = max(float((s0 - _score(m, permute_nodes(deg, seed=s), scaler)).abs().max()) for s in range(6))
    assert diff < 1e-5, diff


# --------------------------------------------------------------------------------------- 4. phi / eta
def _moved(pool: JetPool, d_eta: float = 0.0, phi_to: float | None = None, d_phi: float = 0.0) -> JetPool:
    """Rigidly move every jet (axis and constituents) by d_eta in eta and to phi_to (or by d_phi) in phi."""
    const = pool.const.copy()
    real = const[..., 0] > 0
    new_phi = wrap_phi(pool.phi.astype(np.float64) + d_phi) if phi_to is None else np.full(len(pool), phi_to)
    rel = wrap_phi(const[..., 2].astype(np.float64) - pool.phi[:, None].astype(np.float64))
    const[..., 1] = np.where(real, const[..., 1] + d_eta, 0.0)
    const[..., 2] = np.where(real, wrap_phi(new_phi[:, None] + rel), 0.0)
    eta = (pool.eta + d_eta).astype(np.float32)
    e = (const[..., 0] * np.cosh(const[..., 1])).sum(1).astype(np.float32)
    return _replace(pool, const=const, eta=eta, phi=new_phi.astype(np.float32), e=np.maximum(e, pool.pt))


@pytest.mark.parametrize("name", list(MODELS))
def test_score_invariant_across_phi_wraparound(name, pool, scaler):
    """Catches a missing wrap_phi: jets straddling phi = +-pi (or rotated in phi) must score as at phi = 0."""
    m = _prepared(name, pool, scaler)
    idx = np.arange(16)
    base = _moved(pool, phi_to=0.0)
    s0 = _score(m, base, scaler, idx)
    for phi_to in (np.pi - 0.05, -np.pi + 0.05):
        moved = _moved(pool, phi_to=phi_to)
        straddles = (moved.const[idx, :, 2] > 0).any(1) & (moved.const[idx, :, 2] < 0).any(1)
        assert straddles.sum() >= 3  # the test really crosses the boundary
        s1 = _score(m, moved, scaler, idx)
        assert torch.allclose(s0, s1, rtol=1e-4, atol=1e-5), (phi_to, (s0 - s1).abs().max())
    s2 = _score(m, _moved(pool, d_phi=2.3), scaler, idx)
    assert torch.allclose(_score(m, pool, scaler, idx), s2, rtol=1e-4, atol=1e-5)


def test_node_features_invariant_to_rigid_shift(pool):
    """Catches jet-relative features that are not relative (phi shifts: all 7; eta shifts: all but log E)."""

    def feats(p):
        return node_features(p.const, p.pt, p.eta, p.phi, p.e)

    f0, m0 = feats(pool)
    f1, m1 = feats(_moved(pool, d_phi=3.0))
    assert np.array_equal(m0, m1)
    np.testing.assert_allclose(f0, f1, atol=2e-5)
    f2, m2 = feats(_moved(pool, d_eta=0.7))
    assert np.array_equal(m0, m2)
    boost_inv = [0, 1, 2, 4, 6]  # deta, dphi, log pT, log pT/pT_jet, dR (log E, log E/E_jet change by design)
    np.testing.assert_allclose(f0[..., boost_inv], f2[..., boost_inv], atol=2e-5)


@pytest.mark.parametrize("name", KNN_FREE)
def test_direction_only_encoder_invariant_to_eta_shift(name, pool, scaler):
    """Catches an absolute eta (or energy) leaking into the IRC-safe model, which only sees (deta, dphi, dR, z)."""
    m = _prepared(name, pool, scaler)
    idx = np.arange(16)
    s0 = _score(m, pool, scaler, idx)
    for d_eta in (-0.9, 0.6):
        s1 = _score(m, _moved(pool, d_eta=d_eta), scaler, idx)
        assert torch.allclose(s0, s1, rtol=1e-4, atol=1e-5), (d_eta, (s0 - s1).abs().max())


# ------------------------------------------------------------------------------------------ 5. events
@pytest.mark.parametrize("how", ["max", "min", "sum", "prod"])
def test_aggregate_symmetric_in_the_two_jets(how):
    """Catches an event score that depends on which jet is leading (column order)."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=(200, 2))
    ref = np.sort(rng.normal(size=500))
    s[:20] = rng.choice(ref, size=(20, 2))  # exact ties with the reference
    a = aggregate(s, how, ref=ref)
    b = aggregate(s[:, ::-1].copy(), how, ref=ref)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_prod_aggregate_invariant_to_monotone_rescaling():
    """Catches 'prod' depending on the raw score scale instead of the background-CDF rank."""
    rng = np.random.default_rng(1)
    s = rng.normal(size=(300, 2))
    ref = np.sort(rng.normal(size=400))
    s[:30] = rng.choice(ref, size=(30, 2))
    a = aggregate(s, "prod", ref=ref)
    for f in (np.exp, lambda x: 3.0 * x + 7.0, lambda x: x**3 + x):
        np.testing.assert_array_equal(aggregate(f(s), "prod", ref=np.sort(f(ref))), a)


# --------------------------------------------------------------------------------------- 6. gradients
GRAD_MODELS = ("set_ae", "ef_ae", "ef_ae_vae_recon", "ef_ae_ewmp")


def _grads(m, b) -> dict[str, torch.Tensor]:
    m.train()
    m.zero_grad(set_to_none=True)
    torch.manual_seed(123)  # the VAE samples its latent in training mode
    loss = m.losses(b)["total"].mean()  # the loss fit() minimises
    assert torch.isfinite(loss)
    loss.backward()
    return {k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("name", GRAD_MODELS)
def test_training_gradients_invariant_to_node_order(name, pool, scaler):
    """Catches an order-dependent training signal even when the forward score is invariant."""
    m = _model(name)
    idx = np.arange(16)
    g0 = _grads(m, make_batch(pool, idx, scaler))
    assert g0 and all(torch.isfinite(g).all() for g in g0.values())
    for seed, pad in ((21, 0), (22, 7)):
        g1 = _grads(m, make_batch(permute_nodes(pool, seed=seed, pad_extra=pad), idx, scaler))
        assert g1.keys() == g0.keys()
        for k in g0:
            assert torch.allclose(g0[k], g1[k], atol=1e-5), (seed, pad, k, (g0[k] - g1[k]).abs().max())
