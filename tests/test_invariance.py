"""Permutation and padding invariance of every model's anomaly score (the core design requirement)."""

import numpy as np
import pytest
import torch

from gaehep.data.dataset import make_batch
from gaehep.models import build_model
from gaehep.models.edgeconv import EdgeConvBlock, EdgeConvEncoder
from tests.conftest import permute_nodes

# set_ae compares its first n emitted points with the n real constituents, so n_out must cover the padded
# node count of the test batches (30 stored + 7 extra padding slots)
MODELS = {
    "set_ae": dict(hidden=(16, 16), k=4, latent_dim=4, n_out=40, dec_hidden=(32,)),
    "set_ae_cond": dict(hidden=(16, 16), k=4, latent_dim=4, n_out=40, dec_hidden=(32,), cond_dim=2),
    "set_ae_sinkhorn": dict(hidden=(16, 16), k=4, latent_dim=4, n_out=40, dec_hidden=(32,), loss="sinkhorn"),
    "node_gae": dict(enc_hidden=(16, 16), node_latent=2, dec_hidden=(16,), k=4),
    "deep_svdd": dict(hidden=(16, 16), k=4, latent_dim=4),
    "classifier": dict(hidden=(16, 16), k=4, head_hidden=(8,)),
    "rnd": dict(hidden=(16, 16), teacher_hidden=(16, 16), k=4, out_dim=6, head_hidden=(16,)),
    "set_ae_max": dict(hidden=(16, 16), k=4, latent_dim=4, n_out=40, dec_hidden=(32,), aggr="max"),
    "ef_ae": dict(hidden=(16, 16), k=4, latent_dim=4, n_out=8, dec_hidden=(32,)),
    "ef_ae_cond": dict(hidden=(16, 16), k=4, latent_dim=4, n_out=8, dec_hidden=(32,), cond_dim=2),
    "ef_ae_vae_recon": dict(  # the final-model configuration (score = reconstruction)
        hidden=(16, 16), k=4, latent_dim=4, n_out=8, dec_hidden=(32,), variational=True
    ),
    "ef_ae_nll": dict(hidden=(16, 16), k=4, latent_dim=4, n_out=8, dec_hidden=(32,), loss="nll"),
    "ef_ae_align_pow": dict(
        hidden=(16, 16), k=4, latent_dim=4, n_out=8, dec_hidden=(32,), align=True, energy_power=2.0
    ),
    "ef_ae_ewmp": dict(
        hidden=(16, 16),
        latent_dim=4,
        n_out=8,
        dec_hidden=(32,),
        encoder="ewmp",
        radii=(0.1, 0.4),
        variational=True,
    ),
    "ef_ae_vae_kl": dict(
        hidden=(16, 16), k=4, latent_dim=4, n_out=8, dec_hidden=(32,), variational=True, score="kl"
    ),
}


def _model(name):
    torch.manual_seed(0)
    key = next((k for k in ("set_ae", "ef_ae") if name.startswith(k)), name)
    m = build_model(key, **MODELS[name]).eval()
    return m


@pytest.mark.parametrize("name", list(MODELS))
def test_score_invariant_to_node_order_and_padding(name, pool, scaler):
    """Scores are unchanged when the stored constituents are permuted, and when they are permuted *and*
    7 extra padding slots are added (padding interleaved with the real nodes)."""
    m = _model(name)
    idx = np.arange(12)
    b0 = make_batch(pool, idx, scaler)
    if name in ("deep_svdd", "rnd"):
        m.prepare([b0])
    perm_pool = permute_nodes(pool, seed=3)
    pad_pool = permute_nodes(pool, seed=4, pad_extra=7)
    b2 = make_batch(pad_pool, idx, scaler)
    assert b2.x.shape[1] == b0.x.shape[1] + 7
    with torch.no_grad():
        s0 = m.score(b0)
        s1 = m.score(make_batch(perm_pool, idx, scaler))
        s2 = m.score(b2)
    assert torch.allclose(s0, s1, rtol=1e-4, atol=1e-5), (s0 - s1).abs().max()
    assert torch.allclose(s0, s2, rtol=1e-4, atol=1e-5), (s0 - s2).abs().max()


@pytest.mark.parametrize("conv", ["gcn", "gin", "gat", "sage"])
def test_legacy_gae_score_invariant(conv, pool, scaler):
    pytest.importorskip("torch_geometric")
    torch.manual_seed(0)
    m = build_model(
        "legacy_gae", in_dim=3, hidden_channels=16, latent_channels=8, dropout=0.0, conv_type=conv
    ).eval()
    idx = np.arange(8)
    with torch.no_grad():
        s0 = m.score(make_batch(pool, idx, scaler))
        s1 = m.score(make_batch(permute_nodes(pool, seed=6, pad_extra=5), idx, scaler))
    assert torch.allclose(s0, s1, rtol=1e-4, atol=1e-3), (s0 - s1).abs().max()


@pytest.mark.parametrize("name", list(MODELS))
def test_score_invariant_to_extra_padding(name, pool, scaler):
    m = _model(name)
    idx = np.arange(10)
    b0 = make_batch(pool, idx, scaler)
    if name in ("deep_svdd", "rnd"):
        m.prepare([b0])
    padded = permute_nodes(pool, seed=5, pad_extra=7)
    with torch.no_grad():
        assert torch.allclose(m.score(b0), m.score(make_batch(padded, idx, scaler)), rtol=1e-4, atol=1e-5)


def test_edgeconv_block_is_equivariant(pool, scaler):
    torch.manual_seed(0)
    blk = EdgeConvBlock(7, 8, k=5).eval()
    b = make_batch(pool, np.arange(6), scaler)
    perm = torch.randperm(b.x.shape[1])
    with torch.no_grad():
        y = blk(b.x, b.mask, b.coords)
        yp = blk(b.x[:, perm], b.mask[:, perm], b.coords[:, perm])
    assert torch.allclose(y[:, perm], yp, atol=1e-5)


def test_edgeconv_matches_pyg(pool, scaler):
    pyg = pytest.importorskip("torch_geometric.nn")
    from gaehep.graphs.ops import masked_knn

    torch.manual_seed(0)
    blk = EdgeConvBlock(7, 8, k=5, aggr="mean").eval()
    b = make_batch(pool, np.arange(4), scaler)
    # PyG's EdgeConv re-initialises the module it wraps: give it a copy built from our weights, then
    # restore the copy's parameters (reset happens inside the constructor)
    eq = blk.edge_mlp_equivalent()
    state = {k: v.clone() for k, v in eq.state_dict().items()}
    conv = pyg.EdgeConv(eq, aggr="mean")
    eq.load_state_dict(state)
    with torch.no_grad():
        ours = blk(b.x, b.mask, b.coords)
        idx, valid = masked_knn(b.coords, b.mask, 5)
        bsz, n = b.mask.shape
        g, i, j = valid.nonzero(as_tuple=True)
        src = g * n + idx[g, i, j]
        dst = g * n + i
        x_flat = b.x.reshape(bsz * n, -1)
        agg = conv(x_flat, torch.stack([src, dst]))
        ref = blk.act(agg + blk.shortcut(x_flat)).view(bsz, n, -1) * b.mask[..., None]
    assert torch.allclose(ours, ref, atol=1e-5)


def test_encoder_pool_invariant(pool, scaler):
    torch.manual_seed(0)
    enc = EdgeConvEncoder(7, (8, 8), k=4).eval()
    b = make_batch(pool, np.arange(5), scaler)
    perm = torch.randperm(b.x.shape[1])
    with torch.no_grad():
        p0 = enc.pool(enc(b.x, b.mask, b.coords), b.mask)
        p1 = enc.pool(enc(b.x[:, perm], b.mask[:, perm], b.coords[:, perm]), b.mask[:, perm])
    assert torch.allclose(p0, p1, atol=1e-5)


@pytest.mark.parametrize("name", list(MODELS))
def test_score_and_embed_consistent(name, pool, scaler):
    m = _model(name)
    b = make_batch(pool, np.arange(8), scaler)
    if name in ("deep_svdd", "rnd"):
        m.prepare([b])
    with torch.no_grad():
        s1, z1 = m.score_and_embed(b)
        s2, z2 = m.score(b), m.embed(b)
    assert torch.allclose(s1, s2, atol=1e-6)
    assert (z1 is None and z2 is None) or torch.allclose(z1, z2, atol=1e-6)


def test_vae_is_deterministic_in_eval_and_returns_all_terms(pool, scaler):
    m = _model("ef_ae_vae_kl")
    b = make_batch(pool, np.arange(8), scaler)
    with torch.no_grad():
        r1, r2 = m.losses(b), m.losses(b)
    assert torch.allclose(r1["score"], r2["score"]) and torch.allclose(r1["score"], r1["kl"])
    assert torch.all(r1["kl"] >= 0) and torch.all(r1["rec"] >= 0)
    assert torch.allclose(r1["total"], r1["rec"] + m.beta * r1["kl"])
    m.train()
    torch.manual_seed(1)
    with torch.no_grad():
        assert not torch.allclose(m.losses(b)["rec"], r1["rec"])  # sampled latent in training mode


def test_score_pool_extra_keys_match_single_pass(pool, scaler):
    from gaehep.train.loop import score_pool

    m = _model("ef_ae_vae_kl")
    sub = pool.subset(np.arange(10))
    a = score_pool(m, sub, scaler, batch_size=4, with_embed=True, keys=("rec", "kl"))
    b = score_pool(m, sub, scaler, batch_size=4, with_embed=True)
    assert np.allclose(a["score"], b["score"], atol=1e-6) and np.allclose(a["z"], b["z"], atol=1e-6)
    assert np.allclose(a["kl"], a["score"], atol=1e-6) and a["rec"].shape == (10,)
