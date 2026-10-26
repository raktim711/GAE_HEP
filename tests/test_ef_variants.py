"""Properties of the energy-flow variants of RESULTS.md §11: NLL loss, principal-axis frame, IRC-safe encoder."""

import numpy as np
import pytest
import torch

from gaehep.data.dataset import JetPool, make_batch
from gaehep.losses.mmd import energy_mmd, energy_nll
from gaehep.models import build_model
from gaehep.models.ef_ae import principal_axis_frame


def _split_one(pool: JetPool, frac: float = 0.3) -> JetPool:
    """Split the hardest constituent of every jet into two collinear parts (pT fractions frac, 1 - frac)."""
    n = pool.const.shape[1]
    const = np.zeros((len(pool), n + 1, 3), dtype=np.float32)
    const[:, :n] = pool.const
    hard = const[:, 0].copy()  # constituents are stored pT-ordered, index 0 is the hardest
    const[:, 0, 0] = hard[:, 0] * frac
    const[:, n] = hard
    const[:, n, 0] = hard[:, 0] * (1.0 - frac)
    return JetPool(**{**{k: getattr(pool, k) for k in pool.__dataclass_fields__}, "const": const})


def _add_soft(pool: JetPool, z: float) -> JetPool:
    n = pool.const.shape[1]
    const = np.zeros((len(pool), n + 1, 3), dtype=np.float32)
    const[:, :n] = pool.const
    k = (pool.const[..., 0] > 0).sum(1)
    for i in range(len(pool)):
        const[i, k[i]] = [z * pool.const[i, :, 0].sum(), pool.eta[i] + 0.4, pool.phi[i] - 0.5]
    return JetPool(**{**{k_: getattr(pool, k_) for k_ in pool.__dataclass_fields__}, "const": const})


EWMP = dict(hidden=(16, 16), latent_dim=4, n_out=8, dec_hidden=(32,), encoder="ewmp", radii=(0.1, 0.4))


@pytest.mark.parametrize(
    "extra", [dict(), dict(align=True), dict(loss="nll"), dict(loss="mmd+nll", align=True)]
)
def test_ewmp_model_is_collinear_safe(extra, pool, scaler):
    """The IRC-safe encoder with an energy-flow loss: splitting a constituent in two collinear parts leaves
    the score unchanged (all node features it reads are directions; pT enters only as a weight)."""
    torch.manual_seed(0)
    m = build_model("ef_ae", **EWMP, **extra).eval()
    idx = np.arange(12)
    with torch.no_grad():
        s0 = m.score(make_batch(pool, idx, scaler))
        s1 = m.score(make_batch(_split_one(pool), idx, scaler))
    assert torch.allclose(s0, s1, rtol=1e-4, atol=1e-5), (s0 - s1).abs().max()


def test_ewmp_model_soft_emission_is_a_small_change(pool, scaler):
    torch.manual_seed(0)
    m = build_model("ef_ae", **EWMP).eval()
    idx = np.arange(12)
    b0 = make_batch(pool, idx, scaler)
    with torch.no_grad():
        s0 = m.score(b0)
        d_small = (m.score(make_batch(_add_soft(pool, 1e-6), idx, scaler)) - s0).abs().max()
        d_large = (m.score(make_batch(_add_soft(pool, 1e-1), idx, scaler)) - s0).abs().max()
    assert d_small < 1e-4 and d_small < 0.1 * d_large


def test_edgeconv_model_is_not_collinear_safe(pool, scaler):
    """Control: the default (EdgeConv on log-pT features, k-NN graph) is *not* collinear safe."""
    torch.manual_seed(0)
    m = build_model("ef_ae", hidden=(16, 16), k=4, latent_dim=4, n_out=8, dec_hidden=(32,)).eval()
    idx = np.arange(12)
    with torch.no_grad():
        d = (
            (m.score(make_batch(pool, idx, scaler)) - m.score(make_batch(_split_one(pool), idx, scaler)))
            .abs()
            .max()
        )
    assert d > 1e-4


def test_principal_axis_frame_is_rotation_invariant():
    g = torch.Generator().manual_seed(0)
    xy = torch.randn(32, 20, 2, generator=g) * torch.tensor([0.4, 0.1])
    z = torch.rand(32, 20, generator=g)
    z = z / z.sum(1, keepdim=True)
    ref = principal_axis_frame(xy, z)
    for ang in (0.7, 2.1, -1.3):
        c, s = float(np.cos(ang)), float(np.sin(ang))
        rot = torch.tensor([[c, -s], [s, c]])
        assert torch.allclose(principal_axis_frame(xy @ rot.T, z), ref, atol=1e-4)
    # the major axis carries the larger energy-weighted variance, and the third moments are non-negative
    assert (ref[..., 0].pow(2) * z).sum(1).ge((ref[..., 1].pow(2) * z).sum(1)).all()
    assert ((z * ref[..., 0].pow(3)).sum(1) >= -1e-7).all() and (
        (z * ref[..., 1].pow(3)).sum(1) >= -1e-7
    ).all()


def test_energy_nll_properties():
    g = torch.Generator().manual_seed(1)
    pred = torch.randn(4, 6, 2, generator=g) * 0.2
    pw = torch.softmax(torch.randn(4, 6, generator=g), -1)
    sig = torch.full((4, 6), 0.05)
    tgt = torch.randn(4, 9, 2, generator=g) * 0.2
    tw = torch.rand(4, 9, generator=g)
    tw = tw / tw.sum(1, keepdim=True)
    base = energy_nll(pred, pw, sig, tgt, tw)
    # padding with zero-weight points changes nothing
    tgt_p = torch.cat([tgt, torch.randn(4, 3, 2, generator=g)], 1)
    tw_p = torch.cat([tw, torch.zeros(4, 3)], 1)
    assert torch.allclose(base, energy_nll(pred, pw, sig, tgt_p, tw_p), atol=1e-6)
    # collinear split of a target point
    tgt_s = torch.cat([tgt, tgt[:, :1]], 1)
    tw_s = torch.cat([tw, tw[:, :1] * 0.4], 1)
    tw_s[:, 0] = tw[:, 0] * 0.6
    assert torch.allclose(base, energy_nll(pred, pw, sig, tgt_s, tw_s), atol=1e-5)
    # a better-placed mixture has a lower cross-entropy, and the loss is unbounded (unlike the MMD)
    far = energy_nll(pred + 5.0, pw, sig, tgt, tw)
    assert (far > base + 100).all()
    assert (energy_mmd(pred + 5.0, pw, tgt, tw) < 3 * 2).all()


def test_ewmp_checkpointing_gives_the_same_gradients(pool, scaler):
    """Gradient checkpointing only trades compute for memory: loss and gradients are unchanged."""
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        m = build_model("ef_ae", **EWMP, ewmp_checkpoint=ckpt).train()
        torch.manual_seed(1)
        loss = m.losses(make_batch(pool, np.arange(12), scaler))["total"].mean()
        loss.backward()
        grads.append([p.grad.clone() for p in m.parameters() if p.grad is not None])
    assert len(grads[0]) == len(grads[1]) > 0
    for a, b in zip(*grads, strict=True):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)
