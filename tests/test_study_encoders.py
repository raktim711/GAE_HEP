"""The encoders of the study (docs/STUDY_PROTOCOL.md, section 3): IRC safety of efn / ew_attention, the
collapse of gcn_complete, masking in the attention encoders and the PMA pooling, the configs/study files and
a 1-epoch smoke run of each of them.  Permutation and padding invariance are in test_invariance*.py (MODELS)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from gaehep.config import load_yaml
from gaehep.data.dataset import JetBatch, make_batch
from gaehep.data.fake import make_fake
from gaehep.models import build_model
from gaehep.models.edgeconv import EdgeConvEncoder, batch_tie_order
from gaehep.models.ef_ae import ENCODERS, IRC_SAFE
from gaehep.models.set_encoders import AttentionEncoder, CompleteGraphEncoder, PMAPool, key_bias
from gaehep.train.run import run
from tests.test_ef_variants import _add_soft, _split_one

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs" / "study"
SMALL = dict(hidden=(16, 16), k=4, latent_dim=4, n_out=8, dec_hidden=(32,), variational=True)
UNSAFE = tuple(e for e in ENCODERS if e not in IRC_SAFE)


def _small(encoder: str, **kw):
    torch.manual_seed(0)
    extra = dict(radii=(0.1, 0.4)) if encoder == "ewmp" else {}
    return build_model("ef_ae", **SMALL, encoder=encoder, **extra, **kw).eval()


def _f64(b: JetBatch) -> JetBatch:
    return JetBatch(**{k: (v.double() if v.is_floating_point() else v) for k, v in vars(b).items()})


def _score_embed(m, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return m.score_and_embed(b)


# -------------------------------------------------------------------------------------------- IRC safety
@pytest.mark.parametrize("encoder", IRC_SAFE)
def test_irc_safe_encoders_are_collinear_safe(encoder, pool, scaler):
    """Splitting the hardest constituent into two collinear parts leaves latent and score unchanged:
    float32 within round-off (1e-5, as for ewmp), and in float64 to 1e-10 for a 50/50 split (exact in binary,
    so the remaining difference is only the summation order); an IRC-unsafe model changes by > 1e-3."""
    m = _small(encoder)
    idx = np.arange(12)
    s0, z0 = _score_embed(m, make_batch(pool, idx, scaler))
    s1, z1 = _score_embed(m, make_batch(_split_one(pool, 0.3), idx, scaler))
    assert torch.allclose(s0, s1, rtol=1e-4, atol=1e-5), (s0 - s1).abs().max()
    assert torch.allclose(z0, z1, rtol=1e-4, atol=1e-5), (z0 - z1).abs().max()
    m.double()
    s0, z0 = _score_embed(m, _f64(make_batch(pool, idx, scaler)))
    s1, z1 = _score_embed(m, _f64(make_batch(_split_one(pool, 0.5), idx, scaler)))
    assert (s0 - s1).abs().max() < 1e-10 and (z0 - z1).abs().max() < 1e-10


@pytest.mark.parametrize("encoder", IRC_SAFE)
def test_irc_safe_encoders_soft_emission_is_order_z(encoder, pool, scaler):
    """A soft constituent of pT fraction z changes the latent and the score by O(z): the change falls by
    ~10x per decade of z (float64, z = 1e-2 ... 1e-5)."""
    m = _small(encoder).double()
    idx = np.arange(12)
    s0, z0 = _score_embed(m, _f64(make_batch(pool, idx, scaler)))
    ds, dz = [], []
    for z in (1e-2, 1e-3, 1e-4, 1e-5):
        s, e = _score_embed(m, _f64(make_batch(_add_soft(pool, z), idx, scaler)))
        ds.append(float((s - s0).abs().max()))
        dz.append(float((e - z0).abs().max()))
    for d in (ds, dz):
        ratios = [a / b for a, b in itertools.pairwise(d)]
        assert all(7.0 < r < 14.0 for r in ratios), (d, ratios)
        assert d[-1] < 1e-3, d


@pytest.mark.parametrize("encoder", UNSAFE)
def test_unsafe_encoders_are_not_collinear_safe(encoder, pool, scaler):
    """Control: the encoders on the 7 standard features (log pT, log E, ...) see a collinear split."""
    m = _small(encoder)
    idx = np.arange(12)
    s0, _ = _score_embed(m, make_batch(pool, idx, scaler))
    s1, _ = _score_embed(m, make_batch(_split_one(pool, 0.3), idx, scaler))
    assert (s0 - s1).abs().max() > 1e-4


def test_ew_attention_weights_are_z_times_exp_logits():
    """The energy-weighted attention weights are z_j exp(q_i.k_j / sqrt(d)) / sum_l z_l exp(q_i.k_l / sqrt(d))
    over the real nodes (padded and z = 0 keys get weight 0)."""
    g = torch.Generator().manual_seed(0)
    b, n, dim, heads = 3, 7, 8, 2
    blk = AttentionEncoder(3, (dim,), heads=heads, energy_weighted=True).blocks[0].attn
    q, k, v = (torch.randn(b, n, dim, generator=g, dtype=torch.float64) for _ in range(3))
    mask = torch.rand(b, n, generator=g) > 0.3
    mask[:, 0] = True
    z = torch.rand(b, n, generator=g, dtype=torch.float64) * mask
    z = z / z.sum(1, keepdim=True)
    got = blk.attend(q, k, v, mask, z)
    dh = dim // heads
    for gi in range(b):
        for h in range(heads):
            sl = slice(h * dh, (h + 1) * dh)
            w = z[gi][None, :] * torch.exp(q[gi, :, sl] @ k[gi, :, sl].T / dh**0.5)
            w = w / w.sum(1, keepdim=True)
            assert torch.allclose(got[gi, :, sl], w @ v[gi, :, sl], atol=1e-12)


def test_key_bias_handles_padding_and_empty_graphs():
    mask = torch.tensor([[True, True, False], [False, False, False]])
    z = torch.tensor([[0.25, 0.75, 0.0], [0.0, 0.0, 0.0]])
    bias = key_bias(mask, z)[:, 0, 0]
    assert torch.allclose(bias[0, :2], torch.log(z[0, :2])) and bias[0, 2] == float("-inf")
    assert torch.equal(bias[1], torch.zeros(3))  # no usable key: finite, the output is zeroed instead
    assert torch.equal(key_bias(mask)[:, 0, 0], torch.tensor([[0.0, 0.0, float("-inf")], [0.0, 0.0, 0.0]]))


# ---------------------------------------------------------------------------- collapse, masking, pooling
def test_gcn_complete_gives_identical_node_embeddings(batch):
    """gcn_complete: after its first layer every real node of a jet has exactly the same embedding (the
    complete-graph collapse); a k-NN EdgeConv layer does not."""
    torch.manual_seed(0)
    one = CompleteGraphEncoder(7, (16,))
    full = build_model("ef_ae", **SMALL, encoder="gcn_complete").encoder
    edge = EdgeConvEncoder(7, (16,), k=4)
    with torch.no_grad():
        outs = [one(batch.x, batch.mask), full(batch.x, batch.mask)]
        h_edge = edge(batch.x, batch.mask, batch.coords, batch_tie_order(batch))
    for h in outs:
        for i in range(len(batch)):
            real = h[i][batch.mask[i]]
            assert len(real) >= 3 and torch.equal(real, real[:1].expand_as(real))
            assert torch.equal(h[i][~batch.mask[i]], torch.zeros_like(h[i][~batch.mask[i]]))
    assert not torch.equal(h_edge[0][batch.mask[0]][0], h_edge[0][batch.mask[0]][1])


@pytest.mark.parametrize("encoder", ["transformer", "ew_attention"])
def test_attention_ignores_padded_node_content(encoder, batch):
    """Padded nodes are masked as keys: garbage in their features does not reach any real node."""
    torch.manual_seed(0)
    enc = AttentionEncoder(7, (16, 16), energy_weighted=encoder == "ew_attention").eval()
    z = batch.raw[..., 0] * batch.mask
    z = z / z.sum(1, keepdim=True)
    x_bad = batch.x.clone()
    x_bad[~batch.mask] = torch.randn(int((~batch.mask).sum()), 7) * 1e3
    with torch.no_grad():
        h0, h1 = enc(batch.x, batch.mask, z), enc(x_bad, batch.mask, z)
    assert torch.equal(h0, h1)
    assert torch.equal(h0[~batch.mask], torch.zeros_like(h0[~batch.mask]))


@pytest.mark.parametrize(
    ("encoder", "pool_kind"),
    [("transformer", None), ("ew_attention", None), ("deepsets", "pma"), ("edgeconv", "pma")],
)
def test_attention_finite_on_empty_and_single_node_jets(encoder, pool_kind, pool, scaler):
    """All-padding rows (softmax over -inf only) and one-node jets give finite scores and gradients."""
    m = _small(encoder, pool=pool_kind)
    const = pool.const[:4].copy()
    const[0] = 0.0
    const[1, 1:] = 0.0
    sub = pool.subset(np.arange(4))
    sub = type(sub)(**{**{k: getattr(sub, k) for k in sub.__dataclass_fields__}, "const": const})
    b = make_batch(sub, np.arange(4), scaler)
    m.train()
    loss = m.losses(b)["total"]
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_pma_pool_masks_padding_and_is_permutation_invariant():
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(0)
    pma = PMAPool(16).eval()
    h = torch.randn(4, 9, 16, generator=g)
    mask = torch.rand(4, 9, generator=g) > 0.3
    mask[:, 0] = True
    mask[3] = False
    h_bad = h.clone()
    h_bad[~mask] = 1e4
    perm = torch.randperm(9, generator=g)
    with torch.no_grad():
        p0 = pma(h, mask)
        assert torch.equal(p0, pma(h_bad, mask))
        assert torch.allclose(p0, pma(h[:, perm], mask[:, perm]), atol=1e-6)
        assert torch.isfinite(p0).all() and p0.shape == (4, 16)


def test_pool_option_validation_and_default_state_dict_keys():
    for enc in ("transformer", "efn", "ewmp", "ew_attention", "gcn_complete"):
        with pytest.raises(ValueError, match="pool"):
            build_model("ef_ae", **SMALL, encoder=enc, pool="pma")
    with pytest.raises(ValueError, match="pool"):
        build_model("ef_ae", **SMALL, encoder="deepsets", pool="sum")
    with pytest.raises(ValueError, match="encoder"):
        build_model("ef_ae", **SMALL, encoder="gat")
    assert build_model("ef_ae", **SMALL, encoder="efn", pool="weighted").pool_kind == "weighted"
    for enc in ENCODERS:  # only the pma ablation adds parameters outside encoder / to_latent / decoder
        top = {k.split(".")[0] for k in build_model("ef_ae", **SMALL, encoder=enc).state_dict()}
        assert top == {"encoder", "to_latent", "decoder"}
        assert build_model("ef_ae", **SMALL, encoder=enc).irc == (enc in IRC_SAFE)
    pma = build_model("ef_ae", **SMALL, encoder="deepsets", pool="pma").state_dict()
    assert {k.split(".")[0] for k in pma} == {"encoder", "to_latent", "decoder", "pool_attn"}


# ----------------------------------------------------------------------------------------------- configs
def test_study_configs_differ_from_the_recipe_only_in_the_encoder():
    base = load_yaml(ROOT / "configs" / "ef_vae_k8_e2.yaml")
    assert sorted(p.stem for p in STUDY.glob("*.yaml")) == sorted(ENCODERS)
    for enc in ENCODERS:
        cfg = load_yaml(STUDY / f"{enc}.yaml")
        want = {**base, "model_args": {**base["model_args"], "encoder": enc}}
        if enc == "ewmp":
            want["model_args"]["radii"] = [0.3]
        assert cfg == want, enc


@pytest.fixture(scope="module")
def fake_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    make_fake(root / "data", n_bkg=300, n_sig=60, n_max=20)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GAEHEP_DATA", str(root / "data"))
        mp.setenv("GAEHEP_RESULTS", str(root / "results"))
        yield root


@pytest.mark.parametrize("encoder", ENCODERS)
def test_study_config_smoke_run(encoder, fake_env):
    """1 epoch of each configs/study file (model as configured) on tiny synthetic data, then scoring."""
    cfg = load_yaml(STUDY / f"{encoder}.yaml")
    cfg["data"].update(n_const=20, latent_train_events=50)
    cfg["train"].update(
        epochs=1, jets_per_epoch=128, batch_size=64, val_jets=64, prepare_batches=1, threads=1, device="cpu"
    )
    summary = run(cfg, f"smoke_{encoder}", stages=("train", "score"), log=lambda *a: None)
    assert summary["status"] == "done"
    out = fake_env / "results" / "runs" / f"smoke_{encoder}"
    assert (out / "model.pt").is_file()
    with np.load(out / "scores.npz") as f:
        assert {"bkg_val", "bkg_test", "sig_test", "sig3_test"} <= set(f.files)
        assert all(np.isfinite(f[k]).all() and f[k].shape[1] == 2 for k in f.files)
