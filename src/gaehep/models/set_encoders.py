"""Set encoders of the encoder study (docs/STUDY_PROTOCOL.md, section 3) and the attention pooling.

All of them map node features ``x`` (B, N, F) with the real-node ``mask`` (B, N) to node embeddings
(B, N, width), zero on padded nodes, and are permutation-equivariant; the energy-flow autoencoder
(:class:`~gaehep.models.ef_ae.EnergyFlowAE`) pools them with EdgeConv's masked mean + max or EWMP's
pT-fraction weighted moments.

* :class:`DeepSetsEncoder`: a per-particle MLP, no interaction (``deepsets``; ``efn`` on direction-only
  inputs with pT-weighted pooling, i.e. an energy flow network).
* :class:`AttentionEncoder`: pre-LayerNorm self-attention blocks without position encoding
  (``transformer``).  With ``energy_weighted=True`` (``ew_attention``) log z_j is added to the logits, so the
  weights are z_j exp(q_i.k_j / sqrt(d)) / sum_l z_l exp(q_i.k_l / sqrt(d)): on direction-only inputs the
  block is infrared and collinear safe (a collinear split of j into z_a + z_b = z_j gives two identical keys
  whose weights add up to the weight of j; a soft node of fraction z gets a weight O(z)).
* :class:`CompleteGraphEncoder`: the mean over all real nodes of the jet, itself included, then a linear
  layer + ReLU (a GCN on the complete graph, ``gcn_complete``): every node of a jet has the same embedding
  after the first layer (the collapse control).
* :class:`PMAPool`: one learned seed attending over the nodes (Set Transformer pooling by multi-head
  attention, k = 1), the ``pma`` pooling ablation.

Padded nodes never act as keys (their attention logits are -inf); a jet without any real node attends to
nothing (its weights are set to zero instead of a 0/0 softmax), so every output stays finite.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from gaehep.graphs.ops import masked_mean


def _width(hidden: Sequence[int]) -> int:
    if len(hidden) == 0 or len(set(hidden)) != 1:
        raise ValueError(
            f"hidden={tuple(hidden)}: residual blocks need one constant width, e.g. (48, 48, 48)"
        )
    return int(hidden[0])


def key_bias(mask: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    """Additive attention bias over the keys, (B, 1, 1, N): 0 (or log z_j) on real nodes, -inf on padding.

    Graphs without any usable key get a bias of 0 everywhere instead (a softmax over -inf only is NaN); the
    caller zeroes their attention output with :func:`has_keys`.
    """
    use = mask if z is None else mask & (z > 0)
    if z is None:
        bias = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    else:
        bias = torch.log(torch.where(use, z, torch.ones_like(z)))
    bias = bias.masked_fill(~use, float("-inf"))
    bias = torch.where(use.any(1, keepdim=True), bias, torch.zeros_like(bias))
    return bias[:, None, None, :]


def has_keys(mask: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    """(B, 1, 1) float, 1 for graphs with at least one usable key (see :func:`key_bias`)."""
    use = mask if z is None else mask & (z > 0)
    return use.any(1).to(torch.float32)[:, None, None]


class DeepSetsEncoder(nn.Module):
    """Per-particle MLP phi (Linear + ReLU per layer), no interaction between the nodes."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (48, 48, 48)):
        super().__init__()
        dims = [in_dim, *hidden]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden)))
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in self.layers:
            h = torch.relu(lin(h))
        return h * mask[..., None].to(h.dtype)


class CompleteGraphEncoder(nn.Module):
    """h_i' = ReLU(W mean_{j real} h_j + b) for every real node i: a GCN on the complete graph with self loops."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (48, 48, 48)):
        super().__init__()
        dims = [in_dim, *hidden]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden)))
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, m = x, mask[..., None].to(x.dtype)
        for lin in self.layers:
            h = torch.relu(lin(masked_mean(h, mask)))[:, None, :] * m  # the same vector on every real node
        return h


class _MultiHead(nn.Module):
    """Scaled dot-product attention of queries over masked, optionally energy-weighted keys."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"width {dim} is not divisible by heads={heads}")
        self.heads, self.d_head = heads, dim // heads

    def split(self, t: torch.Tensor) -> torch.Tensor:  # (B, L, dim) -> (B, heads, L, d_head)
        return t.view(t.shape[0], t.shape[1], self.heads, self.d_head).transpose(1, 2)

    def attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, z: torch.Tensor | None
    ) -> torch.Tensor:
        q, k, v = self.split(q), self.split(k), self.split(v)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.d_head) + key_bias(mask, z).to(q.dtype)
        a = torch.softmax(logits, dim=-1) * has_keys(mask, z).to(q.dtype)[..., None]
        out = a @ v  # (B, heads, L, d_head)
        return out.transpose(1, 2).reshape(q.shape[0], -1, self.heads * self.d_head)


class AttentionBlock(nn.Module):
    """Pre-LayerNorm transformer block: h + Attn(LN(h)), then h + MLP(LN(h)) (GELU, hidden mlp_ratio x width)."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn = _MultiHead(dim, heads)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, h: torch.Tensor, mask: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        h = h + self.proj(self.attn.attend(q, k, v, mask, z))
        return h + self.mlp(self.norm2(h))


class AttentionEncoder(nn.Module):
    """Linear embedding, ``len(hidden)`` pre-LN self-attention blocks of constant width, final LayerNorm.

    No position encoding (permutation-equivariant); padded nodes are masked as keys and zeroed in the output.
    ``energy_weighted=True`` adds log z_j to the logits (``forward`` then needs the pT fractions ``z``).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int] = (48, 48, 48),
        heads: int = 4,
        mlp_ratio: int = 2,
        energy_weighted: bool = False,
    ):
        super().__init__()
        dim = _width(hidden)
        self.energy_weighted = energy_weighted
        self.embed = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(AttentionBlock(dim, heads, mlp_ratio) for _ in hidden)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if self.energy_weighted and z is None:
            raise ValueError("energy-weighted attention needs the pT fractions z")
        zz = z if self.energy_weighted else None
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, mask, zz)
        return self.norm(h) * mask[..., None].to(h.dtype)


class PMAPool(nn.Module):
    """Pooling by multi-head attention with one learned seed S (Set Transformer, Lee et al. 2019, k = 1).

    O = q(S) + Attn(q(S), k(h), v(h)) over the real nodes; output O + ReLU(W_o O), of size ``dim``.  A jet
    without real nodes gives q(S) + ReLU(W_o q(S)) (no attention), the same for every such jet.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.seed)
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.attn = _MultiHead(dim, heads)
        self.out_dim = dim

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.q(self.seed).expand(h.shape[0], 1, -1)
        o = q + self.attn.attend(q, self.k(h), self.v(h), mask, None)
        return (o + F.relu(self.o(o)))[:, 0]
