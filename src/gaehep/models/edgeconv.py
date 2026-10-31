"""EdgeConv (DGCNN / ParticleNet-style) blocks on dense, masked jet graphs.

For node i with k nearest neighbours N(i):

    h_i' = act( AGG_{j in N(i)} MLP([h_i, h_j - h_i]) + W h_i )

AGG is a masked mean or max, so the block is permutation-equivariant; the encoder's masked mean+max
pooling is permutation-invariant.  The first block builds its k-NN graph in (deta, dphi); later blocks can
rebuild it in feature space ("dynamic" graph, as in DGCNN).

Distance ties (e.g. constituents at one (eta, phi), or equal hidden features) are broken by a node priority,
``tie_order`` = :func:`batch_tie_order` (larger pT first, then (deta, dphi)), the same for every block of the
encoder, static or dynamic.  It is a property of the nodes themselves, so every k-NN graph, and the slot order
of its neighbours, is exactly permutation-equivariant.  Without ties the graphs do not depend on it.

CPU efficiency: the first (widest) layer of the edge MLP is linear in [h_i, h_j - h_i], so it is computed
per *node* and only gathered per edge:  W [h_i, h_j - h_i] + b = (W_a - W_b) h_i + W_b h_j + b.  This is an
exact reparametrisation (``edge_mlp_equivalent`` rebuilds the plain edge MLP for cross-checks) and cuts the
per-edge cost by ~3x for the default sizes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from gaehep.data.dataset import JetBatch
from gaehep.graphs.ops import gather_neighbors, knn_tie_order, masked_knn, masked_max, masked_mean


def batch_tie_order(b: JetBatch) -> torch.Tensor:
    """k-NN tie-break priority of a batch's nodes: larger pT first, then (deta, dphi) (:func:`knn_tie_order`)."""
    return knn_tie_order(b.raw[..., 0], b.coords)


class EdgeConvBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int = 16,
        n_mlp: int = 2,
        aggr: str = "mean",
        bias: bool = True,
        act: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if aggr not in ("mean", "max"):
            raise ValueError(f"aggr must be 'mean' or 'max', got {aggr}")
        if n_mlp < 1:
            raise ValueError("n_mlp must be >= 1")
        self.k, self.aggr = k, aggr
        self.theta_i = nn.Linear(in_dim, out_dim, bias=bias)  # (W_a - W_b), carries the bias
        self.theta_j = nn.Linear(in_dim, out_dim, bias=False)  # W_b
        rest: list[nn.Module] = [act()]
        for _ in range(n_mlp - 1):
            rest += [nn.Linear(out_dim, out_dim, bias=bias), act()]
        self.edge_rest = nn.Sequential(*rest)
        self.shortcut = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = act()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        knn_coords: torch.Tensor,
        tie_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx, valid = masked_knn(knn_coords, mask, self.k, tie_order=tie_order)
        if idx.shape[-1] == 0:
            agg = torch.zeros(*x.shape[:2], self.shortcut.out_features, dtype=x.dtype, device=x.device)
        else:
            u = self.theta_i(x)  # (B, N, out)
            v = self.theta_j(x)
            h = self.edge_rest(u.unsqueeze(2) + gather_neighbors(v, idx))  # (B, N, k, out)
            vm = valid[..., None]
            if self.aggr == "mean":
                agg = (h * vm).sum(2) / vm.sum(2).clamp(min=1).to(h.dtype)
            else:
                agg = h.masked_fill(~vm, float("-inf")).amax(2)
                agg = torch.where(valid.any(2, keepdim=True), agg, torch.zeros_like(agg))
        out = self.act(agg + self.shortcut(x))
        return out * mask[..., None].to(out.dtype)

    def edge_mlp_equivalent(self) -> nn.Sequential:
        """The plain edge MLP acting on [h_i, h_j - h_i] that this block computes (for cross-checks)."""
        lin = nn.Linear(
            2 * self.theta_i.in_features, self.theta_i.out_features, bias=self.theta_i.bias is not None
        )
        with torch.no_grad():
            w_b = self.theta_j.weight
            lin.weight.copy_(torch.cat([self.theta_i.weight + w_b, w_b], dim=1))
            if self.theta_i.bias is not None:
                lin.bias.copy_(self.theta_i.bias)
        return nn.Sequential(lin, *self.edge_rest)


class EdgeConvEncoder(nn.Module):
    """Stack of EdgeConv blocks returning node embeddings; ``pool`` gives a permutation-invariant vector."""

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int] = (64, 64, 64),
        k: int = 16,
        dynamic: bool = True,
        aggr: str = "mean",
        bias: bool = True,
        n_mlp: int = 2,
    ):
        super().__init__()
        self.dynamic = dynamic
        dims = [in_dim, *hidden]
        self.blocks = nn.ModuleList(
            EdgeConvBlock(dims[i], dims[i + 1], k=k, n_mlp=n_mlp, aggr=aggr, bias=bias)
            for i in range(len(hidden))
        )
        self.out_dim = dims[-1]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        tie_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``tie_order`` (B, N): k-NN tie-break priority (:func:`batch_tie_order`), used by every block."""
        h = x
        for i, blk in enumerate(self.blocks):
            knn_c = coords if (i == 0 or not self.dynamic) else h
            h = blk(h, mask, knn_c, tie_order)
        return h

    @staticmethod
    def pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([masked_mean(h, mask), masked_max(h, mask)], dim=-1)
