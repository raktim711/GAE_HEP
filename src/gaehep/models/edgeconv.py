"""EdgeConv (DGCNN / ParticleNet-style) blocks on dense, masked jet graphs.

For node i with k nearest neighbours N(i):

    h_i' = act( AGG_{j in N(i)} MLP([h_i, h_j - h_i]) + W h_i )

AGG is a masked mean or max, so the block is permutation-equivariant; the encoder's masked mean+max
pooling is permutation-invariant.  The first block builds its k-NN graph in (deta, dphi); later blocks can
rebuild it in feature space ("dynamic" graph, as in DGCNN).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from gaehep.graphs.ops import gather_neighbors, masked_knn, masked_max, masked_mean


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
        self.k, self.aggr = k, aggr
        layers: list[nn.Module] = []
        d = 2 * in_dim
        for _ in range(n_mlp):
            layers += [nn.Linear(d, out_dim, bias=bias), act()]
            d = out_dim
        self.mlp = nn.Sequential(*layers)
        self.shortcut = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = act()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, knn_coords: torch.Tensor) -> torch.Tensor:
        idx, valid = masked_knn(knn_coords, mask, self.k)
        if idx.shape[-1] == 0:
            agg = torch.zeros(*x.shape[:2], self.shortcut.out_features, dtype=x.dtype, device=x.device)
        else:
            xi = x.unsqueeze(2).expand(-1, -1, idx.shape[-1], -1)
            xj = gather_neighbors(x, idx)
            h = self.mlp(torch.cat([xi, xj - xi], dim=-1))  # (B, N, k, out)
            v = valid[..., None]
            if self.aggr == "mean":
                agg = (h * v).sum(2) / v.sum(2).clamp(min=1).to(h.dtype)
            else:
                agg = h.masked_fill(~v, float("-inf")).amax(2)
                agg = torch.where(valid.any(2, keepdim=True), agg, torch.zeros_like(agg))
        out = self.act(agg + self.shortcut(x))
        return out * mask[..., None].to(out.dtype)


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
    ):
        super().__init__()
        self.dynamic = dynamic
        dims = [in_dim, *hidden]
        self.blocks = nn.ModuleList(
            EdgeConvBlock(dims[i], dims[i + 1], k=k, aggr=aggr, bias=bias) for i in range(len(hidden))
        )
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        h = x
        for i, blk in enumerate(self.blocks):
            knn_c = coords if (i == 0 or not self.dynamic) else h
            h = blk(h, mask, knn_c)
        return h

    @staticmethod
    def pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([masked_mean(h, mask), masked_max(h, mask)], dim=-1)
