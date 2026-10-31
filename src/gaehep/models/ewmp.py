"""Energy-weighted message passing on radius graphs: an infrared- and collinear-safe jet encoder.

After the IRC-safe graph autoencoder of Atkinson et al. (arXiv:2204.12231; read for this project): nodes
carry only *directions* (deta, dphi, dR), neighbourhoods are fixed-radius balls in the (deta, dphi) plane
(a k-nearest-neighbour graph changes under a soft emission, a ball does not), and messages are averaged with
the neighbours' pT fractions as weights,

    h_i' = relu( W_s h_i + [ sum_j A_ij z_j relu(W_a h_i + W_b h_j + b) / sum_j A_ij z_j ]_radii ),

with a weighted-mean (and weighted mean-of-squares) readout.  The pT fractions z enter only as weights, never
as features, so (i) splitting a constituent into collinear parts leaves every output unchanged and (ii) a
soft constituent changes the output by O(z).  Several radii can be used side by side (multi-scale balls).
Order- and padding-invariant like the EdgeConv encoder.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from gaehep.graphs.ops import masked_radius_mask


class EWMPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, radii: Sequence[float]):
        super().__init__()
        self.radii = tuple(float(r) for r in radii)
        self.theta_i = nn.ModuleList(nn.Linear(in_dim, out_dim) for _ in self.radii)
        self.theta_j = nn.ModuleList(nn.Linear(in_dim, out_dim, bias=False) for _ in self.radii)
        self.out = nn.Linear(in_dim + out_dim * len(self.radii), out_dim)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        adj = masked_radius_mask(coords, mask, self.radii)  # (R, B, N, N) bool; one cdist for all radii
        parts = [h]
        for a, ti, tj in zip(adj, self.theta_i, self.theta_j, strict=True):
            wgt = a.to(h.dtype) * z[:, None, :]  # A_ij z_j  (self included: d_ii = 0 < r)
            msg = torch.relu(ti(h)[:, :, None, :] + tj(h)[:, None, :, :])  # (B, N, N, out)
            agg = (wgt[..., None] * msg).sum(2) / wgt.sum(-1, keepdim=True).clamp(min=1e-12)
            parts.append(agg)
        return torch.relu(self.out(torch.cat(parts, dim=-1))) * mask[..., None].to(h.dtype)


class EWMPEncoder(nn.Module):
    """Stack of :class:`EWMPBlock`; ``pool`` is the pT-weighted mean (and mean of squares) of the nodes."""

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int] = (48, 48, 48),
        radii: Sequence[float] = (0.3,),
        checkpoint: bool = False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        dims = [in_dim, *hidden]
        self.blocks = nn.ModuleList(EWMPBlock(dims[i], dims[i + 1], radii) for i in range(len(hidden)))
        self.out_dim = dims[-1]

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            if self.checkpoint and self.training and torch.is_grad_enabled():
                h = checkpoint(blk, h, mask, coords, z, use_reentrant=False)
            else:
                h = blk(h, mask, coords, z)
        return h

    @staticmethod
    def pool(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        zn = z / z.sum(1, keepdim=True).clamp(min=1e-12)
        m1 = (zn[..., None] * h).sum(1)
        m2 = (zn[..., None] * h.pow(2)).sum(1)
        return torch.cat([m1, m2], dim=-1)
