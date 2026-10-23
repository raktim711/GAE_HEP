"""Energy-flow graph autoencoder ("M2e"): reconstruct the jet's pT-weighted energy flow.

Same permutation-invariant EdgeConv encoder and jet-level bottleneck as :class:`~gaehep.models.set_ae.SetAE`,
but the decoder emits ``n_out`` points in (deta, dphi) **with pT fractions** (softmax over the points),
and the loss / anomaly score is the energy-weighted multi-scale MMD between the input energy flow and its
reconstruction (:func:`gaehep.losses.mmd.energy_mmd`).  Soft constituents carry little weight, so the
score reflects the hard structure (prongs, their separation, i.e. mass and N-subjettiness-like
information) instead of soft multiplicity.  Order-free in both the input and the output sets.

``cond_dim > 0`` conditions the decoder on the jet's standardised (log m, log pT) (mass-decorrelated
variant): the latent then only needs to describe the energy flow *given* the jet mass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gaehep.data.dataset import JetBatch
from gaehep.losses.mmd import DEFAULT_SIGMAS, energy_mmd
from gaehep.models.base import JetModel, mlp
from gaehep.models.edgeconv import EdgeConvEncoder


class EnergyFlowAE(JetModel):
    def __init__(
        self,
        in_dim: int = 7,
        hidden: Sequence[int] = (48, 48, 48),
        k: int = 12,
        latent_dim: int = 8,
        n_out: int = 32,
        dec_hidden: Sequence[int] = (256, 256),
        cond_dim: int = 0,
        aggr: str = "mean",
        dynamic: bool = True,
        sigmas: Sequence[float] = DEFAULT_SIGMAS,
    ):
        super().__init__()
        self.n_out, self.cond_dim, self.sigmas = n_out, cond_dim, tuple(sigmas)
        self.encoder = EdgeConvEncoder(in_dim, hidden, k=k, dynamic=dynamic, aggr=aggr)
        self.to_latent = mlp(2 * self.encoder.out_dim, [128], latent_dim)
        self.decoder = mlp(latent_dim + cond_dim, dec_hidden, n_out * 3)

    def embed(self, b: JetBatch) -> torch.Tensor:
        h = self.encoder(b.x, b.mask, b.coords)
        return self.to_latent(self.encoder.pool(h, b.mask))

    def forward(self, b: JetBatch) -> dict[str, torch.Tensor]:
        z = self.embed(b)
        zc = torch.cat([z, b.jet[:, : self.cond_dim]], dim=-1) if self.cond_dim else z
        out = self.decoder(zc).view(len(b), self.n_out, 3)
        return {"z": z, "xy": out[..., :2], "w": torch.softmax(out[..., 2], dim=-1)}

    @staticmethod
    def target_flow(b: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Input energy flow: (deta, dphi) and pT fractions over the (top-N) real constituents."""
        pt = b.raw[..., 0] * b.mask
        w = pt / pt.sum(1, keepdim=True).clamp(min=1e-12)
        return b.coords, w

    def score_and_embed(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        out = self(b)
        return self.losses(b, out)["score"], out["z"]

    def losses(self, b: JetBatch, out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        out = self(b) if out is None else out
        xy, w = self.target_flow(b)
        d = energy_mmd(out["xy"], out["w"], xy, w, self.sigmas)
        return {"total": d, "score": d}
