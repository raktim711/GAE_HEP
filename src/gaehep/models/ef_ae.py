"""Energy-flow graph autoencoder ("M2e"): reconstruct the jet's pT-weighted energy flow.

Same permutation-invariant EdgeConv encoder and jet-level bottleneck as :class:`~gaehep.models.set_ae.SetAE`,
but the decoder emits ``n_out`` points in (deta, dphi) **with pT fractions** (softmax over the points),
and the loss / anomaly score is the energy-weighted multi-scale MMD between the input energy flow and its
reconstruction (:func:`gaehep.losses.mmd.energy_mmd`).  Soft constituents carry little weight, so the
score reflects the hard structure (prongs, their separation, i.e. mass and N-subjettiness-like
information) instead of soft multiplicity.  Order-free in both the input and the output sets.

``cond_dim > 0`` conditions the decoder on the jet's standardised (log m, log pT) (mass-decorrelated
variant): the latent then only needs to describe the energy flow *given* the jet mass.

``variational=True`` turns the bottleneck into a Gaussian VAE latent (loss = MMD + beta * KL); the KL term
of a jet is a density-like anomaly score that does not reward "simple" jets the way reconstruction does.
``score`` selects the anomaly score: ``"recon"`` (MMD at z = mu), ``"kl"`` or ``"total"``.  All of them
are returned by :meth:`losses` (``rec``, ``kl``) so several scores can be evaluated in one pass.
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
        variational: bool = False,
        beta: float = 1e-3,
        score: str = "recon",
    ):
        super().__init__()
        if score not in ("recon", "kl", "total") or (score != "recon" and not variational):
            raise ValueError(f"score={score!r} needs variational=True (or use 'recon')")
        self.n_out, self.cond_dim, self.sigmas = n_out, cond_dim, tuple(sigmas)
        self.variational, self.beta, self.score_kind = variational, beta, score
        self.encoder = EdgeConvEncoder(in_dim, hidden, k=k, dynamic=dynamic, aggr=aggr)
        self.to_latent = mlp(2 * self.encoder.out_dim, [128], latent_dim * (2 if variational else 1))
        self.decoder = mlp(latent_dim + cond_dim, dec_hidden, n_out * 3)

    def _encode(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.to_latent(self.encoder.pool(self.encoder(b.x, b.mask, b.coords), b.mask))
        if not self.variational:
            return h, None
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar.clamp(-12.0, 8.0)

    def embed(self, b: JetBatch) -> torch.Tensor:
        return self._encode(b)[0]

    def forward(self, b: JetBatch) -> dict[str, torch.Tensor]:
        mu, logvar = self._encode(b)
        z = mu
        if self.variational and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        zc = torch.cat([z, b.jet[:, : self.cond_dim]], dim=-1) if self.cond_dim else z
        out = self.decoder(zc).view(len(b), self.n_out, 3)
        res = {"z": mu, "xy": out[..., :2], "w": torch.softmax(out[..., 2], dim=-1)}
        if logvar is not None:
            res["logvar"] = logvar
        return res

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
        rec = energy_mmd(out["xy"], out["w"], xy, w, self.sigmas)
        res = {"rec": rec, "total": rec}
        if self.variational:
            mu, logvar = out["z"], out["logvar"]
            res["kl"] = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1)
            res["total"] = rec + self.beta * res["kl"]
        res["score"] = {"recon": rec, "kl": res.get("kl"), "total": res["total"]}[self.score_kind]
        return res
