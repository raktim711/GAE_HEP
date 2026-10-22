"""Graph-level set autoencoder (main model, "M2").

EdgeConv encoder -> masked mean+max pooling -> latent z (the bottleneck is per *jet*, not per node) ->
MLP set decoder that emits ``n_out`` points in (deta, dphi, log pT) (standardised) -> masked Chamfer
distance to the jet's constituents.  The first ``n`` emitted points are compared with the ``n`` real
constituents (n = min(multiplicity, n_out)); a small head also predicts the jet multiplicity from z.

Permutation invariance: z is invariant (equivariant blocks + symmetric pooling) and the Chamfer distance is
invariant to the order of both point sets, so the anomaly score does not depend on constituent order.

``cond_dim > 0`` gives the *conditional* variant: the decoder (and multiplicity head) also see the jet's
standardised (log m, log pT), so the latent does not need to - and the score is less able to - encode mass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gaehep.data.dataset import RECON_IDX, JetBatch
from gaehep.losses.chamfer import masked_chamfer
from gaehep.losses.sinkhorn import masked_sinkhorn
from gaehep.models.base import JetModel, mlp
from gaehep.models.edgeconv import EdgeConvEncoder


class SetAE(JetModel):
    def __init__(
        self,
        in_dim: int = 7,
        hidden: Sequence[int] = (64, 64, 64),
        k: int = 16,
        latent_dim: int = 8,
        n_out: int = 50,
        dec_hidden: Sequence[int] = (256, 256),
        cond_dim: int = 0,
        aggr: str = "mean",
        dynamic: bool = True,
        lambda_mult: float = 0.1,
        loss: str = "chamfer",
        sinkhorn_eps: float = 0.05,
    ):
        super().__init__()
        self.recon_idx = list(RECON_IDX)
        self.n_out, self.cond_dim, self.lambda_mult = n_out, cond_dim, lambda_mult
        self.loss_name, self.sinkhorn_eps = loss, sinkhorn_eps
        self.encoder = EdgeConvEncoder(in_dim, hidden, k=k, dynamic=dynamic, aggr=aggr)
        self.to_latent = mlp(2 * self.encoder.out_dim, [128], latent_dim)
        self.mult_head = mlp(latent_dim + cond_dim, [32], 1)
        self.decoder = mlp(latent_dim + cond_dim, dec_hidden, n_out * len(self.recon_idx))

    def embed(self, b: JetBatch) -> torch.Tensor:
        h = self.encoder(b.x, b.mask, b.coords)
        return self.to_latent(self.encoder.pool(h, b.mask))

    def _cond(self, b: JetBatch, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, b.jet[:, : self.cond_dim]], dim=-1) if self.cond_dim else z

    def forward(self, b: JetBatch) -> dict[str, torch.Tensor]:
        z = self.embed(b)
        zc = self._cond(b, z)
        recon = self.decoder(zc).view(len(b), self.n_out, len(self.recon_idx))
        return {"z": z, "recon": recon, "mult": self.mult_head(zc).squeeze(-1)}

    def losses(self, b: JetBatch) -> dict[str, torch.Tensor]:
        out = self(b)
        n_nodes = b.x.shape[1]
        if n_nodes > self.n_out:
            raise ValueError(f"batch has {n_nodes} nodes but the decoder emits n_out={self.n_out}")
        target = b.x[..., self.recon_idx]
        tmask = b.mask
        n = b.n.clamp(max=self.n_out)
        pmask = torch.arange(self.n_out, device=n.device)[None, :] < n[:, None]
        if self.loss_name == "chamfer":
            rec = masked_chamfer(out["recon"], pmask, target, tmask)
        elif self.loss_name == "sinkhorn":
            rec = masked_sinkhorn(out["recon"], pmask, target, tmask, eps=self.sinkhorn_eps)
        else:
            raise ValueError(self.loss_name)
        mult = (out["mult"] - b.jet[:, 2]) ** 2
        total = rec + self.lambda_mult * mult
        return {"total": total, "score": total, "recon": rec, "mult": mult}
