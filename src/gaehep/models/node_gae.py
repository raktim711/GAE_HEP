"""Node-level graph autoencoder ("M1"): the corrected successor of the AIMS GAE.

Differences from the AIMS model (see ``legacy_gae.py``):

* local k-NN graphs and EdgeConv (relative features h_j - h_i) instead of complete graphs with GCN/GIN,
  which give every node the same embedding;
* standardised, jet-relative node features instead of raw (pT, eta, phi);
* a real per-node bottleneck: each node's own features have ~3 degrees of freedom given the jet
  (deta, dphi, pT), so the node latent is 2-dimensional and reconstruction must use the neighbourhood;
* no ReLU / dropout on the latent; the decoder passes messages on the same (deta, dphi) graph.

Score = masked mean over nodes of the per-node mean squared error (permutation invariant).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from gaehep.data.dataset import JetBatch
from gaehep.models.base import JetModel
from gaehep.models.edgeconv import EdgeConvEncoder, batch_tie_order


class NodeGAE(JetModel):
    def __init__(
        self,
        in_dim: int = 7,
        enc_hidden: Sequence[int] = (64, 64),
        node_latent: int = 2,
        dec_hidden: Sequence[int] = (64, 64),
        k: int = 16,
        aggr: str = "mean",
        dynamic: bool = True,
    ):
        super().__init__()
        self.encoder = EdgeConvEncoder(in_dim, enc_hidden, k=k, dynamic=dynamic, aggr=aggr)
        self.to_latent = nn.Linear(self.encoder.out_dim, node_latent)
        self.decoder = EdgeConvEncoder(node_latent, dec_hidden, k=k, dynamic=False, aggr=aggr)
        self.out = nn.Linear(self.decoder.out_dim, in_dim)

    def forward(self, b: JetBatch) -> dict[str, torch.Tensor]:
        m = b.mask[..., None].to(b.x.dtype)
        tie = batch_tie_order(b)
        z = self.to_latent(self.encoder(b.x, b.mask, b.coords, tie)) * m
        xhat = self.out(self.decoder(z, b.mask, b.coords, tie)) * m
        return {"z_nodes": z, "xhat": xhat}

    def embed(self, b: JetBatch) -> torch.Tensor:
        z = self(b)["z_nodes"]
        return self.encoder.pool(z, b.mask)

    def score_and_embed(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        out = self(b)
        return self._per_jet(b, out["xhat"]), self.encoder.pool(out["z_nodes"], b.mask)

    @staticmethod
    def _per_jet(b: JetBatch, xhat: torch.Tensor) -> torch.Tensor:
        per_node = ((xhat - b.x) ** 2).mean(-1)
        m = b.mask.to(per_node.dtype)
        return (per_node * m).sum(1) / m.sum(1).clamp(min=1)

    def losses(self, b: JetBatch) -> dict[str, torch.Tensor]:
        per_jet = self._per_jet(b, self(b)["xhat"])
        return {"total": per_jet, "score": per_jet}
