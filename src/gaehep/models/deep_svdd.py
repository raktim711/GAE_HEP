"""One-class Deep SVDD ("M4") with the permutation-invariant EdgeConv encoder.

The network phi maps background jets close to a fixed centre c; the anomaly score is |phi(x) - c|^2.
Following Ruff et al. (ICML 2018) all layers are bias-free and c is fixed from the initial network
(components with |c| < eps are pushed to +-eps) to avoid the trivial collapsed solution.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from gaehep.data.dataset import JetBatch
from gaehep.models.base import JetModel, mlp
from gaehep.models.edgeconv import EdgeConvEncoder, batch_tie_order


class DeepSVDD(JetModel):
    def __init__(
        self,
        in_dim: int = 7,
        hidden: Sequence[int] = (64, 64, 64),
        k: int = 16,
        latent_dim: int = 16,
        aggr: str = "mean",
        dynamic: bool = True,
        center_eps: float = 0.1,
    ):
        super().__init__()
        self.encoder = EdgeConvEncoder(in_dim, hidden, k=k, dynamic=dynamic, aggr=aggr, bias=False)
        self.head = mlp(2 * self.encoder.out_dim, [64], latent_dim, act=nn.ReLU, bias=False)
        self.center_eps = center_eps
        self.register_buffer("center", torch.zeros(latent_dim))
        self.register_buffer("center_ready", torch.tensor(False))

    def embed(self, b: JetBatch) -> torch.Tensor:
        h = self.encoder(b.x, b.mask, b.coords, batch_tie_order(b))
        return self.head(self.encoder.pool(h, b.mask))

    @torch.no_grad()
    def prepare(self, batches: Iterable[JetBatch]) -> None:
        if bool(self.center_ready):
            return
        was_training = self.training
        self.eval()
        zs = [self.embed(b) for b in batches]
        c = torch.cat(zs).mean(0)
        eps = self.center_eps
        c = torch.where((c.abs() < eps) & (c < 0), torch.full_like(c, -eps), c)
        c = torch.where((c.abs() < eps) & (c >= 0), torch.full_like(c, eps), c)
        self.center.copy_(c)
        self.center_ready.fill_(True)
        self.train(was_training)

    def score_and_embed(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(b)
        return ((z - self.center) ** 2).sum(-1), z

    def losses(self, b: JetBatch) -> dict[str, torch.Tensor]:
        if not bool(self.center_ready):
            raise RuntimeError("Deep SVDD centre not initialised; call prepare() first")
        d = ((self.embed(b) - self.center) ** 2).sum(-1)
        return {"total": d, "score": d}
