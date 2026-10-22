"""Supervised EdgeConv jet classifier: the idealised upper bound (uses signal labels)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from gaehep.data.dataset import JetBatch
from gaehep.models.base import JetModel, mlp
from gaehep.models.edgeconv import EdgeConvEncoder


class EdgeConvClassifier(JetModel):
    supervised = True

    def __init__(
        self,
        in_dim: int = 7,
        hidden: Sequence[int] = (64, 64, 64),
        k: int = 16,
        head_hidden: Sequence[int] = (64,),
        aggr: str = "mean",
        dynamic: bool = True,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = EdgeConvEncoder(in_dim, hidden, k=k, dynamic=dynamic, aggr=aggr)
        self.head = mlp(2 * self.encoder.out_dim, head_hidden, 1)
        self.pos_weight = pos_weight

    def embed(self, b: JetBatch) -> torch.Tensor:
        h = self.encoder(b.x, b.mask, b.coords)
        return self.encoder.pool(h, b.mask)

    def losses(self, b: JetBatch) -> dict[str, torch.Tensor]:
        logit = self.head(self.embed(b)).squeeze(-1)
        y = b.label.to(logit.dtype)
        pw = torch.tensor(self.pos_weight, dtype=logit.dtype, device=logit.device)
        bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw, reduction="none")
        return {"total": bce, "score": logit}
