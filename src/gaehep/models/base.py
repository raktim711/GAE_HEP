"""Common interface for all jet models and a small MLP helper."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from gaehep.data.dataset import JetBatch


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: type[nn.Module] = nn.ReLU,
    bias: bool = True,
    final_act: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h, bias=bias), act()]
        d = h
    layers.append(nn.Linear(d, out_dim, bias=bias))
    if final_act:
        layers.append(act())
    return nn.Sequential(*layers)


class JetModel(nn.Module):
    """Base class: every model maps a :class:`JetBatch` to per-jet losses and a per-jet anomaly score.

    ``losses(batch)`` returns a dict of per-jet tensors with at least ``total`` (minimised in training) and
    ``score`` (larger = more anomalous).  Unsupervised models must not read ``batch.label``.
    """

    supervised: bool = False

    def losses(self, b: JetBatch) -> dict[str, torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, b: JetBatch) -> torch.Tensor:
        return self.losses(b)["score"]

    def embed(self, b: JetBatch) -> torch.Tensor | None:
        """Graph-level latent vector (B, d) if the model has one."""
        return None

    def prepare(self, batches: Iterable[JetBatch]) -> None:
        """Hook run once before training (e.g. Deep SVDD centre)."""
