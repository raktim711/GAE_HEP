"""Random-network distillation ("RND") with permutation-invariant EdgeConv networks: a fixed-target score.

A *teacher* (a randomly initialised, frozen jet encoder + linear head; its outputs are standardised on
background-training jets) defines a fixed, smooth, permutation-invariant embedding of every jet.  A *student*
of the same type is trained on background jets to reproduce the teacher's embedding; the anomaly score is the
student's squared error (Burda et al., "Exploration by random network distillation", 2018).  There is no
reconstruction of the input, so the score does not reward "simple" jets (the complexity bias of
autoencoders), and, unlike Deep SVDD, there is no trivial collapsed solution because the target is fixed.
The literature sweep (docs/literature.md) found fixed-target scores to beat reconstruction scores on
Dark Machines events (Ostdiek, arXiv:2109.01695).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from gaehep.data.dataset import JetBatch
from gaehep.models.base import JetModel, mlp
from gaehep.models.edgeconv import EdgeConvEncoder


class RandomDistillation(JetModel):
    def __init__(
        self,
        in_dim: int = 7,
        hidden: Sequence[int] = (48, 48, 48),
        teacher_hidden: Sequence[int] = (64, 64, 64),
        k: int = 8,
        out_dim: int = 32,
        head_hidden: Sequence[int] = (128,),
        aggr: str = "mean",
        dynamic: bool = True,
        teacher_seed: int = 12345,
    ):
        super().__init__()
        self.student = EdgeConvEncoder(in_dim, hidden, k=k, dynamic=dynamic, aggr=aggr)
        self.student_head = mlp(2 * self.student.out_dim, list(head_hidden), out_dim)
        with torch.random.fork_rng(devices=[]):  # the teacher does not depend on the training seed
            torch.manual_seed(teacher_seed)
            self.teacher = EdgeConvEncoder(in_dim, teacher_hidden, k=k, dynamic=dynamic, aggr=aggr)
            self.teacher_head = mlp(2 * self.teacher.out_dim, [], out_dim)
        for p in (*self.teacher.parameters(), *self.teacher_head.parameters()):
            p.requires_grad_(False)
        self.register_buffer("t_mean", torch.zeros(out_dim))
        self.register_buffer("t_std", torch.ones(out_dim))
        self.register_buffer("ready", torch.tensor(False))

    def _teacher(self, b: JetBatch) -> torch.Tensor:
        h = self.teacher(b.x, b.mask, b.coords)
        return self.teacher_head(self.teacher.pool(h, b.mask))

    def _target(self, b: JetBatch) -> torch.Tensor:
        with torch.no_grad():
            return (self._teacher(b) - self.t_mean) / self.t_std

    def embed(self, b: JetBatch) -> torch.Tensor:
        return self.student_head(self.student.pool(self.student(b.x, b.mask, b.coords), b.mask))

    @torch.no_grad()
    def prepare(self, batches: Iterable[JetBatch]) -> None:
        if bool(self.ready):
            return
        t = torch.cat([self._teacher(b) for b in batches])
        self.t_mean.copy_(t.mean(0))
        self.t_std.copy_(t.std(0).clamp(min=1e-6))
        self.ready.fill_(True)

    def score_and_embed(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(b)
        return ((z - self._target(b)) ** 2).mean(-1), z

    def losses(self, b: JetBatch) -> dict[str, torch.Tensor]:
        if not bool(self.ready):
            raise RuntimeError("RND teacher statistics not initialised; call prepare() first")
        d = ((self.embed(b) - self._target(b)) ** 2).mean(-1)
        return {"total": d, "score": d}
