"""Normalising-flow density on a trained model's graph-level latents ("M3").

The anomaly score is -log p(z) under a neural spline flow fitted to background-training latents; since z
is permutation invariant, so is the score.  Latents (and context) are standardised before fitting.

With ``context_dim > 0`` the flow is *conditional*, p(z | c), e.g. with c = (log m_j, log pT_j): the score
-log p(z | m_j) then measures how unusual the jet's substructure is *for its mass* (mass-decorrelated).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import zuko


class LatentFlow:
    def __init__(
        self,
        dim: int,
        transforms: int = 4,
        hidden: tuple[int, ...] = (128, 128),
        seed: int = 0,
        context_dim: int = 0,
    ):
        torch.manual_seed(seed)
        self.context_dim = context_dim
        self.flow = zuko.flows.NSF(
            features=dim, context=context_dim, transforms=transforms, hidden_features=hidden
        )
        self.mean = np.zeros(dim, dtype=np.float32)
        self.std = np.ones(dim, dtype=np.float32)
        self.c_mean = np.zeros(context_dim, dtype=np.float32)
        self.c_std = np.ones(context_dim, dtype=np.float32)
        self.history: dict[str, list[float]] = {"train_nll": [], "val_nll": []}

    def _t(self, z: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((z - self.mean) / self.std).astype(np.float32))

    def _c(self, c: np.ndarray | None) -> torch.Tensor | None:
        if not self.context_dim:
            return None
        if c is None:
            raise ValueError("this flow is conditional: pass the context")
        return torch.from_numpy(((c - self.c_mean) / self.c_std).astype(np.float32))

    def _log_prob(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        return self.flow(c).log_prob(x)

    def fit(
        self,
        z_train: np.ndarray,
        z_val: np.ndarray,
        epochs: int = 60,
        batch_size: int = 1024,
        lr: float = 1e-3,
        patience: int = 8,
        seed: int = 0,
        c_train: np.ndarray | None = None,
        c_val: np.ndarray | None = None,
    ) -> LatentFlow:
        self.mean = z_train.mean(0).astype(np.float32)
        self.std = np.maximum(z_train.std(0), 1e-6).astype(np.float32)
        if self.context_dim:
            self.c_mean = c_train.mean(0).astype(np.float32)
            self.c_std = np.maximum(c_train.std(0), 1e-6).astype(np.float32)
        xt, xv = self._t(z_train), self._t(z_val)
        ct, cv = self._c(c_train), self._c(c_val)
        opt = torch.optim.Adam(self.flow.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        g = torch.Generator().manual_seed(seed)
        best, best_state, bad = float("inf"), None, 0
        for _ in range(epochs):
            self.flow.train()
            perm = torch.randperm(len(xt), generator=g)
            tot, n = 0.0, 0
            for s in range(0, len(xt), batch_size):
                sel = perm[s : s + batch_size]
                loss = -self._log_prob(xt[sel], None if ct is None else ct[sel]).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 5.0)
                opt.step()
                tot, n = tot + loss.item() * len(sel), n + len(sel)
            sched.step()
            val = float(self._nll_t(xv, cv).mean())
            self.history["train_nll"].append(tot / n)
            self.history["val_nll"].append(val)
            if val < best - 1e-4:
                best, best_state, bad = val, copy.deepcopy(self.flow.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.flow.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _nll_t(self, x: torch.Tensor, c: torch.Tensor | None, batch_size: int = 8192) -> np.ndarray:
        self.flow.eval()
        out = [
            -self._log_prob(x[s : s + batch_size], None if c is None else c[s : s + batch_size])
            for s in range(0, len(x), batch_size)
        ]
        return torch.cat(out).numpy()

    def nll(
        self,
        z: np.ndarray,
        standardised: torch.Tensor | None = None,
        batch_size: int = 8192,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        x = self._t(z) if standardised is None else standardised
        return self._nll_t(x, self._c(context), batch_size)
