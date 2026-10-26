"""Training-free scores in the geometry of the energy-flow MMD: kernel mean embeddings + nearest neighbours.

The energy MMD of :mod:`gaehep.losses.mmd` is the RKHS distance between the jets' kernel mean embeddings
``mu(jet) = sum_i z_i k(x_i, .)``.  Random Fourier features turn that into an explicit finite-dimensional,
permutation- and padding-invariant, collinear-safe jet embedding,

    phi(jet) = sum_i z_i * sqrt(2/D) cos(W x_i + b)     (one block per Gaussian scale sigma, W ~ N(0, sigma^-2)),

with ``|phi(a) - phi(b)|^2 ~ MMD^2(a, b)`` for the *sum* of the Gaussian kernels.  Nothing is trained: the
anomaly score is a distance to the background-training jets -- the mean distance to the k nearest neighbours
(an optimal-transport "medoid"-type score in the spirit of Fraser et al., arXiv:2110.06948, whose training-
free Wasserstein k-medoids matched a VAE) or a Mahalanobis distance to the background distribution of the
embedding.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from gaehep.data.dataset import JetPool, Scaler, iterate_batches
from gaehep.losses.mmd import DEFAULT_SIGMAS


class KernelEmbedding:
    def __init__(self, sigmas: Sequence[float] = DEFAULT_SIGMAS, dim_per_scale: int = 128, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.w = [torch.randn(2, dim_per_scale, generator=g) / s for s in self.sigmas]
        self.b = [2 * torch.pi * torch.rand(dim_per_scale, generator=g) for _ in self.sigmas]
        self.scale = (2.0 / dim_per_scale) ** 0.5

    def __call__(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """coords (B, N, 2) in (deta, dphi), z (B, N) pT fractions (0 on padding) -> (B, S * D)."""
        parts = [
            self.scale * (z[..., None] * torch.cos(coords @ w + b)).sum(1)
            for w, b in zip(self.w, self.b, strict=True)
        ]
        return torch.cat(parts, dim=-1)


@torch.no_grad()
def embed_pool(pool: JetPool, scaler: Scaler, emb: KernelEmbedding, batch_size: int = 2048) -> np.ndarray:
    out = []
    for b in iterate_batches(pool, scaler, batch_size):
        pt = b.raw[..., 0] * b.mask
        z = pt / pt.sum(1, keepdim=True).clamp(min=1e-12)
        out.append(emb(b.coords, z).numpy())
    return np.concatenate(out).astype(np.float32)


class EmbeddingScorer:
    """PCA-reduced embedding of a background bank; scores = kNN distance / Mahalanobis distance."""

    def __init__(self, bank: np.ndarray, n_pca: int = 64, k: int = 10):
        self.mean = bank.mean(0)
        _u, s, vt = np.linalg.svd(bank[: min(len(bank), 50_000)] - self.mean, full_matrices=False)
        self.vt = vt[:n_pca].astype(np.float32)
        self.sv = (s[:n_pca] / np.sqrt(min(len(bank), 50_000) - 1)).astype(np.float32)  # std along each PC
        self.k = k
        self.bank = torch.from_numpy(self._project(bank))
        self.bank_sq = (self.bank**2).sum(1)

    def _project(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) @ self.vt.T).astype(np.float32)

    @torch.no_grad()
    def scores(self, x: np.ndarray, batch: int = 1024) -> dict[str, np.ndarray]:
        q_all = self._project(x)
        knn, knn_w = [], []
        bank_w = self.bank / torch.from_numpy(self.sv)
        bank_w_sq = (bank_w**2).sum(1)
        for s in range(0, len(q_all), batch):
            q = torch.from_numpy(q_all[s : s + batch])
            d2 = (q**2).sum(1)[:, None] + self.bank_sq[None, :] - 2.0 * q @ self.bank.T
            knn.append(d2.clamp(min=0).topk(self.k, dim=1, largest=False).values.mean(1))
            qw = q / torch.from_numpy(self.sv)
            d2w = (qw**2).sum(1)[:, None] + bank_w_sq[None, :] - 2.0 * qw @ bank_w.T
            knn_w.append(d2w.clamp(min=0).topk(self.k, dim=1, largest=False).values.mean(1))
        maha = ((q_all / self.sv) ** 2).sum(1)
        return {"knn": torch.cat(knn).numpy(), "knn_white": torch.cat(knn_w).numpy(), "maha": maha}
