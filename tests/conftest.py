"""Shared fixtures: synthetic jets so that every test runs without the LHCO data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gaehep.data.dataset import JetPool, fit_scaler, make_batch

torch.set_num_threads(1)


def synthetic_pool(
    n_jets: int = 64, n_nodes: int = 30, seed: int = 0, prongs: int = 1, label: int = 0
) -> JetPool:
    """Toy jets: constituents scattered around ``prongs`` sub-axes near the jet axis, variable multiplicity."""
    rng = np.random.default_rng(seed)
    const = np.zeros((n_jets, n_nodes, 3), dtype=np.float32)
    n_real = rng.integers(3, n_nodes + 1, size=n_jets)
    jet_eta = rng.uniform(-1.5, 1.5, n_jets)
    jet_phi = rng.uniform(-np.pi, np.pi, n_jets)
    for i in range(n_jets):
        n = n_real[i]
        axes = rng.normal(0, 0.3, size=(prongs, 2))
        which = rng.integers(0, prongs, size=n)
        d = axes[which] + rng.normal(0, 0.08, size=(n, 2))
        pt = np.sort(rng.exponential(60.0, size=n) + 1.0)[::-1]
        const[i, :n, 0] = pt
        const[i, :n, 1] = jet_eta[i] + d[:, 0]
        const[i, :n, 2] = (jet_phi[i] + d[:, 1] + np.pi) % (2 * np.pi) - np.pi
    pt_j = const[..., 0].sum(1)
    e_j = (const[..., 0] * np.cosh(const[..., 1])).sum(1)
    m_j = rng.uniform(20, 400, n_jets).astype(np.float32)
    return JetPool(
        const=const,
        event=np.arange(n_jets) // 2,
        jet=np.arange(n_jets) % 2,
        pt=pt_j.astype(np.float32),
        eta=jet_eta.astype(np.float32),
        phi=jet_phi.astype(np.float32),
        m=m_j,
        e=np.maximum(e_j, pt_j).astype(np.float32),
        n_const=n_real.astype(np.int32),
        label=np.full(n_jets, label, dtype=np.int8),
    )


@pytest.fixture
def pool() -> JetPool:
    return synthetic_pool()


@pytest.fixture
def scaler(pool):
    return fit_scaler(pool)


@pytest.fixture
def batch(pool, scaler):
    return make_batch(pool, np.arange(16), scaler)


def permute_nodes(pool: JetPool, seed: int = 1, pad_extra: int = 0) -> JetPool:
    """Randomly permute the *stored* order of every jet's nodes (padding included) and optionally add padding."""
    rng = np.random.default_rng(seed)
    const = pool.const.copy()
    out = np.zeros((len(pool), const.shape[1] + pad_extra, 3), dtype=np.float32)
    for i in range(len(pool)):
        perm = rng.permutation(const.shape[1] + pad_extra)
        tmp = np.zeros((const.shape[1] + pad_extra, 3), dtype=np.float32)
        tmp[: const.shape[1]] = const[i]
        out[i] = tmp[perm]
    return JetPool(**{**{k: getattr(pool, k) for k in pool.__dataclass_fields__}, "const": out})
