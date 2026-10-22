"""Masked (symmetric) Chamfer distance between padded point sets.

The Chamfer distance compares two *sets*: it is invariant to the order of the points in either input,
which is what makes a set autoencoder's reconstruction error a permutation-invariant anomaly score.
"""

from __future__ import annotations

import torch


def masked_chamfer(
    pred: torch.Tensor, pmask: torch.Tensor, target: torch.Tensor, tmask: torch.Tensor
) -> torch.Tensor:
    """Per-set Chamfer distance.

    CD(P, T) = mean_{p in P} min_{t in T} |p - t|^2 + mean_{t in T} min_{p in P} |p - t|^2

    pred : (B, P, D), pmask : (B, P) bool;  target : (B, T, D), tmask : (B, T) bool.
    Returns (B,) distances (0 for sets without real points).
    """
    d = torch.cdist(pred, target, compute_mode="donot_use_mm_for_euclid_dist").pow(2)  # (B, P, T)
    d_pt = d.masked_fill(~tmask[:, None, :], float("inf")).amin(2)  # nearest target for every prediction
    d_tp = d.masked_fill(~pmask[:, :, None], float("inf")).amin(1)  # nearest prediction for every target
    ok = (pmask.any(1) & tmask.any(1))[:, None]
    d_pt = torch.where(pmask & ok, d_pt, torch.zeros_like(d_pt))
    d_tp = torch.where(tmask & ok, d_tp, torch.zeros_like(d_tp))
    n_p = pmask.sum(1).clamp(min=1).to(pred.dtype)
    n_t = tmask.sum(1).clamp(min=1).to(pred.dtype)
    return d_pt.sum(1) / n_p + d_tp.sum(1) / n_t
