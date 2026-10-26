"""Energy-weighted maximum mean discrepancy between two jets' energy flows.

A jet's *energy flow* is the set of constituent directions (deta, dphi) weighted by pT fractions
z_i = pT_i / sum pT.  With a characteristic kernel k (here a sum of Gaussians at several angular scales)

    MMD^2(P, Q) = sum_ij z_i z_j k(x_i, x_j) + sum_ab w_a w_b k(y_a, y_b) - 2 sum_ia z_i w_a k(x_i, y_a)

is a metric between the two energy distributions.  It is invariant to the order of the points, and
because soft particles carry little weight it is (approximately) infrared and collinear safe: the score
depends on the hard energy flow - prongs and their angular separation - rather than on the multiplicity
of soft constituents, which is what made the uniform-weight Chamfer score track multiplicity.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

DEFAULT_SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.4)


def _kernel(a: torch.Tensor, b: torch.Tensor, sigmas: Sequence[float]) -> torch.Tensor:
    d2 = torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist").pow(2)  # (B, Na, Nb)
    return sum(torch.exp(-d2 / (2.0 * s * s)) for s in sigmas)


def energy_mmd(
    pred: torch.Tensor,
    pred_w: torch.Tensor,
    target: torch.Tensor,
    target_w: torch.Tensor,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> torch.Tensor:
    """Per-jet MMD^2 between weighted point sets.

    pred (B, P, 2) with weights pred_w (B, P) summing to 1; target (B, T, 2) with weights target_w (B, T)
    summing to 1 (padded points must carry zero weight).  Returns (B,) values >= 0 (up to rounding).
    """
    k_pp = _kernel(pred, pred, sigmas)
    k_tt = _kernel(target, target, sigmas)
    k_pt = _kernel(pred, target, sigmas)
    t1 = torch.einsum("bi,bij,bj->b", pred_w, k_pp, pred_w)
    t2 = torch.einsum("bi,bij,bj->b", target_w, k_tt, target_w)
    t3 = torch.einsum("bi,bij,bj->b", pred_w, k_pt, target_w)
    return (t1 + t2 - 2.0 * t3).clamp(min=0.0)


def energy_nll(
    pred: torch.Tensor,
    pred_w: torch.Tensor,
    pred_sigma: torch.Tensor,
    target: torch.Tensor,
    target_w: torch.Tensor,
) -> torch.Tensor:
    """Per-jet energy-weighted cross-entropy  -sum_i z_i log sum_a w_a N(x_i | mu_a, sigma_a^2 I).

    The decoder is read as a Gaussian mixture density for the *energy flow*: constituent i (a sample of the
    flow, weighted by its energy fraction z_i) is scored by the mixture.  Like the MMD it is invariant to the
    order of both sets, to padding (zero weight) and to collinear splittings, and changes by O(z) under a
    soft emission; unlike the MMD (bounded by the kernel's range) it is unbounded for energy placed where the
    decoder has no mass, i.e. it behaves like a log-likelihood in the tails.  It contains the entropy of the
    target flow, so it also grows with the spread of the jet.

    pred (B, P, 2), pred_w (B, P) summing to 1, pred_sigma (B, P) > 0; target (B, T, 2), target_w (B, T).
    """
    d2 = (target[:, :, None, :] - pred[:, None, :, :]).pow(2).sum(-1)  # (B, T, P)
    var = pred_sigma.pow(2)[:, None, :]
    logp = torch.log(pred_w.clamp(min=1e-12))[:, None, :] - torch.log(2.0 * torch.pi * var) - 0.5 * d2 / var
    return -(target_w * torch.logsumexp(logp, dim=-1)).sum(-1)
