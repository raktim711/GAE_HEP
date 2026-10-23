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
