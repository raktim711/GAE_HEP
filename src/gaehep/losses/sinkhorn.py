"""Entropy-regularised optimal-transport (Sinkhorn) distance between weighted, padded point sets.

An energy-mover's-distance-like alternative to Chamfer: target points can be weighted by their pT
fraction so that hard constituents dominate the reconstruction error.  Log-domain iterations keep it
stable for small regularisation.  Invariant to the order of points in both sets.
"""

from __future__ import annotations

import torch


def masked_sinkhorn(
    pred: torch.Tensor,
    pmask: torch.Tensor,
    target: torch.Tensor,
    tmask: torch.Tensor,
    pweight: torch.Tensor | None = None,
    tweight: torch.Tensor | None = None,
    eps: float = 0.05,
    n_iter: int = 50,
) -> torch.Tensor:
    """Per-set entropic OT cost <P, C> with squared-Euclidean ground cost.

    Weights default to uniform over the real points of each set and are normalised to sum to one.
    Returns (B,) costs (0 for sets without real points).
    """
    c = torch.cdist(pred, target, compute_mode="donot_use_mm_for_euclid_dist").pow(2)  # (B, P, T)

    def norm(w: torch.Tensor | None, m: torch.Tensor) -> torch.Tensor:
        w = m.to(pred.dtype) if w is None else w * m
        return w / w.sum(1, keepdim=True).clamp(min=1e-12)

    a, b = norm(pweight, pmask), norm(tweight, tmask)
    log_a, log_b = a.clamp(min=1e-30).log(), b.clamp(min=1e-30).log()
    neg_inf = float("-inf")
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(n_iter):
        s = ((g[:, None, :] - c) / eps).masked_fill(~tmask[:, None, :], neg_inf)
        f = torch.where(pmask, eps * (log_a - torch.logsumexp(s, dim=2)), torch.zeros_like(f))
        s = ((f[:, :, None] - c) / eps).masked_fill(~pmask[:, :, None], neg_inf)
        g = torch.where(tmask, eps * (log_b - torch.logsumexp(s, dim=1)), torch.zeros_like(g))
    pair = pmask[:, :, None] & tmask[:, None, :]
    log_p = ((f[:, :, None] + g[:, None, :] - c) / eps).masked_fill(~pair, neg_inf)
    cost = (log_p.exp() * c.masked_fill(~pair, 0.0)).sum((1, 2))
    ok = pmask.any(1) & tmask.any(1)
    return torch.where(ok, cost, torch.zeros_like(cost))
