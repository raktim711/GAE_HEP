"""Batched, masked graph operations on dense padded tensors.

Conventions: ``B`` graphs per batch, ``N`` (padded) nodes per graph, ``F`` features; ``mask`` (B, N) is
True for real nodes.  Every op here is permutation-equivariant in the node dimension (or invariant, for
pooling) and ignores padded nodes, which is what makes the models built on top permutation invariant.

Dense tensors (instead of PyG's sparse ``edge_index``) are faster on CPU for jets with N <= ~100 nodes and
avoid PyG's compiled extensions (torch-cluster / torch-scatter).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def knn_tie_order(pt: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """The node priority used to break k-NN distance ties: larger pT first, then smaller deta, then dphi.

    Parameters
    ----------
    pt : (B, N) transverse momenta (0 on padding; padded nodes are excluded by ``masked_knn``'s mask anyway).
    coords : (B, N, 2) (deta, dphi).

    Returns
    -------
    order : (B, N) long, a permutation of the nodes of each graph, highest priority first.  It depends only
    on the nodes' own (pT, deta, dphi), so it is permutation-equivariant; only exact duplicates (identical
    pT, deta and dphi, hence identical features) keep their storage order, and those are interchangeable.
    """
    # stable sorts from the least to the most significant key (LSD radix order)
    order = torch.sort(coords[..., 1], dim=-1, stable=True).indices
    order = order.gather(-1, torch.sort(coords[..., 0].gather(-1, order), dim=-1, stable=True).indices)
    by_pt = torch.sort(pt.gather(-1, order), dim=-1, descending=True, stable=True).indices
    return order.gather(-1, by_pt)


def masked_knn(
    coords: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    exclude_self: bool = True,
    tie_order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k nearest neighbours of every node among the *real* nodes of its graph.

    The neighbours are listed by increasing distance.  Equal distances are ordered by ``tie_order`` (a
    node priority, e.g. :func:`knn_tie_order`: larger pT first, then (deta, dphi)), which makes both the
    neighbour sets and the slot order exactly permutation-equivariant.  Without ``tie_order`` ties fall back
    to the storage index (not permutation-invariant when ties occur).  Where no distance ties occur the
    result is identical with and without ``tie_order``, and identical to ``torch.topk(sorted=True)``.

    Parameters
    ----------
    coords : (B, N, D) positions used for the distance (e.g. (deta, dphi) or hidden features).
    mask : (B, N) bool, real nodes.
    k : number of neighbours; reduced to N-1 if the padded size is smaller.
    tie_order : optional (B, N) long, a permutation of the nodes of each graph, highest priority first.

    Returns
    -------
    idx : (B, N, k') long, neighbour indices (k' = min(k, N - exclude_self)); invalid slots point at the node itself.
    valid : (B, N, k') bool, True where the neighbour exists (real target, real source, not self).
    """
    b, n, _ = coords.shape
    kk = min(k, n - 1 if exclude_self else n)
    if kk <= 0:
        empty = torch.zeros(b, n, 0, dtype=torch.long, device=coords.device)
        return empty, empty.bool()
    d = torch.cdist(coords, coords, compute_mode="donot_use_mm_for_euclid_dist")
    d = d.masked_fill(~mask[:, None, :], float("inf"))
    if exclude_self:
        eye = torch.eye(n, dtype=torch.bool, device=coords.device)
        d = d.masked_fill(eye, float("inf"))
    # one more than needed: a tie at the k-th place shows up as an equal (k+1)-th distance
    vals, idx = torch.topk(d, min(kk + 1, n), dim=-1, largest=False, sorted=True)
    fin = torch.isfinite(vals)
    # rows whose k nearest are ambiguous: equal finite distances among the k + 1 nearest.  Only these rows
    # are re-sorted with the tie-break; the others keep topk's order, which is then the unique ascending one.
    tied = ((vals[..., 1:] == vals[..., :-1]) & fin[..., 1:]).any(-1)
    vals, idx, fin = vals[..., :kk], idx[..., :kk], fin[..., :kk]
    if bool(tied.any()):
        gi, ni = tied.nonzero(as_tuple=True)
        if tie_order is None:
            cols = torch.arange(n, device=coords.device).expand(len(gi), n)
        else:
            cols = tie_order[gi]  # (M, N) candidates in priority order
        # a stable sort by distance of the candidates in priority order: equal distances keep that order
        pos = torch.sort(d[gi, ni].gather(-1, cols), dim=-1, stable=True).indices[:, :kk]
        idx = idx.index_put((gi, ni), cols.gather(-1, pos))
    valid = fin & mask[:, :, None]
    self_idx = torch.arange(n, device=coords.device).view(1, n, 1).expand(b, n, kk)
    idx = torch.where(valid, idx, self_idx)
    return idx, valid


def masked_radius_mask(
    coords: torch.Tensor,
    mask: torch.Tensor,
    radius: float | Sequence[float],
    include_self: bool = True,
) -> torch.Tensor:
    """Adjacency of the fixed-radius (open) balls among the *real* nodes of each graph.

    ``A[b, i, j]`` is True iff nodes i and j are both real and ``||coords[b, i] - coords[b, j]|| < radius``:
    strict ``<`` on the Euclidean distance of :func:`torch.cdist` with
    ``compute_mode="donot_use_mm_for_euclid_dist"`` (the exact difference-based distance, not the
    matrix-multiply expansion, so ``d_ii`` is exactly 0 and ``d_ij == d_ji``).  Rows and columns of padded
    nodes are all False whatever their coordinates.  The mask is symmetric and permutation-equivariant, and
    has no tie-breaking (unlike k-NN).

    Self pairs: with ``include_self=True`` the diagonal obeys the same rule, so every real node is its own
    neighbour for ``radius > 0``, while for ``radius <= 0`` the open ball is empty (no pairs at all, self
    included).  ``include_self=False`` sets the diagonal to False for every radius.

    Parameters
    ----------
    coords : (B, N, D) positions (e.g. (deta, dphi)).
    mask : (B, N) bool, real nodes.
    radius : a float, or a sequence (list / tuple) of R floats; the distances are computed once either way.

    Returns
    -------
    (B, N, N) bool for a float ``radius``; (R, B, N, N) bool for a sequence, one slice per radius.
    """
    d = torch.cdist(coords, coords, compute_mode="donot_use_mm_for_euclid_dist")  # (B, N, N)
    real = mask[:, :, None] & mask[:, None, :]
    if not include_self:
        real = real & ~torch.eye(coords.shape[1], dtype=torch.bool, device=coords.device)
    if not isinstance(radius, Sequence):
        return (d < float(radius)) & real
    if len(radius) == 0:
        return real.new_zeros((0, *real.shape))
    return torch.stack([(d < float(r)) & real for r in radius])


def gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B, N, F), idx (B, N, k) -> (B, N, k, F) with out[b, i, j] = x[b, idx[b, i, j]]."""
    b, n, f = x.shape
    k = idx.shape[-1]
    base = torch.arange(b, device=x.device).view(b, 1, 1) * n
    return x.reshape(b * n, f)[(idx + base).reshape(-1)].view(b, n, k, f)


def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum over real nodes: (B, N, F) -> (B, F)."""
    return (x * mask[..., None]).sum(1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real nodes: (B, N, F) -> (B, F); zero for empty graphs."""
    return masked_sum(x, mask) / mask.sum(1, keepdim=True).clamp(min=1).to(x.dtype)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max over real nodes: (B, N, F) -> (B, F); zero for empty graphs."""
    out = x.masked_fill(~mask[..., None], float("-inf")).amax(1)
    return torch.where(mask.any(1, keepdim=True), out, torch.zeros_like(out))
