import math

import numpy as np
import pytest
import torch

from gaehep.graphs.ops import (
    gather_neighbors,
    knn_tie_order,
    masked_knn,
    masked_max,
    masked_mean,
    masked_radius_mask,
)


def test_masked_knn_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    b, n, k = 5, 12, 4
    coords = torch.randn(b, n, 2, generator=g)
    mask = torch.rand(b, n, generator=g) > 0.3
    mask[:, 0] = True
    idx, valid = masked_knn(coords, mask, k)
    for gi in range(b):
        real = np.flatnonzero(mask[gi].numpy())
        for i in range(n):
            got = set(idx[gi, i][valid[gi, i]].tolist())
            if not mask[gi, i]:
                assert got == set()
                continue
            others = [j for j in real if j != i]
            d = [float(torch.dist(coords[gi, i], coords[gi, j])) for j in others]
            want = {others[t] for t in np.argsort(d)[:k]}
            assert got == want


def test_masked_knn_breaks_distance_ties_by_pt_then_deta_dphi():
    """Equidistant neighbours: the larger pT first, then the smaller deta, then the smaller dphi."""
    #               query   (0,1)   (1,0)   (0,-1)  (-1,0)  (1,0)   (1,0)   far
    coords = torch.tensor([[[0.0, 0.0], [0, 1], [1, 0], [0, -1], [-1, 0], [1, 0], [1, 0], [5, 5]]])
    pt = torch.tensor([[9.0, 1.0, 2.0, 3.0, 3.0, 7.0, 2.0, 50.0]])
    mask = torch.ones(1, 8, dtype=torch.bool)
    order = knn_tie_order(pt, coords)
    assert order[0].tolist() == [7, 0, 5, 4, 3, 2, 6, 1]
    idx, valid = masked_knn(coords, mask, 6, tie_order=order)
    assert valid[0, 0].all()
    assert idx[0, 0].tolist() == [
        5,
        4,
        3,
        2,
        6,
        1,
    ]  # all at distance 1: pT 7, 3 (deta -1), 3 (deta 0), 2, 2, 1


def test_masked_knn_with_ties_is_exactly_permutation_equivariant():
    """Catches index-based tie-breaking: on a coarse grid (many equal distances, coincident points with
    distinct pT) the neighbour lists, slot order included, must follow any permutation of the nodes exactly.
    Exact duplicates (same pT and position) are interchangeable, so the neighbours are compared by content."""
    g = torch.Generator().manual_seed(0)
    b, n, k = 4, 24, 6
    coords = torch.randint(-2, 3, (b, n, 2), generator=g).float() * 0.1
    pt = torch.randint(1, 9, (b, n), generator=g).float()
    mask = torch.rand(b, n, generator=g) > 0.2
    pt = pt * mask

    def neighbours(c, p, m):
        idx, valid = masked_knn(c, m, k, tie_order=knn_tie_order(p, c))
        node = torch.cat([p[..., None], c], dim=-1)  # (pT, deta, dphi) of every node
        content = gather_neighbors(node, idx)  # (B, N, k, 3), slot order kept
        return content.masked_fill(~valid[..., None], -1.0), valid

    ref, valid = neighbours(coords, pt, mask)
    assert (valid.sum(-1) == k).sum() > 50
    for _ in range(5):
        p = torch.randperm(n, generator=g)
        got, valid_p = neighbours(coords[:, p], pt[:, p], mask[:, p])
        assert torch.equal(valid_p, valid[:, p]) and torch.equal(got, ref[:, p])


def test_masked_knn_without_ties_is_unchanged_by_the_tie_break():
    """Without distance ties the tie-break changes nothing: same neighbours in the same (ascending) slot order
    as torch.topk(sorted=True) on the masked distances, with or without ``tie_order``."""
    g = torch.Generator().manual_seed(3)
    b, n, k = 6, 40, 8
    coords = torch.randn(b, n, 2, generator=g)
    mask = torch.rand(b, n, generator=g) > 0.25
    pt = torch.rand(b, n, generator=g) * mask
    d = torch.cdist(coords, coords).masked_fill(~mask[:, None, :], float("inf"))
    d = d.masked_fill(torch.eye(n, dtype=torch.bool), float("inf"))
    vals, ref = torch.topk(d, k, dim=-1, largest=False, sorted=True)
    ok = torch.isfinite(vals) & mask[:, :, None]
    for order in (None, knn_tie_order(pt, coords)):
        idx, valid = masked_knn(coords, mask, k, tie_order=order)
        assert torch.equal(valid, ok)
        assert torch.equal(idx[valid], ref[valid])


def test_knn_small_graph_has_no_spurious_neighbours():
    coords = torch.randn(1, 6, 2)
    mask = torch.tensor([[True, True, False, False, False, False]])
    idx, valid = masked_knn(coords, mask, 4)
    assert valid[0, 0].sum() == 1 and valid[0, 1].sum() == 1
    assert idx[0, 0][valid[0, 0]].item() == 1


def test_gather_neighbors():
    x = torch.arange(2 * 3 * 2, dtype=torch.float32).view(2, 3, 2)
    idx = torch.tensor([[[1, 2], [0, 2], [0, 1]], [[2, 2], [0, 0], [1, 1]]])
    out = gather_neighbors(x, idx)
    for b in range(2):
        for i in range(3):
            for j in range(2):
                assert torch.equal(out[b, i, j], x[b, idx[b, i, j]])


def test_pooling_ignores_padding():
    x = torch.randn(3, 5, 4)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    x_garbage = x.clone()
    x_garbage[~mask] = 1e6
    assert torch.allclose(masked_mean(x, mask), masked_mean(x_garbage, mask))
    assert torch.allclose(masked_max(x, mask), masked_max(x_garbage, mask))
    assert torch.equal(masked_mean(x, mask)[2], torch.zeros(4))
    assert torch.allclose(masked_mean(x, mask)[0], x[0, :2].mean(0))


def _random_cloud(seed: int, b: int = 3, n: int = 9, scale: float = 0.5):
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(b, n, 2, generator=g, dtype=torch.float64) * scale
    mask = torch.rand(b, n, generator=g) > 0.35
    mask[:, 0] = True
    mask[-1, 1:] = False  # one graph with a single real node
    return coords, mask


@pytest.mark.parametrize("radius", [0.25, 0.6, 1.5])
@pytest.mark.parametrize("include_self", [True, False])
def test_masked_radius_mask_matches_double_loop(radius, include_self):
    """Catches a wrong comparison (<= vs <), a missing pad mask or a mishandled diagonal vs a plain loop."""
    coords, mask = _random_cloud(0)
    got = masked_radius_mask(coords, mask, radius, include_self=include_self)
    b, n, _ = coords.shape
    assert got.shape == (b, n, n) and got.dtype == torch.bool
    for gi in range(b):
        pts = coords[gi].tolist()
        for i in range(n):
            for j in range(n):
                want = bool(mask[gi, i] and mask[gi, j] and math.dist(pts[i], pts[j]) < radius)
                want = want and (include_self or i != j)
                assert bool(got[gi, i, j]) == want, (gi, i, j)


def test_masked_radius_mask_sequence_stacks_the_scalar_masks():
    """Catches the multi-radius form (used by EWMPBlock) drifting from the per-radius masks or their order."""
    coords, mask = _random_cloud(1)
    radii = (0.2, 0.7, 0.45)
    got = masked_radius_mask(coords, mask, radii)
    assert got.shape == (len(radii), *mask.shape, mask.shape[1])
    for k, r in enumerate(radii):
        assert torch.equal(got[k], masked_radius_mask(coords, mask, r))
    assert masked_radius_mask(coords, mask, ()).shape == (0, *mask.shape, mask.shape[1])


def test_masked_radius_mask_is_symmetric():
    """Catches an asymmetric pad mask (rows masked but not columns) or a non-symmetric distance."""
    coords, mask = _random_cloud(2, b=4, n=20)
    a = masked_radius_mask(coords.float(), mask, 0.5)
    assert torch.equal(a, a.transpose(1, 2))


def test_masked_radius_mask_padded_nodes_are_never_neighbours():
    """Catches padded nodes entering a ball when they sit on top of real nodes (or at the origin)."""
    coords, mask = _random_cloud(3, n=12)
    coords = coords.clone()
    for gi in range(len(coords)):
        coords[gi, ~mask[gi]] = coords[gi, 0].clone()  # padding exactly on a real node: distance 0
    a = masked_radius_mask(coords, mask, 1e6)
    pad = ~mask
    assert not a[pad[:, :, None].expand_as(a)].any()
    assert not a[pad[:, None, :].expand_as(a)].any()


def test_masked_radius_mask_tiny_radius_gives_only_self_pairs():
    """Catches the self pair being dropped (or kept for include_self=False / r <= 0) at a vanishing radius."""
    coords, mask = _random_cloud(4)
    eye = torch.eye(mask.shape[1], dtype=torch.bool)
    diag = eye & mask[:, :, None]
    assert torch.equal(masked_radius_mask(coords, mask, 1e-9), diag)
    assert not masked_radius_mask(coords, mask, 1e-9, include_self=False).any()
    # open ball of radius <= 0 is empty: no pairs at all, not even self (the EWMPBlock convention)
    assert not masked_radius_mask(coords, mask, 0.0).any()
    assert not masked_radius_mask(coords, mask, -1.0).any()


def test_masked_radius_mask_huge_radius_gives_all_real_pairs():
    """Catches real pairs being lost at large radius, e.g. a distance overflow or an extra truncation."""
    coords, mask = _random_cloud(5)
    real = mask[:, :, None] & mask[:, None, :]
    assert torch.equal(masked_radius_mask(coords, mask, 1e6), real)
    assert torch.equal(masked_radius_mask(coords, mask, float("inf")), real)
    eye = torch.eye(mask.shape[1], dtype=torch.bool)
    assert torch.equal(masked_radius_mask(coords, mask, 1e6, include_self=False), real & ~eye)


def test_masked_radius_mask_is_permutation_equivariant():
    """Catches any dependence on the stored node order (e.g. index-based tie-breaking or a fixed pad layout)."""
    coords, mask = _random_cloud(6, b=4, n=15)
    a = masked_radius_mask(coords, mask, 0.5)
    g = torch.Generator().manual_seed(7)
    for gi in range(len(coords)):
        p = torch.randperm(coords.shape[1], generator=g)
        ap = masked_radius_mask(coords[gi : gi + 1, p], mask[gi : gi + 1, p], 0.5)[0]
        assert torch.equal(ap, a[gi][p][:, p])
