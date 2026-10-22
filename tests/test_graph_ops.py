import numpy as np
import torch

from gaehep.graphs.ops import gather_neighbors, masked_knn, masked_max, masked_mean


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
