import torch

from gaehep.losses import energy_mmd, masked_chamfer, masked_sinkhorn


def _sets(seed=0, b=4, n=7, d=3):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(b, n, d, generator=g)
    m = torch.ones(b, n, dtype=torch.bool)
    m[0, 5:] = False
    m[2, 3:] = False
    return a, m


def test_chamfer_zero_on_identical_sets():
    a, m = _sets()
    assert torch.allclose(masked_chamfer(a, m, a, m), torch.zeros(4), atol=1e-6)


def test_chamfer_permutation_invariant_and_symmetric():
    a, m = _sets(0)
    b, mb = _sets(1)
    perm = torch.randperm(a.shape[1])
    d0 = masked_chamfer(a, m, b, mb)
    assert torch.allclose(d0, masked_chamfer(a[:, perm], m[:, perm], b, mb), atol=1e-6)
    assert torch.allclose(d0, masked_chamfer(b, mb, a, m), atol=1e-6)


def test_chamfer_ignores_padded_points():
    a, m = _sets(0)
    b, mb = _sets(1)
    a2 = a.clone()
    a2[~m] = 1e3
    assert torch.allclose(masked_chamfer(a, m, b, mb), masked_chamfer(a2, m, b, mb), atol=1e-5)


def test_chamfer_known_value():
    p = torch.tensor([[[0.0], [1.0]]])
    t = torch.tensor([[[0.0], [3.0]]])
    m = torch.ones(1, 2, dtype=torch.bool)
    # pred->target: 0, (1-0)^2=1 -> mean 0.5 ; target->pred: 0, (3-1)^2=4 -> mean 2
    assert torch.allclose(masked_chamfer(p, m, t, m), torch.tensor([2.5]))


def test_sinkhorn_small_on_identical_and_invariant():
    a, m = _sets(0, d=2)
    a = a * 3  # well separated points -> nearly identity plan
    c_same = masked_sinkhorn(a, m, a, m, eps=0.01, n_iter=200)
    assert torch.all(c_same < 1e-2)
    b, mb = _sets(1, d=2)
    perm = torch.randperm(a.shape[1])
    c0 = masked_sinkhorn(a, m, b, mb, eps=0.05)
    c1 = masked_sinkhorn(a[:, perm], m[:, perm], b, mb, eps=0.05)
    assert torch.allclose(c0, c1, rtol=1e-4, atol=1e-5)
    assert torch.all(torch.isfinite(c0))


def _flows(seed=0, b=3, n=9):
    g = torch.Generator().manual_seed(seed)
    x = 0.3 * torch.randn(b, n, 2, generator=g)
    w = torch.rand(b, n, generator=g) ** 3
    w[0, 6:] = 0.0  # padding carries zero weight
    return x, w / w.sum(1, keepdim=True)


def test_energy_mmd_zero_on_identical_and_positive_otherwise():
    x, w = _flows(0)
    y, v = _flows(1)
    assert torch.allclose(energy_mmd(x, w, x, w), torch.zeros(3), atol=1e-6)
    d = energy_mmd(x, w, y, v)
    assert torch.all(d > 1e-3)
    assert torch.allclose(d, energy_mmd(y, v, x, w), atol=1e-6)


def test_energy_mmd_permutation_and_padding_invariant():
    x, w = _flows(0)
    y, v = _flows(1)
    perm = torch.randperm(x.shape[1])
    d0 = energy_mmd(x, w, y, v)
    assert torch.allclose(d0, energy_mmd(x[:, perm], w[:, perm], y, v), atol=1e-6)
    x2 = x.clone()
    x2[w == 0] = 5.0  # positions of zero-weight points are irrelevant
    assert torch.allclose(d0, energy_mmd(x2, w, y, v), atol=1e-6)


def test_energy_mmd_infrared_and_collinear_safe():
    x, w = _flows(0)
    y, v = _flows(1)
    d0 = energy_mmd(x, w, y, v)
    # collinear split: particle 0 -> two particles at the same position sharing its weight
    xs = torch.cat([x, x[:, :1]], dim=1)
    ws = torch.cat([w, 0.5 * w[:, :1]], dim=1)
    ws[:, 0] *= 0.5
    assert torch.allclose(d0, energy_mmd(xs, ws, y, v), atol=1e-6)
    # soft emission: a particle with weight 1e-4 anywhere barely changes the distance
    xe = torch.cat([x, torch.full_like(x[:, :1], 0.7)], dim=1)
    we = torch.cat([w * (1 - 1e-4), torch.full_like(w[:, :1], 1e-4)], dim=1)
    assert torch.allclose(d0, energy_mmd(xe, we, y, v), atol=5e-3)
