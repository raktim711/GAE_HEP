"""The AIMS GAE on complete graphs: GCN and GIN(eps=0) give every node of a graph the same embedding."""

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from gaehep.data.dataset import make_batch
from gaehep.models.legacy_gae import LegacyGAE, complete_graph


def _node_spread(model, batch):
    model.eval()
    with torch.no_grad():
        _, z, xhat, g = model.node_outputs(batch)
    spreads = []
    for gi in torch.unique(g):
        zi = z[g == gi]
        if len(zi) > 1:
            spreads.append(float((zi - zi.mean(0)).abs().max()))
    return max(spreads), z, xhat, g


def test_complete_graph_edges():
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    flat_real, ei, batch = complete_graph(mask)
    assert len(flat_real) == 5 and ei.shape[1] == 3 * 2 + 2 * 1
    assert torch.equal(batch, torch.tensor([0, 0, 0, 1, 1]))
    assert not torch.any(ei[0] == ei[1])


@pytest.mark.parametrize("conv", ["gcn", "gin"])
def test_gcn_gin_collapse_on_complete_graphs(conv, pool, scaler):
    torch.manual_seed(0)
    m = LegacyGAE(in_dim=3, hidden_channels=16, latent_channels=8, dropout=0.0, conv_type=conv)
    b = make_batch(pool, np.arange(8), scaler)
    spread, z, xhat, g = _node_spread(m, b)
    assert spread < 1e-3 * (1 + float(z.abs().max())), spread
    # consequently every node of a graph is reconstructed with the same vector
    for gi in torch.unique(g):
        xi = xhat[g == gi]
        assert float((xi - xi[0]).abs().max()) < 1e-3 * (1 + float(xi.abs().max()))


def test_sage_keeps_node_identity(pool, scaler):
    torch.manual_seed(0)
    m = LegacyGAE(in_dim=3, hidden_channels=16, latent_channels=8, dropout=0.0, conv_type="sage")
    b = make_batch(pool, np.arange(8), scaler)
    spread, *_ = _node_spread(m, b)
    assert spread > 1e-2


def test_legacy_loss_runs(pool, scaler):
    m = LegacyGAE(in_dim=3, hidden_channels=16, latent_channels=8, conv_type="gat", gat_heads=4)
    b = make_batch(pool, np.arange(8), scaler)
    out = m.losses(b)
    assert out["score"].shape == (8,) and torch.all(torch.isfinite(out["score"]))
