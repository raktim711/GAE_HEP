"""Faithful port of the AIMS HLT graph autoencoder ("M0"), used as a baseline and as a cautionary example.

Ported from ``GraphAutoEncoder`` in ``ModelTesting_after_processing.ipynb`` (cell 50) of
github.com/raktim711/AIMS-project---Anomaly-Detection: three PyG convolutions (in -> hidden -> hidden/2 ->
latent), each followed by (BatchNorm), ReLU and dropout - including on the latent - and a node-wise MLP
decoder; MSE on raw node features; score = mean node error per graph.  Graphs are *complete* (all ordered
pairs i != j), node features are the raw (pT, eta, phi) of the constituents (no per-type one-hot: LHCO
particles carry no type).

Known property (tested in ``tests/test_legacy_collapse.py``): on a complete graph, ``GCNConv`` (with its
default self loops and symmetric normalisation every weight is 1/N) and ``GINConv`` with eps = 0 (the MLP
sees x_i + sum_{j != i} x_j = sum_j x_j) return *identical* outputs for all nodes of a graph.  The decoder
therefore reconstructs every node of a graph with the same vector, so the "anomaly score" is essentially
the spread of raw node features within the graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gaehep.data.dataset import JetBatch
from gaehep.models.base import JetModel


def complete_graph(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Directed complete graphs (no self loops) over the real nodes of a padded batch.

    Returns ``flat_real`` (positions of real nodes in the flattened B*N grid), ``edge_index`` (2, E) over the
    compacted real nodes (row 0 = source j, row 1 = target i) and ``batch`` (num_real,) graph ids.
    """
    b, n = mask.shape
    flat_real = mask.reshape(-1).nonzero().squeeze(1)
    node_id = torch.full((b * n,), -1, dtype=torch.long, device=mask.device)
    node_id[flat_real] = torch.arange(len(flat_real), device=mask.device)
    eye = torch.eye(n, dtype=torch.bool, device=mask.device)
    pair = mask[:, :, None] & mask[:, None, :] & ~eye[None]  # [g, i (target), j (source)]
    g, i, j = pair.nonzero(as_tuple=True)
    edge_index = torch.stack([node_id[g * n + j], node_id[g * n + i]])
    return flat_real, edge_index, flat_real // n


class LegacyGAE(JetModel):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 16,
        dropout: float = 0.1,
        conv_type: str = "gcn",
        sage_aggr: str = "mean",
        use_batchnorm: bool = False,
        gat_heads: int = 4,
    ):
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

        self.conv_type, self.dropout, self.use_batchnorm = conv_type.lower(), dropout, use_batchnorm
        mid = max(1, hidden_channels // 2)
        h, lat = hidden_channels, latent_channels
        if self.conv_type == "gcn":
            self.conv1, self.conv_mid, self.conv2 = GCNConv(in_dim, h), GCNConv(h, mid), GCNConv(mid, lat)
        elif self.conv_type == "sage":
            self.conv1 = SAGEConv(in_dim, h, aggr=sage_aggr)
            self.conv_mid = SAGEConv(h, mid, aggr=sage_aggr)
            self.conv2 = SAGEConv(mid, lat, aggr=sage_aggr)
        elif self.conv_type == "gin":
            seq = nn.Sequential
            self.conv1 = GINConv(seq(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h)))
            self.conv_mid = GINConv(seq(nn.Linear(h, mid), nn.ReLU(), nn.Linear(mid, mid)))
            self.conv2 = GINConv(seq(nn.Linear(mid, lat), nn.ReLU(), nn.Linear(lat, lat)))
        elif self.conv_type == "gat":
            if h % gat_heads or mid % gat_heads:
                raise ValueError("hidden and hidden/2 must be divisible by gat_heads")
            self.conv1 = GATConv(in_dim, h // gat_heads, heads=gat_heads, concat=True)
            self.conv_mid = GATConv(h, mid // gat_heads, heads=gat_heads, concat=True)
            self.conv2 = GATConv(mid, lat, heads=gat_heads, concat=False)
        else:
            raise ValueError(f"unknown conv_type {conv_type}")
        if use_batchnorm:
            self.bn1, self.bn_mid, self.bn2 = nn.BatchNorm1d(h), nn.BatchNorm1d(mid), nn.BatchNorm1d(lat)
        self.dec_lin1 = nn.Linear(lat, h)
        self.dec_lin2 = nn.Linear(h, in_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, bn in (("conv1", "bn1"), ("conv_mid", "bn_mid"), ("conv2", "bn2")):
            x = getattr(self, conv)(x, edge_index)
            if self.use_batchnorm:
                x = getattr(self, bn)(x)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.dropout(F.relu(self.dec_lin1(z)), p=self.dropout, training=self.training)
        return self.dec_lin2(h)

    def node_outputs(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_real, edge_index, batch = complete_graph(b.mask)
        x = b.raw.reshape(-1, b.raw.shape[-1])[flat_real]
        z = self.encode(x, edge_index)
        return x, z, self.decode(z), batch

    def losses(self, b: JetBatch) -> dict[str, torch.Tensor]:
        x, _, xhat, batch = self.node_outputs(b)
        per_node = ((xhat - x) ** 2).mean(-1)
        sums = torch.zeros(len(b), dtype=per_node.dtype, device=per_node.device).index_add_(
            0, batch, per_node
        )
        counts = torch.bincount(batch, minlength=len(b)).clamp(min=1).to(per_node.dtype)
        per_graph = sums / counts
        return {"total": per_graph, "score": per_graph}
