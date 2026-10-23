"""Quantify the node-embedding collapse of the legacy (AIMS) GAE on real LHCO jets.

For each convolution type the legacy encoder (complete graph, raw (pT, eta, phi) node features) embeds the
constituents of real background jets.  The *within-jet variance share*

    sum_jets sum_nodes |z_i - mean_jet(z)|^2  /  sum_all_nodes |z_i - mean_all(z)|^2

is 0 when all nodes of a jet get the same embedding (the decoder can then only output one vector per
jet) and close to 1 when embeddings vary mostly between nodes of the same jet.  It is recorded after every
convolution layer (pre-activation), at initialisation (several seeds) and, if given, for trained dev runs.
On a complete graph message passing averages over the whole jet, so the share shrinks with depth
("over-smoothing"); for GCN and GIN (eps = 0) it is exactly zero from the first layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from gaehep.config import results_dir, run_metadata, save_json

LAYERS = ("conv1", "conv_mid", "conv2")


def within_jet_variance_share(z: torch.Tensor, graph: torch.Tensor) -> float:
    z = z.double()
    total = float(((z - z.mean(0)) ** 2).sum())
    if total == 0:
        return 0.0
    n_g = int(graph.max()) + 1
    cnt = torch.bincount(graph, minlength=n_g).clamp(min=1).double()
    mean_g = torch.zeros(n_g, z.shape[1], dtype=z.dtype).index_add_(0, graph, z) / cnt[:, None]
    within = float(((z - mean_g[graph]) ** 2).sum())
    return within / total


@torch.no_grad()
def layer_shares(model, batch) -> list[float]:
    """Within-jet variance share of the raw input and after each convolution (pre-activation)."""
    from gaehep.models.legacy_gae import complete_graph

    flat_real, edge_index, graph = complete_graph(batch.mask)
    h = batch.raw.reshape(-1, batch.raw.shape[-1])[flat_real]
    shares = [within_jet_variance_share(h, graph)]
    for name in LAYERS:
        pre = getattr(model, name)(h, edge_index)
        if model.use_batchnorm:
            pre = getattr(model, {"conv1": "bn1", "conv_mid": "bn_mid", "conv2": "bn2"}[name])(pre)
        shares.append(within_jet_variance_share(pre, graph))
        h = F.relu(pre)
    return shares


@torch.no_grad()
def collapse_study(
    n_jets: int = 512, seeds: tuple[int, ...] = (0, 1, 2), dev_runs: tuple[str, ...] = (), log=print
) -> dict[str, Any]:
    from gaehep.data.dataset import make_batch
    from gaehep.models import build_model
    from gaehep.train.run import Data

    data = Data(n_const=30)
    scaler = data.scaler()
    pool = data.pool("bkg_val", n_jets // 2)
    b = make_batch(pool, np.arange(len(pool)), scaler)
    out: dict[str, Any] = {"n_jets": len(pool), "layers": ["input", *LAYERS], "init": {}, "trained": {}}
    base = yaml.safe_load((Path(__file__).parents[3] / "configs" / "legacy_gcn.yaml").read_text())
    for conv in ("gcn", "gin", "gat", "sage"):
        per_seed = []
        for s in seeds:
            torch.manual_seed(s)
            args = {**base["model_args"], "conv_type": conv, "dropout": 0.0}
            if conv == "sage":
                args["sage_aggr"] = "sum"
            per_seed.append(layer_shares(build_model("legacy_gae", **args).eval(), b))
        out["init"][conv] = per_seed
        med = np.median(np.asarray(per_seed), axis=0)
        log(
            f"[collapse] {conv:4s} init, within-jet variance share input/conv1/conv_mid/conv2: "
            + " / ".join(f"{v:.2g}" for v in med)
        )
    for run in dev_runs:
        d = results_dir() / "dev" / run
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        m = build_model(cfg["model"], **cfg["model_args"])
        m.load_state_dict(torch.load(d / "model.pt", weights_only=True, map_location="cpu"))
        out["trained"][run] = layer_shares(m.eval(), b)
        log(f"[collapse] {run} trained: " + " / ".join(f"{v:.2g}" for v in out["trained"][run]))
    save_json({**out, **run_metadata()}, results_dir() / "legacy_collapse.json")
    return out
