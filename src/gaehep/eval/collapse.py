"""Quantify the node-embedding collapse of the legacy (AIMS) GAE on real LHCO jets.

For each convolution type the legacy encoder (complete graph, raw (pT, eta, phi) node features) embeds the
constituents of real background jets.  The *within-jet variance share*

    sum_jets sum_nodes |z_i - mean_jet(z)|^2  /  sum_all_nodes |z_i - mean_all(z)|^2

is 0 when all nodes of a jet get the same embedding (the decoder can then only output one vector per
jet) and close to 1 when embeddings vary mostly between nodes of the same jet.  It is computed at
initialisation (several seeds) and, if given, for trained dev runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from gaehep.config import results_dir, run_metadata, save_json


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
    out: dict[str, Any] = {"n_jets": len(pool), "init": {}, "trained": {}}
    base = yaml.safe_load((Path(__file__).parents[3] / "configs" / "legacy_gcn.yaml").read_text())[
        "model_args"
    ]
    for conv in ("gcn", "gin", "gat", "sage"):
        shares = []
        for s in seeds:
            torch.manual_seed(s)
            args = {**base, "conv_type": conv, "dropout": 0.0}
            if conv == "sage":
                args["sage_aggr"] = "sum"
            m = build_model("legacy_gae", **args).eval()
            _, z, _, g = m.node_outputs(b)
            shares.append(within_jet_variance_share(z, g))
        out["init"][conv] = shares
        log(
            f"[collapse] {conv:4s} at initialisation: within-jet variance share {np.round(shares, 6).tolist()}"
        )
    for run in dev_runs:
        d = results_dir() / "dev" / run
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        m = build_model(cfg["model"], **cfg["model_args"])
        m.load_state_dict(torch.load(d / "model.pt", weights_only=True))
        _, z, _, g = m.eval().node_outputs(b)
        out["trained"][run] = within_jet_variance_share(z, g)
        log(f"[collapse] {run}: within-jet variance share {out['trained'][run]:.6f}")
    save_json({**out, **run_metadata()}, results_dir() / "legacy_collapse.json")
    return out
