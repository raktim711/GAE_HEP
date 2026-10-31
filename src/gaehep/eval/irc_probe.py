"""How much do a saved model's anomaly scores move under collinear splittings, soft emissions and re-ordering?

``gaehep irc-probe <run>`` reads ``<results>/runs/<run>`` (``config.yaml`` and ``model.pt``; ``<results>`` =
GAEHEP_RESULTS, default <repo>/results; the jets are read from GAEHEP_DATA) and writes ``irc_probe.json`` into
the run directory.  Jets are 8000 background-validation and 8000 2-prong signal-validation jets of the
signal region.  For each perturbation it reports the jet-level AUC (background vs signal jets) after the
perturbation, the Spearman rank correlation of the background scores before / after, and the percentage of jets
whose membership of the top 1 % of background scores flips.

Perturbations (those of the former ``scripts/irc_probe.py``, which this module replaces, plus the last one):

* ``split top-1`` / ``split top-5`` / ``split all``: the hardest 1 / 5 / all constituents of every jet are split
  into two collinear halves;
* ``10 soft (0.2% pT each)`` / ``10 soft (1% pT each)``: ten soft constituents, each with that fraction of the jet
  pT, at random positions within R = 0.8 of the jet axis;
* ``random re-order``: the stored constituents (padding slots included) are put in a random order.  Every model
  of the study is permutation invariant, so this must leave the scores unchanged up to floating-point noise; the
  entry also holds ``max_abs_diff`` of the background scores.

``knn_ties`` counts, for the k of the model's k-NN graph (``model_args.k``, default 8) and the same jets, the
nodes whose k-th and (k+1)-th nearest neighbour in (deta, dphi) are at exactly the same distance (so the
neighbour set depends on tie-breaking) and the jets that have at least one such node.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from gaehep.config import SR_MJJ, results_dir, run_metadata, save_json
from gaehep.data.dataset import JetPool, Scaler, make_batch
from gaehep.models import build_model
from gaehep.train.loop import score_pool
from gaehep.train.run import Data

SR = SR_MJJ
N_JETS_EVENTS = 8000  # events per class
DEFAULT_K = 8  # the study's k-NN graph (protocol, section 3)
REORDER_SEED = 20201


def repl(pool: JetPool, const: np.ndarray) -> JetPool:
    return JetPool(
        **{**{k: getattr(pool, k) for k in pool.__dataclass_fields__}, "const": const.astype(np.float32)}
    )


def split_top(pool: JetPool, k: int | None) -> JetPool:
    """Split the k hardest (or all, k=None) constituents of every jet into two equal collinear halves."""
    n = pool.const.shape[1]
    kk = n if k is None else k
    out = np.zeros((len(pool), n + kk, 3), dtype=np.float32)
    out[:, :n] = pool.const
    out[:, n : n + kk] = pool.const[:, :kk]
    out[:, :kk, 0] *= 0.5
    out[:, n : n + kk, 0] *= 0.5
    return repl(pool, out)


def add_soft(pool: JetPool, n_soft: int, frac: float, seed: int = 0) -> JetPool:
    rng = np.random.default_rng(seed)
    n = pool.const.shape[1]
    out = np.zeros((len(pool), n + n_soft, 3), dtype=np.float32)
    out[:, :n] = pool.const
    pt_j = pool.const[..., 0].sum(1)
    r = 0.8 * np.sqrt(rng.uniform(size=(len(pool), n_soft)))
    a = rng.uniform(0, 2 * np.pi, size=(len(pool), n_soft))
    out[:, n:, 0] = (frac * pt_j)[:, None]
    out[:, n:, 1] = pool.eta[:, None] + r * np.cos(a)
    out[:, n:, 2] = pool.phi[:, None] + r * np.sin(a)
    # every stored constituent must be real: place the soft ones after the real ones (order is irrelevant)
    return repl(pool, out)


def reorder(pool: JetPool, seed: int = REORDER_SEED) -> JetPool:
    """Random re-ordering of the stored constituents of every jet (padding slots included)."""
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random(pool.const.shape[:2]), axis=1)
    return repl(pool, np.take_along_axis(pool.const, perm[..., None], axis=1))


PERTURBATIONS = {
    "split top-1": lambda p: split_top(p, 1),
    "split top-5": lambda p: split_top(p, 5),
    "split all": lambda p: split_top(p, None),
    "10 soft (0.2% pT each)": lambda p: add_soft(p, 10, 0.002),
    "10 soft (1% pT each)": lambda p: add_soft(p, 10, 0.01),
    "random re-order": reorder,
}


@torch.no_grad()
def knn_tie_counts(pool: JetPool, scaler: Scaler, k: int, batch_size: int = 256) -> dict[str, float | int]:
    """Nodes whose k-th and (k+1)-th nearest neighbours (excluding the node, among the real nodes, in the
    (deta, dphi) plane of the jet frame, distances as in ``graphs.ops.masked_knn``) are exactly equidistant,
    and jets with at least one such node.  Only nodes with at least k + 1 other real nodes can tie."""
    n_nodes = n_tied = n_jets_tied = 0
    for s in range(0, len(pool), batch_size):
        b = make_batch(pool, np.arange(s, min(s + batch_size, len(pool))), scaler)
        n = b.coords.shape[1]
        if n < k + 2:
            n_nodes += int(b.mask.sum())
            continue
        d = torch.cdist(b.coords, b.coords, compute_mode="donot_use_mm_for_euclid_dist")
        d = d.masked_fill(~b.mask[:, None, :], float("inf")).masked_fill(
            torch.eye(n, dtype=torch.bool), float("inf")
        )
        v = torch.topk(d, k + 1, dim=-1, largest=False, sorted=True).values
        tie = (v[..., k - 1] == v[..., k]) & torch.isfinite(v[..., k]) & b.mask
        n_nodes += int(b.mask.sum())
        n_tied += int(tie.sum())
        n_jets_tied += int(tie.any(1).sum())
    return {
        "k": k,
        "n_jets": len(pool),
        "n_nodes": n_nodes,
        "nodes_with_tie": n_tied,
        "jets_with_tie": n_jets_tied,
        "jet_fraction": n_jets_tied / max(len(pool), 1),
        "node_fraction": n_tied / max(n_nodes, 1),
    }


def probe(
    name: str,
    data: Data | None = None,
    n_events: int = N_JETS_EVENTS,
    k: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Run the perturbations on run ``name`` and write ``irc_probe.json`` into its directory."""
    run_dir = results_dir() / "runs" / name
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    data = data or Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    scaler = data.scaler()
    bkg = data.pool("bkg_val", n_events, mjj=SR)
    sig = data.pool("sig_val", n_events, mjj=SR)
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True, map_location="cpu"))
    model.eval()

    def sc(p: JetPool) -> np.ndarray:
        return score_pool(model, p, scaler, batch_size=128)["score"]

    s0b, s0s = sc(bkg), sc(sig)
    y = np.r_[np.zeros(len(s0b)), np.ones(len(s0s))]
    base = float(roc_auc_score(y, np.r_[s0b, s0s]))
    res: dict[str, Any] = {"auc": base}
    log(f"{name:14s} unperturbed jet AUC {base:.3f}")
    for label, fn in PERTURBATIONS.items():
        sb, ss = sc(fn(bkg)), sc(fn(sig))
        auc = float(roc_auc_score(y, np.r_[sb, ss]))
        rho = float(spearmanr(s0b, sb).statistic)
        thr0, thr1 = np.quantile(s0b, 0.99), np.quantile(sb, 0.99)
        flip = float(((s0b > thr0) != (sb > thr1)).mean() * 100)
        res[label] = {"auc": auc, "rho": rho, "flip_pct_at_1pct": flip}
        if label == "random re-order":
            res[label]["max_abs_diff"] = float(np.max(np.abs(sb.astype(np.float64) - s0b)))
        log(
            f"   {label:24s} AUC {auc:.3f}  (d = {auc - base:+.3f})  Spearman(bkg) {rho:.4f}  "
            f"top-1% membership flips {flip:.2f}% of jets"
        )
    k = int(k or cfg.get("model_args", {}).get("k", DEFAULT_K))
    res["knn_ties"] = {"bkg": knn_tie_counts(bkg, scaler, k), "sig": knn_tie_counts(sig, scaler, k)}
    log(
        f"   k-NN (k = {k}) ties: {res['knn_ties']['bkg']['jets_with_tie']} of {len(bkg)} background jets, "
        f"{res['knn_ties']['sig']['jets_with_tie']} of {len(sig)} signal jets"
    )
    out = {
        "name": name,
        "n_bkg_jets": len(bkg),
        "n_sig_jets": len(sig),
        "sr": list(SR),
        **res,
        **run_metadata(),
    }
    save_json(out, run_dir / "irc_probe.json")
    return out


def main(names: list[str], k: int | None = None) -> None:
    torch.set_num_threads(1)
    for name in names:
        probe(name, k=k)
