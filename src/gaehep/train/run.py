"""Experiment orchestration: config -> data pools -> train -> score -> evaluate (+ post-hoc variants).

A run lives in ``results/runs/<name>/`` with ``config.yaml``, ``history.json``, ``metrics.json``,
figures, and git-ignored ``model.pt`` / ``checkpoint.pt`` / ``scores.npz`` / ``latents.npz``.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from gaehep.config import load_json, processed_dir, results_dir, run_metadata, save_json
from gaehep.data.dataset import JetPool, JetStore, Scaler, fit_scaler
from gaehep.data.splits import load_splits
from gaehep.eval.evaluate import TEST_KEYS, evaluate_scores, make_figures, write_metrics
from gaehep.models import build_model
from gaehep.models.latent_flow import LatentFlow
from gaehep.train.loop import TrainConfig, fit, score_pool

SPLIT_SAMPLE = {
    "bkg_train": "rnd",
    "bkg_val": "rnd",
    "bkg_test": "rnd",
    "sig_train": "rnd",
    "sig_val": "rnd",
    "sig_test": "rnd",
    "sig3_test": "qqq",
}
SCORE_SPLITS = ("bkg_val", "bkg_test", "sig_test", "sig3_test")
TABLE_KEYS = ("mjj", "m", "pt", "n_const", "tau1", "tau2", "tau3", "label")


def concat_pools(pools: list[JetPool]) -> JetPool:
    return JetPool(
        **{k: np.concatenate([getattr(p, k) for p in pools]) for k in JetPool.__dataclass_fields__}
    )


class Data:
    """Lazy access to split pools / tables / scaler for a given number of nodes per jet."""

    def __init__(self, n_const: int, root: Path | None = None):
        self.n_const = n_const
        self.root = Path(root) if root else processed_dir()
        self.splits = load_splits(self.root)
        self.stores: dict[str, JetStore] = {}
        self._pools: dict[str, JetPool] = {}

    def store(self, sample: str) -> JetStore:
        if sample not in self.stores:
            self.stores[sample] = JetStore(sample, self.root)
        return self.stores[sample]

    def has(self, split: str) -> bool:
        return split in self.splits and (self.root / SPLIT_SAMPLE[split] / "events.npz").exists()

    def events(self, split: str, max_events: int | None = None, seed: int = 0) -> np.ndarray:
        ev = self.splits[split]
        if max_events is not None and len(ev) > max_events:
            ev = np.sort(np.random.default_rng(seed).choice(ev, max_events, replace=False))
        return ev

    def pool(self, split: str, max_events: int | None = None) -> JetPool:
        key = f"{split}:{max_events}"
        if key not in self._pools:
            self._pools[key] = self.store(SPLIT_SAMPLE[split]).pool(
                self.events(split, max_events), self.n_const
            )
        return self._pools[key]

    def table(self, split: str) -> dict[str, np.ndarray]:
        st = self.store(SPLIT_SAMPLE[split])
        ev = self.events(split)
        return {k: st[k][ev] for k in TABLE_KEYS if k in st.table}

    def scaler(self) -> Scaler:
        path = self.root / f"scaler_n{self.n_const}.json"
        if path.exists():
            return Scaler.from_json(path)
        sc = fit_scaler(self.pool("bkg_train"))
        sc.to_json(path)
        return sc


def _run_dir(name: str) -> Path:
    return results_dir() / "runs" / name


def run(
    cfg: dict[str, Any],
    name: str,
    stages: tuple[str, ...] = ("train", "score", "evaluate"),
    data: Data | None = None,
    log=print,
) -> dict[str, Any]:
    run_dir = _run_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    dcfg = cfg.get("data", {})
    data = data or Data(n_const=int(dcfg.get("n_const", 50)))
    scaler = data.scaler()
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[{name}] model {cfg['model']} with {n_params:,} parameters, {data.n_const} nodes/jet")
    tcfg = TrainConfig.from_dict(cfg.get("train"))
    max_train = dcfg.get("max_train_events")
    summary: dict[str, Any] = {"status": "skipped"}

    if "train" in stages:
        train_pool = data.pool("bkg_train", max_train)
        val_pool = data.pool("bkg_val")
        if model.supervised:
            train_pool = concat_pools([train_pool, data.pool("sig_train")])
            val_pool = concat_pools([val_pool, data.pool("sig_val")])
        summary = fit(model, train_pool, val_pool, scaler, tcfg, run_dir, log=log)
        if summary["status"] != "done":
            return summary
    else:
        model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
        if (run_dir / "history.json").exists():
            summary = load_json(run_dir / "history.json")

    if "score" in stages:
        torch.set_num_threads(tcfg.threads)
        scores, latents = {}, {}
        lat_splits = [("bkg_train", int(dcfg.get("latent_train_events", 100_000)))]
        for split, cap in [(s, None) for s in SCORE_SPLITS if data.has(s)] + lat_splits:
            out = score_pool(model, data.pool(split, cap), scaler, with_embed=True)
            if split != "bkg_train":
                scores[split] = out["score"].reshape(-1, 2).astype(np.float32)
            if "z" in out:
                latents[split] = out["z"].reshape(-1, 2, out["z"].shape[-1]).astype(np.float32)
            log(f"[{name}] scored {split}: {len(out['score'])} jets")
        np.savez(run_dir / "scores.npz", **scores)
        if latents:
            np.savez(run_dir / "latents.npz", **latents)

    if "evaluate" in stages:
        evaluate_run(name, data=data, cfg=cfg, history=summary.get("history"))
    return summary


def _tables(data: Data) -> dict[str, dict[str, np.ndarray]]:
    return {k: data.table(k) for k in TEST_KEYS if data.has(k)}


def evaluate_run(
    name: str,
    data: Data | None = None,
    cfg: dict | None = None,
    history: list | None = None,
    scores: dict[str, np.ndarray] | None = None,
    n_boot: int = 30,
) -> dict[str, Any]:
    run_dir = _run_dir(name)
    if scores is None:
        with np.load(run_dir / "scores.npz") as f:
            scores = {k: f[k] for k in f.files}
    if cfg is None and (run_dir / "config.yaml").exists():
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    data = data or Data(n_const=int((cfg or {}).get("data", {}).get("n_const", 50)))
    tables = _tables(data)
    metrics = evaluate_scores({k: scores[k] for k in TEST_KEYS if k in scores}, tables, n_boot=n_boot)
    write_metrics(metrics, run_dir / "metrics.json", meta={"name": name, "config": cfg, **run_metadata()})
    make_figures(scores, tables, run_dir / "figures", name, history=history)
    return metrics


# ------------------------------------------------------------------------------- post-hoc variants
def latent_flow_variant(name: str, data: Data | None = None, seed: int = 0, log=print) -> dict[str, Any]:
    """Fit a flow on background-training latents of run ``name``; score = -log p(z) per jet."""
    src = _run_dir(name)
    with np.load(src / "latents.npz") as f:
        lat = {k: f[k] for k in f.files}
    d = lat["bkg_train"].shape[-1]
    flow = LatentFlow(d, seed=seed).fit(
        lat["bkg_train"].reshape(-1, d), lat["bkg_val"].reshape(-1, d), seed=seed
    )
    scores = {
        k: flow.nll(v.reshape(-1, d)).reshape(-1, 2).astype(np.float32)
        for k, v in lat.items()
        if k != "bkg_train"
    }
    out_name = f"{name}+flow"
    out_dir = _run_dir(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scores.npz", **scores)
    save_json({"source_run": name, "flow_history": flow.history}, out_dir / "history.json")
    log(f"[{out_name}] flow fitted ({len(flow.history['val_nll'])} epochs)")
    return evaluate_run(out_name, data=data, cfg={"posthoc": "latent_flow", "source": name}, scores=scores)


def _ecdf_ref(ref: np.ndarray) -> np.ndarray:
    return np.sort(ref)


def _ecdf(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(ref_sorted, x, side="right") / max(len(ref_sorted), 1)


def quantile_normalised_variant(name: str, data: Data | None = None, n_bins: int = 20) -> dict[str, Any]:
    """Mass-decorrelate a run's per-jet scores: replace each score by its background-validation percentile
    within its jet-mass bin (bins = background-validation mass quantiles)."""
    src = _run_dir(name)
    with np.load(src / "scores.npz") as f:
        raw = {k: f[k] for k in f.files}
    cfg = yaml.safe_load((src / "config.yaml").read_text()) if (src / "config.yaml").exists() else {}
    data = data or Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    m_val = data.table("bkg_val")["m"].ravel()
    s_val = raw["bkg_val"].ravel()
    edges = np.quantile(m_val, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    refs = [_ecdf_ref(s_val[(m_val >= lo) & (m_val < hi)]) for lo, hi in itertools.pairwise(edges)]
    out: dict[str, np.ndarray] = {}
    for split, s in raw.items():
        m = data.table(split)["m"]
        b = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, n_bins - 1)
        q = np.empty_like(s, dtype=np.float64)
        for i in range(n_bins):
            sel = b == i
            q[sel] = _ecdf(refs[i], s[sel])
        out[split] = q.astype(np.float32)
    out_name = f"{name}+qn"
    out_dir = _run_dir(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scores.npz", **out)
    return evaluate_run(
        out_name, data=data, cfg={"posthoc": "mass_quantile_normalisation", "source": name}, scores=out
    )


def ensemble_variant(names: list[str], out_name: str, data: Data | None = None) -> dict[str, Any]:
    """Average of per-jet background-validation percentiles of several runs (rank ensemble)."""
    per = []
    for n in names:
        with np.load(_run_dir(n) / "scores.npz") as f:
            per.append({k: f[k] for k in f.files})
    out: dict[str, np.ndarray] = {}
    for split in per[0]:
        acc = np.zeros_like(per[0][split], dtype=np.float64)
        for p in per:
            acc += _ecdf(_ecdf_ref(p["bkg_val"].ravel()), p[split])
        out[split] = (acc / len(per)).astype(np.float32)
    out_dir = _run_dir(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scores.npz", **out)
    return evaluate_run(out_name, data=data, cfg={"posthoc": "rank_ensemble", "sources": names}, scores=out)


def baselines(data: Data | None = None, n_boot: int = 30) -> dict[str, Any]:
    """Single-observable scores (jet mass, multiplicity, -tau21, -tau32) evaluated like any model."""
    from gaehep.eval.evaluate import observable_scores

    data = data or Data(n_const=50)
    tables = _tables(data)
    per_split = {k: observable_scores(v) for k, v in tables.items()}
    results = {}
    for obs in per_split["bkg_test"]:
        name = f"baseline_{obs}"
        scores = {k: v[obs].astype(np.float32) for k, v in per_split.items()}
        _run_dir(name).mkdir(parents=True, exist_ok=True)
        np.savez(_run_dir(name) / "scores.npz", **scores)
        results[name] = evaluate_run(
            name, data=data, cfg={"baseline": obs}, scores=copy.deepcopy(scores), n_boot=n_boot
        )
    return results


def bumphunt_run(name: str, agg: str = "max", data: Data | None = None) -> list[dict[str, Any]]:
    """Bump hunt on the background test split with injected 2-prong test signal, using run ``name``'s scores."""
    from gaehep.eval import plots
    from gaehep.eval.bumphunt import run_bumphunt
    from gaehep.eval.evaluate import aggregate

    run_dir = _run_dir(name)
    with np.load(run_dir / "scores.npz") as f:
        scores = {k: f[k] for k in f.files}
    data = data or Data(n_const=50)
    tb, ts = data.table("bkg_test"), data.table("sig_test")
    rows = run_bumphunt(
        tb["mjj"], aggregate(scores["bkg_test"], agg), ts["mjj"], aggregate(scores["sig_test"], agg)
    )
    save_json({"name": name, "agg": agg, "rows": rows, **run_metadata()}, run_dir / f"bumphunt_{agg}.json")
    plots.plot_bumphunt(rows, run_dir / "figures" / f"bumphunt_{agg}.png", title=f"{name} [{agg}]")
    return rows
