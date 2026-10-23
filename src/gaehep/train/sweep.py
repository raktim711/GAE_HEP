"""Hyper-parameter sweep under the development protocol, followed by seed confirmation and test evaluation.

Phase 1  grid over (latent size, VAE beta, encoder width, jets per epoch) with one seed: every point is a dev
         run (validation splits only, ``gaehep dev`` budget otherwise);
Phase 2  the ``top_k`` grid points by dev AUC (SR, ``prod`` aggregation of the main score) get more seeds;
Phase 3  the configuration with the best seed-averaged dev AUC gets ``final_seeds`` seeds in total, every
         seed is scored on the *test* splits (``promote``) and the seeds are combined in a rank ensemble.

The 3-prong signal is never looked at.  A wall-clock budget stops launching new runs in time; everything is
idempotent (finished dev runs are reused), so a sweep can be resumed by calling it again.  Results:
``results/dev/<prefix>_*`` (dev), ``results/runs/<prefix>_best_s*`` (test) and
``results/dev/<prefix>_sweep.json`` / ``.md`` (ranking).
"""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gaehep.config import load_json, results_dir, run_metadata, save_json
from gaehep.train.dev import DEV_TRAIN, dev_dir, dev_run, promote
from gaehep.train.run import Data

DEFAULT_GRID: dict[str, list] = {
    "latent_dim": [3, 4, 6, 8],
    "beta": [0.0, 1e-3, 1e-2],
    "width": [48, 128],
    "jets_per_epoch": [60_000, 120_000, 300_000],
}


def _fmt(v: float) -> str:
    return f"{v:g}".replace("-0", "-").replace("+", "")


def point_name(prefix: str, p: dict[str, Any]) -> str:
    return f"{prefix}_l{p['latent_dim']}_b{_fmt(p['beta'])}_w{p['width']}_j{p['jets_per_epoch'] // 1000}k"


def point_config(base: dict[str, Any], p: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict:
    """Base config + dev budget + grid point (+ fixed overrides, e.g. the device)."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("train", {}).update(DEV_TRAIN)
    ma = cfg.setdefault("model_args", {})
    ma["latent_dim"] = int(p["latent_dim"])
    ma["variational"] = bool(p["beta"] > 0)
    ma["beta"] = float(p["beta"])
    ma["hidden"] = [int(p["width"])] * len(ma.get("hidden", [48, 48, 48]))
    if p["width"] >= 96:
        ma["k"] = max(int(ma.get("k", 12)), 16)
    cfg["extra_scores"] = ["kl"] if p["beta"] > 0 else []
    cfg["train"]["jets_per_epoch"] = int(p["jets_per_epoch"])
    for key, val in (overrides or {}).items():
        node = cfg
        parts = key.split(".")
        for q in parts[:-1]:
            node = node.setdefault(q, {})
        node[parts[-1]] = val
    return cfg


def _dev_auc(name: str, agg: str = "prod") -> tuple[float, float] | None:
    path = dev_dir(name) / "dev_metrics.json"
    if not path.exists():
        return None
    m = load_json(path)["metrics"][f"SR/{agg}"]
    return float(m["auc"]), float(m["max_sic"])


def sweep(
    base_config: str,
    prefix: str = "sw",
    grid: dict[str, Sequence] | None = None,
    overrides: dict[str, Any] | None = None,
    budget_hours: float = 8.0,
    top_k: int = 3,
    n_seeds: int = 3,
    final_seeds: int = 5,
    phase1_share: float = 0.6,
    log=print,
) -> dict[str, Any]:
    t0 = time.time()
    hours = lambda: (time.time() - t0) / 3600  # noqa: E731
    base = yaml.safe_load(Path(base_config).read_text())
    grid = grid or DEFAULT_GRID
    points = [dict(zip(grid, vals, strict=True)) for vals in itertools.product(*grid.values())]
    # random order (fixed seed), so that a time cut in phase 1 leaves an unbiased subset of the grid
    points = [points[i] for i in np.random.default_rng(2026).permutation(len(points))]
    data_cache: dict[int, Data] = {}

    def run_point(p: dict[str, Any], seed: int) -> tuple[float, float] | None:
        name = point_name(prefix, p) + (f"_s{seed}" if seed != 1 else "")
        if _dev_auc(name) is None:
            cfg = point_config(base, p, overrides)
            cfg["train"]["seed"] = seed
            n_const = int(cfg.get("data", {}).get("n_const", 50))
            data = data_cache.setdefault(n_const, Data(n_const=n_const))
            t1 = time.time()
            try:
                dev_run(cfg, name, log=lambda *a: None, data=data)
            except Exception as exc:  # one failing point (e.g. out of memory) must not stop the sweep
                log(f"[sweep] {name} failed: {exc!r}")
                return None
            r = _dev_auc(name)
            log(f"[sweep] {name}: dev AUC {r[0]:.3f} max SIC {r[1]:.2f} ({(time.time() - t1) / 60:.1f} min)")
        return _dev_auc(name)

    # ---- phase 1: grid, one seed
    for p in points:
        if hours() > phase1_share * budget_hours:
            log(f"[sweep] phase-1 time budget reached after {hours():.1f} h")
            break
        run_point(p, 1)
    scored = [(p, _dev_auc(point_name(prefix, p))) for p in points]
    scored = [(p, r) for p, r in scored if r is not None]
    scored.sort(key=lambda pr: -pr[1][0])

    # ---- phase 2: more seeds for the best grid points
    seeds_done: dict[str, list[tuple[float, float]]] = {}
    for p, _ in scored[:top_k]:
        res = []
        for s in range(1, n_seeds + 1):
            if s > 1 and hours() > 0.85 * budget_hours:
                break
            r = run_point(p, s)
            if r is not None:
                res.append(r)
        seeds_done[point_name(prefix, p)] = res
    ranked = sorted(
        (
            (name, float(np.mean([a for a, _ in res])), float(np.std([a for a, _ in res])))
            for name, res in seeds_done.items()
            if res
        ),
        key=lambda t: -t[1],
    )
    best_point = next((p for p, _ in scored if ranked and point_name(prefix, p) == ranked[0][0]), None)

    # ---- phase 3: final seeds of the best configuration, scored on the test splits
    promoted = []
    if best_point is not None:
        for s in range(1, final_seeds + 1):
            if s > n_seeds and hours() > 0.9 * budget_hours:
                break
            if run_point(best_point, s) is None:
                continue
            dev_name = point_name(prefix, best_point) + (f"_s{s}" if s != 1 else "")
            run_name = f"{prefix}_best_s{s}"
            if not (results_dir() / "runs" / run_name / "metrics.json").exists():
                promote(dev_name, run_name, log=lambda *a: None)
            promoted.append(run_name)
        if len(promoted) > 1:
            from gaehep.train.run import ensemble_variant

            ensemble_variant(promoted, f"{prefix}_best_seeds")

    summary = {
        "base_config": base_config,
        "grid": {k: list(v) for k, v in grid.items()},
        "overrides": overrides or {},
        "phase1": [
            {"name": point_name(prefix, p), "point": p, "auc": r[0], "max_sic": r[1]} for p, r in scored
        ],
        "seeds": [{"name": n, "auc_mean": m, "auc_std": sd, "n": len(seeds_done[n])} for n, m, sd in ranked],
        "best": None if best_point is None else {"point": best_point, "name": point_name(prefix, best_point)},
        "promoted": promoted,
        "hours": hours(),
        **run_metadata(),
    }
    save_json(summary, results_dir() / "dev" / f"{prefix}_sweep.json")
    lines = [
        f"# Sweep `{prefix}` (dev protocol; SR AUC / max SIC of the main score, prod aggregation)",
        "",
        "| rank | grid point | dev AUC | max SIC |",
        "|---:|---|---:|---:|",
    ]
    lines += [
        f"| {i + 1} | {d['name']} | {d['auc']:.3f} | {d['max_sic']:.2f} |"
        for i, d in enumerate(summary["phase1"])
    ]
    lines += ["", "| seed-confirmed point | mean dev AUC | std | seeds |", "|---|---:|---:|---:|"]
    lines += [
        f"| {d['name']} | {d['auc_mean']:.3f} | {d['auc_std']:.3f} | {d['n']} |" for d in summary["seeds"]
    ]
    (results_dir() / "dev" / f"{prefix}_sweep.md").write_text("\n".join(lines) + "\n")
    log(f"[sweep] done in {hours():.1f} h; best: {summary['best']}")
    return summary
