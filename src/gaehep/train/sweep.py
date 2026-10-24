"""Hyper-parameter sweep under the development protocol, followed by seed confirmation and test evaluation.

Phase 1  grid over any config keys with one seed: every point is a dev run (validation splits only,
         ``gaehep dev`` budget otherwise).  Shorthand keys: ``latent_dim``, ``beta`` (also switches the VAE and
         its KL score on/off), ``width`` (all encoder layers), ``jets_per_epoch``; any other key is a dotted
         config path (``model_args.sigmas``, ``train.epochs``, ...).  A grid entry is a list of values or a
         dict {label: value} (e.g. named kernel-scale sets);
Phase 2  the ``top_k`` grid points by dev AUC (SR, ``prod`` aggregation) get more seeds;
Phase 3  the configuration with the best seed-averaged dev AUC gets ``final_seeds`` seeds in total, every
         seed is scored on the *test* splits (``promote``) and the seeds are combined in a rank ensemble.

``score="ens"`` ranks VAE points by the rank ensemble of their reconstruction and KL scores (``<run>+ens``;
points without a KL score are ranked by the reconstruction score) and builds the same ensemble on the test
splits.  The 3-prong signal is never looked at.  A wall-clock budget stops launching new runs in time;
everything is idempotent (finished dev runs are reused), so a sweep can be resumed by calling it again.
Results: ``results/dev/<prefix>_*`` (dev), ``results/runs/<prefix>_best_s*`` (test) and
``results/dev/<prefix>_sweep.json`` / ``.md`` (ranking).
"""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gaehep.config import load_json, results_dir, run_metadata, save_json
from gaehep.train.dev import DEV_TRAIN, dev_dir, dev_ensemble, dev_run, promote
from gaehep.train.run import Data

DEFAULT_GRID: dict[str, list] = {
    "latent_dim": [3, 4, 6, 8],
    "beta": [0.0, 1e-3, 1e-2],
    "width": [48, 128],
    "jets_per_epoch": [60_000, 120_000, 300_000],
}
# short labels of grid keys in run names
ALIAS = {
    "latent_dim": "l",
    "beta": "b",
    "width": "w",
    "jets_per_epoch": "j",
    "model_args.sigmas": "sig",
    "model_args.hidden": "h",
    "model_args.k": "k",
    "model_args.n_out": "o",
    "train.epochs": "e",
    "train.lr": "lr",
    "train.batch_size": "bs",
    "data.n_const": "n",
}


def _fmt(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt(x) for x in v)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    return f"{v:g}".replace("-0", "-").replace("+", "")


def _label(key: str, value: Any) -> str:
    if key == "jets_per_epoch":
        return f"{int(value) // 1000}k"
    return _fmt(value)


def levels(grid: Mapping[str, Any]) -> dict[str, list[tuple[str, Any]]]:
    """Every grid entry as (label, value) pairs; a dict entry maps labels to values."""
    out = {}
    for key, vals in grid.items():
        if isinstance(vals, Mapping):
            out[key] = [(str(lab), v) for lab, v in vals.items()]
        else:
            out[key] = [(_label(key, v), v) for v in vals]
    return out


def point_name(prefix: str, p: Mapping[str, Any], labels: Mapping[str, str] | None = None) -> str:
    labels = labels or {}
    return prefix + "".join(
        f"_{ALIAS.get(k, k.split('.')[-1])}{labels.get(k, _label(k, v))}" for k, v in p.items()
    )


def _set_dotted(cfg: dict[str, Any], key: str, val: Any) -> None:
    node = cfg
    parts = key.split(".")
    for q in parts[:-1]:
        node = node.setdefault(q, {})
    node[parts[-1]] = copy.deepcopy(val)


def point_config(base: dict[str, Any], p: Mapping[str, Any], overrides: dict[str, Any] | None = None) -> dict:
    """Base config + dev budget + grid point (+ fixed overrides, e.g. the device)."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("train", {}).update(DEV_TRAIN)
    ma = cfg.setdefault("model_args", {})
    for key, val in p.items():
        if key == "latent_dim":
            ma["latent_dim"] = int(val)
        elif key == "beta":
            ma["variational"] = bool(val > 0)
            ma["beta"] = float(val)
            cfg["extra_scores"] = ["kl"] if val > 0 else []
        elif key == "width":
            ma["hidden"] = [int(val)] * len(ma.get("hidden", [48, 48, 48]))
            if val >= 96:
                ma["k"] = max(int(ma.get("k", 12)), 16)
        elif key == "jets_per_epoch":
            cfg["train"]["jets_per_epoch"] = int(val)
        else:
            _set_dotted(cfg, key, val)
    for key, val in (overrides or {}).items():
        _set_dotted(cfg, key, val)
    return cfg


def _dev_auc(name: str, agg: str = "prod") -> tuple[float, float] | None:
    path = dev_dir(name) / "dev_metrics.json"
    if not path.exists():
        return None
    m = load_json(path)["metrics"][f"SR/{agg}"]
    return float(m["auc"]), float(m["max_sic"])


def _ranked_entry(name: str, score: str) -> str:
    """The dev entry that ranks a point: the run itself, or with ``score="ens"`` the rank ensemble of its
    reconstruction and KL scores (built on first use)."""
    if score == "ens" and (dev_dir(f"{name}+kl") / "dev_scores.npz").exists():
        if not (dev_dir(f"{name}+ens") / "dev_metrics.json").exists():
            dev_ensemble([name, f"{name}+kl"], f"{name}+ens", log=lambda *a: None)
        return f"{name}+ens"
    return name


def sweep(
    base_config: str,
    prefix: str = "sw",
    grid: Mapping[str, Sequence | Mapping] | None = None,
    overrides: dict[str, Any] | None = None,
    budget_hours: float = 8.0,
    top_k: int = 3,
    n_seeds: int = 3,
    final_seeds: int = 5,
    phase1_share: float = 0.6,
    score: str = "rec",
    log=print,
) -> dict[str, Any]:
    if score not in ("rec", "ens"):
        raise ValueError(f"score must be 'rec' or 'ens', not {score!r}")
    t0 = time.time()
    hours = lambda: (time.time() - t0) / 3600  # noqa: E731
    base = yaml.safe_load(Path(base_config).read_text())
    grid = grid or DEFAULT_GRID
    lv = levels(grid)
    points = [
        (
            {k: v for k, (_, v) in zip(lv, combo, strict=True)},
            {k: lab for k, (lab, _) in zip(lv, combo, strict=True)},
        )
        for combo in itertools.product(*lv.values())
    ]
    # random order (fixed seed), so that a time cut in phase 1 leaves an unbiased subset of the grid
    points = [points[i] for i in np.random.default_rng(2026).permutation(len(points))]
    names = {point_name(prefix, p, lab): (p, lab) for p, lab in points}
    if len(names) != len(points):
        raise ValueError("grid labels are not unique")
    data_cache: dict[int, Data] = {}

    def run_point(name: str, seed: int) -> tuple[float, float] | None:
        p, _ = names[name]
        run = name + (f"_s{seed}" if seed != 1 else "")
        if _dev_auc(run) is None:
            cfg = point_config(base, p, overrides)
            cfg["train"]["seed"] = seed
            n_const = int(cfg.get("data", {}).get("n_const", 50))
            data = data_cache.setdefault(n_const, Data(n_const=n_const))
            t1 = time.time()
            try:
                dev_run(cfg, run, log=lambda *a: None, data=data)
            except Exception as exc:  # one failing point (e.g. out of memory) must not stop the sweep
                log(f"[sweep] {run} failed: {exc!r}")
                return None
            if _dev_auc(run) is None:  # e.g. training paused by its time budget
                log(f"[sweep] {run} did not finish")
                return None
            r = _dev_auc(_ranked_entry(run, score))
            log(f"[sweep] {run}: dev AUC {r[0]:.3f} max SIC {r[1]:.2f} ({(time.time() - t1) / 60:.1f} min)")
        return _dev_auc(_ranked_entry(run, score))

    # ---- phase 1: grid, one seed
    for name in names:
        if hours() > phase1_share * budget_hours:
            log(f"[sweep] phase-1 time budget reached after {hours():.1f} h")
            break
        run_point(name, 1)
    scored = [(n, _dev_auc(_ranked_entry(n, score))) for n in names if _dev_auc(n) is not None]
    scored.sort(key=lambda nr: -nr[1][0])

    # ---- phase 2: more seeds for the best grid points
    seeds_done: dict[str, list[tuple[float, float]]] = {}
    for name, _ in scored[:top_k]:
        res = []
        for s in range(1, n_seeds + 1):
            if s > 1 and hours() > 0.85 * budget_hours:
                break
            r = run_point(name, s)
            if r is not None:
                res.append(r)
        seeds_done[name] = res
    ranked = sorted(
        (
            (name, float(np.mean([a for a, _ in res])), float(np.std([a for a, _ in res])))
            for name, res in seeds_done.items()
            if res
        ),
        key=lambda t: -t[1],
    )
    best = ranked[0][0] if ranked else None

    # ---- phase 3: final seeds of the best configuration, scored on the test splits
    promoted = []
    if best is not None:
        from gaehep.train.run import _run_dir, ensemble_variant

        for s in range(1, final_seeds + 1):
            if s > n_seeds and hours() > 0.9 * budget_hours:
                break
            if run_point(best, s) is None:
                continue
            dev_name = best + (f"_s{s}" if s != 1 else "")
            run_name = f"{prefix}_best_s{s}"
            if not (results_dir() / "runs" / run_name / "metrics.json").exists():
                promote(dev_name, run_name, log=lambda *a: None)
            has_kl = (_run_dir(f"{run_name}+kl") / "scores.npz").exists()
            if score == "ens" and has_kl and not (_run_dir(f"{run_name}+ens") / "metrics.json").exists():
                ensemble_variant([run_name, f"{run_name}+kl"], f"{run_name}+ens")
            promoted.append(run_name)
        if len(promoted) > 1:
            ensemble_variant(promoted, f"{prefix}_best_seeds")
            ens = [f"{r}+ens" for r in promoted if (_run_dir(f"{r}+ens") / "scores.npz").exists()]
            if len(ens) == len(promoted):
                ensemble_variant(ens, f"{prefix}_best_seeds+ens")

    def entry(name: str) -> dict[str, Any]:
        p, lab = names[name]
        rec, sel = _dev_auc(name), _dev_auc(_ranked_entry(name, score))
        return {
            "name": name,
            "point": p,
            "labels": lab,
            "auc": sel[0],
            "max_sic": sel[1],
            "rec_auc": rec[0],
            "rec_max_sic": rec[1],
        }

    summary = {
        "base_config": base_config,
        "grid": {k: v if isinstance(v, Mapping) else list(v) for k, v in grid.items()},
        "overrides": overrides or {},
        "score": score,
        "phase1": [entry(n) for n, _ in scored],
        "seeds": [{"name": n, "auc_mean": m, "auc_std": sd, "n": len(seeds_done[n])} for n, m, sd in ranked],
        "best": None if best is None else {"point": names[best][0], "labels": names[best][1], "name": best},
        "promoted": promoted,
        "hours": hours(),
        **run_metadata(),
    }
    save_json(summary, results_dir() / "dev" / f"{prefix}_sweep.json")
    what = "rank ensemble of reconstruction and KL scores" if score == "ens" else "reconstruction score"
    lines = [
        f"# Sweep `{prefix}` (dev protocol; SR AUC / max SIC, prod aggregation; ranked by the {what})",
        "",
        "| rank | grid point | dev AUC | max SIC | reconstruction score AUC / max SIC |",
        "|---:|---|---:|---:|---|",
    ]
    lines += [
        f"| {i + 1} | {d['name']} | {d['auc']:.3f} | {d['max_sic']:.2f} | {d['rec_auc']:.3f} / {d['rec_max_sic']:.2f} |"
        for i, d in enumerate(summary["phase1"])
    ]
    lines += ["", "| seed-confirmed point | mean dev AUC | std | seeds |", "|---|---:|---:|---:|"]
    lines += [
        f"| {d['name']} | {d['auc_mean']:.3f} | {d['auc_std']:.3f} | {d['n']} |" for d in summary["seeds"]
    ]
    (results_dir() / "dev" / f"{prefix}_sweep.md").write_text("\n".join(lines) + "\n")
    log(f"[sweep] done in {hours():.1f} h; best: {summary['best']}")
    return summary
