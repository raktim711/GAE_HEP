"""Fast development protocol: short training + label-aware metrics on the *validation* splits only.

Architecture and loss choices are compared on background-validation events versus the 2-prong **signal
validation** split (10% of the 2-prong sample, disjoint from ``sig_test``).  The test splits and the 3-prong
signal are never used here, so the final test numbers - and in particular the 3-prong generalisation test -
are not biased by these choices.  Using a signal validation split for design decisions is a weak form of
supervision (as in most LHCO benchmark studies); results based on it are labelled "dev".

A dev run lives in ``results/dev/<name>/`` (``config.yaml``, ``history.json``, ``dev_metrics.json``;
weights are git-ignored).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from gaehep.config import load_json, results_dir, run_metadata, save_json
from gaehep.eval.evaluate import aggregate, region_mask
from gaehep.eval.metrics import auc, summarize
from gaehep.models import build_model
from gaehep.train.loop import TrainConfig, fit, score_pool
from gaehep.train.run import Data

DEV_MIN_BKG = 20


def dev_dir(name: str) -> Path:
    return results_dir() / "dev" / name


def _tab(data: Data, split: str, max_events: int | None) -> dict[str, np.ndarray]:
    st = data.store("rnd")
    ev = data.events(split, max_events)
    return {k: st[k][ev] for k in ("mjj", "m", "n_const", "tau1", "tau2", "label")}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic)


def dev_metrics(
    s_bkg: np.ndarray, s_sig: np.ndarray, t_bkg: dict[str, np.ndarray], t_sig: dict[str, np.ndarray]
) -> dict[str, Any]:
    """s_* : (n_events, 2) per-jet scores; t_* : tables aligned with them."""
    out: dict[str, Any] = {}
    for region in ("full", "SR"):
        mb, ms = region_mask(t_bkg["mjj"], region), region_mask(t_sig["mjj"], region)
        for agg in ("max", "sum", "min"):
            e = summarize(aggregate(s_sig[ms], agg), aggregate(s_bkg[mb], agg), min_bkg=DEV_MIN_BKG)
            out[f"{region}/{agg}"] = {"auc": e["auc"], "max_sic": e["max_sic"], "rej@0.3": e["rej@0.3"]}
        out[f"{region}/jet"] = {"auc": auc(s_sig[ms].ravel(), s_bkg[mb].ravel())}
        # jets ordered by mass: the heavier jet carries X(500), the lighter one Y(100) in signal events
        hb = np.take_along_axis(s_bkg[mb], np.argsort(-t_bkg["m"][mb], axis=1), axis=1)
        hs = np.take_along_axis(s_sig[ms], np.argsort(-t_sig["m"][ms], axis=1), axis=1)
        out[f"{region}/heavy_jet"] = {"auc": auc(hs[:, 0], hb[:, 0])}
        out[f"{region}/light_jet"] = {"auc": auc(hs[:, 1], hb[:, 1])}
    sb = s_bkg.ravel().astype(np.float64)
    tau21 = np.where(t_bkg["tau1"] > 0, t_bkg["tau2"] / np.maximum(t_bkg["tau1"], 1e-12), np.nan).ravel()
    out["bkg_spearman"] = {
        "m": _spearman(sb, t_bkg["m"].ravel()),
        "n_const": _spearman(sb, t_bkg["n_const"].ravel().astype(np.float64)),
        "tau21": _spearman(sb, tau21),
    }
    return out


def headline(m: dict[str, Any]) -> str:
    return (
        f"SR: max {m['SR/max']['auc']:.3f}/{m['SR/max']['max_sic']:.2f}  "
        f"sum {m['SR/sum']['auc']:.3f}/{m['SR/sum']['max_sic']:.2f}  "
        f"min {m['SR/min']['auc']:.3f}/{m['SR/min']['max_sic']:.2f}  "
        f"jet {m['SR/jet']['auc']:.3f}  heavy {m['SR/heavy_jet']['auc']:.3f}  "
        f"light {m['SR/light_jet']['auc']:.3f} | full max {m['full/max']['auc']:.3f} | "
        f"rho(m) {m['bkg_spearman']['m']:+.2f} rho(n) {m['bkg_spearman']['n_const']:+.2f} "
        f"rho(tau21) {m['bkg_spearman']['tau21']:+.2f}"
    )


def dev_run(cfg: dict[str, Any], name: str, bkg_events: int | None = None, log=print) -> dict[str, Any]:
    out_dir = dev_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    dcfg = cfg.get("data", {})
    data = Data(n_const=int(dcfg.get("n_const", 50)))
    scaler = data.scaler()
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    if model.supervised:
        raise ValueError("the dev protocol is for unsupervised models")
    tcfg = TrainConfig.from_dict(cfg.get("train"))
    log(f"[dev:{name}] {cfg['model']} with {sum(p.numel() for p in model.parameters()):,} parameters")
    hist = out_dir / "history.json"
    if (out_dir / "model.pt").exists() and hist.exists() and load_json(hist).get("status") == "done":
        model.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True))
        summary = load_json(hist)
    else:
        t0 = time.time()
        summary = fit(
            model,
            data.pool("bkg_train", dcfg.get("max_train_events")),
            data.pool("bkg_val"),
            scaler,
            tcfg,
            out_dir,
            log=log,
        )
        summary["train_minutes"] = (time.time() - t0) / 60
        save_json(summary, hist)
        if summary["status"] != "done":
            return summary
    t0 = time.time()
    torch.set_num_threads(tcfg.threads)
    keys = tuple(cfg.get("dev", {}).get("extra_scores", ()))
    o_bkg = score_pool(model, data.pool("bkg_val", bkg_events), scaler, keys=keys)
    o_sig = score_pool(model, data.pool("sig_val"), scaler, keys=keys)
    s_bkg, s_sig = o_bkg["score"].reshape(-1, 2), o_sig["score"].reshape(-1, 2)
    tables = (_tab(data, "bkg_val", bkg_events), _tab(data, "sig_val", None))
    metrics = dev_metrics(s_bkg, s_sig, *tables)
    np.savez(out_dir / "dev_scores.npz", bkg_val=s_bkg.astype(np.float32), sig_val=s_sig.astype(np.float32))
    for k in keys:  # extra per-jet scores (e.g. the KL term of a VAE) from the same forward pass
        r = _save_dev(
            f"{name}:{k}",
            o_bkg[k].reshape(-1, 2),
            o_sig[k].reshape(-1, 2),
            data,
            {"source_run": name, "train_minutes": 0, "config": cfg},
            tables=tables,
        )
        log(f"[dev:{name}:{k}] {headline(r['metrics'])}")
    res = {
        "name": name,
        "metrics": metrics,
        "best_val_loss": summary.get("best_val_loss"),
        "epochs": summary.get("epochs_done"),
        "train_minutes": summary.get("train_minutes"),
        "score_minutes": (time.time() - t0) / 60,
        "config": cfg,
        **run_metadata(),
    }
    save_json(res, out_dir / "dev_metrics.json")
    log(f"[dev:{name}] {headline(metrics)}")
    return res


def _save_dev(
    name: str,
    s_bkg: np.ndarray,
    s_sig: np.ndarray,
    data: Data,
    extra: dict[str, Any],
    tables: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None = None,
) -> dict:
    tables = tables or (_tab(data, "bkg_val", None), _tab(data, "sig_val", None))
    metrics = dev_metrics(s_bkg, s_sig, *tables)
    res = {"name": name, "metrics": metrics, **extra, **run_metadata()}
    dev_dir(name).mkdir(parents=True, exist_ok=True)
    save_json(res, dev_dir(name) / "dev_metrics.json")
    return res


def dev_eval_run(run: str, log=print) -> dict[str, Any]:
    """Dev metrics of a fully trained run in ``results/runs/<run>`` (bkg_val scores reused, sig_val scored)."""
    from gaehep.train.run import _run_dir

    src = _run_dir(run)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    data = Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    scaler = data.scaler()
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(src / "model.pt", weights_only=True))
    torch.set_num_threads(TrainConfig.from_dict(cfg.get("train")).threads)
    with np.load(src / "scores.npz") as f:
        s_bkg = f["bkg_val"] if "bkg_val" in f.files else None
    if s_bkg is None:
        s_bkg = score_pool(model, data.pool("bkg_val"), scaler)["score"].reshape(-1, 2)
    s_sig = score_pool(model, data.pool("sig_val"), scaler)["score"].reshape(-1, 2)
    hist = load_json(src / "history.json") if (src / "history.json").exists() else {}
    res = _save_dev(
        f"{run}@full",
        s_bkg,
        s_sig,
        data,
        {
            "source_run": run,
            "best_val_loss": hist.get("best_val_loss"),
            "epochs": hist.get("epochs_done"),
            "train_minutes": sum(h.get("seconds", 0) for h in hist.get("history", [])) / 60,
            "config": cfg,
        },
    )
    log(f"[dev:{run}@full] {headline(res['metrics'])}")
    return res


def _jet_context(pool) -> np.ndarray:
    return np.stack([np.log(np.maximum(pool.m, 1.0)), np.log(pool.pt)], axis=-1).astype(np.float32)


def dev_flow(
    name: str, train_events: int = 50_000, epochs: int = 30, conditional: bool = True, log=print
) -> list[dict[str, Any]]:
    """Latent-density variants of dev run ``name``: a flow fitted on background-train latents, score -log p(z)
    (``+flow``) and, if ``conditional``, a flow conditioned on the jet's (log m, log pT), score -log p(z | m, pT)
    (``+cflow``, mass-decorrelated substructure anomaly)."""
    from gaehep.models.latent_flow import LatentFlow

    src = dev_dir(name)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    data = Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    scaler = data.scaler()
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(src / "model.pt", weights_only=True))
    torch.set_num_threads(TrainConfig.from_dict(cfg.get("train")).threads)
    lat, ctx = {}, {}
    for split, cap in (("bkg_train", train_events), ("bkg_val", None), ("sig_val", None)):
        pool = data.pool(split, cap)
        out = score_pool(model, pool, scaler, with_embed=True)
        if "z" not in out:
            raise ValueError(f"model {cfg['model']} has no graph-level latent")
        lat[split], ctx[split] = out["z"], _jet_context(pool)
    d = lat["bkg_train"].shape[-1]
    vsel = np.random.default_rng(0).choice(len(lat["bkg_val"]), 40_000, replace=False)
    results = []
    for suffix, cdim in (("+flow", 0), ("+cflow", 2)) if conditional else (("+flow", 0),):
        c = ctx if cdim else {k: None for k in ctx}
        flow = LatentFlow(d, seed=0, context_dim=cdim).fit(
            lat["bkg_train"],
            lat["bkg_val"][vsel],
            epochs=epochs,
            patience=5,
            c_train=c["bkg_train"],
            c_val=None if c["bkg_val"] is None else c["bkg_val"][vsel],
        )
        s_bkg = flow.nll(lat["bkg_val"], context=c["bkg_val"]).reshape(-1, 2)
        s_sig = flow.nll(lat["sig_val"], context=c["sig_val"]).reshape(-1, 2)
        res = _save_dev(
            f"{name}{suffix}",
            s_bkg,
            s_sig,
            data,
            {"source_run": name, "flow_val_nll": flow.history["val_nll"], "train_minutes": 0, "config": cfg},
        )
        log(f"[dev:{name}{suffix}] {headline(res['metrics'])}")
        results.append(res)
    return results


def dev_baselines(log=print) -> None:
    """Single-observable references under the dev protocol."""
    from gaehep.eval.evaluate import observable_scores

    data = Data(n_const=50)
    st = data.store("rnd")
    keys = ("m", "n_const", "tau1", "tau2", "tau3")
    ob = observable_scores({k: st[k][data.events("bkg_val")] for k in keys})
    os_ = observable_scores({k: st[k][data.events("sig_val")] for k in keys})
    for obs in ob:
        res = _save_dev(f"baseline_{obs}", ob[obs], os_[obs], data, {"baseline": obs, "train_minutes": 0})
        log(f"[dev:baseline_{obs}] {headline(res['metrics'])}")


def dev_report(root: Path | None = None) -> str:
    root = Path(root) if root else results_dir() / "dev"
    lines = [
        "| dev run | SR max AUC/maxSIC | SR sum AUC/maxSIC | SR min AUC | SR jet AUC | heavy | light | "
        "ρ(m) | ρ(n) | ρ(τ21) | min |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(root.glob("*/dev_metrics.json")):
        d = load_json(p)
        m = d["metrics"]
        lines.append(
            f"| {d['name']} | {m['SR/max']['auc']:.3f} / {m['SR/max']['max_sic']:.2f} | "
            f"{m['SR/sum']['auc']:.3f} / {m['SR/sum']['max_sic']:.2f} | {m['SR/min']['auc']:.3f} | "
            f"{m['SR/jet']['auc']:.3f} | {m['SR/heavy_jet']['auc']:.3f} | {m['SR/light_jet']['auc']:.3f} | "
            f"{m['bkg_spearman']['m']:+.2f} | {m['bkg_spearman']['n_const']:+.2f} | "
            f"{m['bkg_spearman']['tau21']:+.2f} | {d.get('train_minutes') or 0:.0f} |"
        )
    md = "\n".join(lines)
    (root / "summary.md").write_text(
        "# Development runs (validation splits: bkg_val vs 2-prong sig_val; never the test splits)\n\n"
        "AUC / max SIC in the signal region (3.3-3.7 TeV) for event scores max/sum/min over the two jets, "
        "jet-level AUC, heavier/lighter jet AUC, and Spearman correlations of the per-jet score with "
        "m_j, multiplicity and τ21 on background; last column = training minutes (2 CPU cores).\n\n"
        + md
        + "\n"
    )
    return md
