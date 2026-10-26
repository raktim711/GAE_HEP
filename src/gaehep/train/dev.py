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

from gaehep.config import load_json, results_dir, run_metadata, save_json, seed_everything
from gaehep.data.dataset import iterate_batches
from gaehep.eval.evaluate import aggregate, region_mask
from gaehep.eval.metrics import auc, summarize
from gaehep.models import build_model
from gaehep.train.loop import TrainConfig, fit, resolve_device, score_pool
from gaehep.train.run import Data, extra_score_keys, train_window

DEV_MIN_BKG = 20
# name prefixes of hyper-parameter-sweep dev entries (CPU study: hp_, fx_, cd_, e2_, seed ensembles; GPU: gpu_)
SWEEP_PREFIXES = ("hp_", "fx_", "cd_", "e2_", "seeds", "x_", "gpu_", "lt_")
# training budget of the dev protocol (applied over a model config's ``train`` section by ``gaehep dev``):
# 5 epochs x 120k jets, 20k validation jets, one thread so that two dev runs share the 2 CPU cores.
DEV_TRAIN = {"epochs": 5, "jets_per_epoch": 120_000, "patience": 3, "val_jets": 20_000, "threads": 1}


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
    ref = np.sort(np.asarray(s_bkg, dtype=np.float64).ravel())  # background CDF for the 'prod' aggregation
    for region in ("full", "SR"):
        mb, ms = region_mask(t_bkg["mjj"], region), region_mask(t_sig["mjj"], region)
        for agg in ("max", "sum", "min", "prod"):
            e = summarize(aggregate(s_sig[ms], agg, ref), aggregate(s_bkg[mb], agg, ref), min_bkg=DEV_MIN_BKG)
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
        f"prod {m['SR/prod']['auc']:.3f}/{m['SR/prod']['max_sic']:.2f}  "
        f"jet {m['SR/jet']['auc']:.3f}  heavy {m['SR/heavy_jet']['auc']:.3f}  "
        f"light {m['SR/light_jet']['auc']:.3f} | full max {m['full/max']['auc']:.3f} | "
        f"rho(m) {m['bkg_spearman']['m']:+.2f} rho(n) {m['bkg_spearman']['n_const']:+.2f} "
        f"rho(tau21) {m['bkg_spearman']['tau21']:+.2f}"
    )


def dev_run(
    cfg: dict[str, Any], name: str, bkg_events: int | None = None, log=print, data: Data | None = None
) -> dict[str, Any]:
    """Train (or reuse) a dev-protocol model and evaluate it on the validation splits.  ``data`` lets a sweep
    reuse loaded jet pools (it must have the config's ``data.n_const``)."""
    out_dir = dev_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    dcfg = cfg.get("data", {})
    n_const = int(dcfg.get("n_const", 50))
    if data is None or data.n_const != n_const:
        data = Data(n_const=n_const)
    scaler = data.scaler()
    tcfg = TrainConfig.from_dict(cfg.get("train"))
    seed_everything(tcfg.seed)  # the initial weights are part of the seed's randomness
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    if model.supervised:
        raise ValueError("the dev protocol is for unsupervised models")
    log(f"[dev:{name}] {cfg['model']} with {sum(p.numel() for p in model.parameters()):,} parameters")
    hist = out_dir / "history.json"
    if (out_dir / "model.pt").exists() and hist.exists() and load_json(hist).get("status") == "done":
        model.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True, map_location="cpu"))
        model.to(resolve_device(tcfg.device))
        summary = load_json(hist)
    else:
        t0 = time.time()
        summary = fit(
            model,
            data.pool("bkg_train", dcfg.get("max_train_events"), mjj=train_window(dcfg)),
            data.pool("bkg_val", mjj=train_window(dcfg)),
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
    keys = extra_score_keys(cfg)
    o_bkg = score_pool(model, data.pool("bkg_val", bkg_events), scaler, keys=keys)
    o_sig = score_pool(model, data.pool("sig_val"), scaler, keys=keys)
    s_bkg, s_sig = o_bkg["score"].reshape(-1, 2), o_sig["score"].reshape(-1, 2)
    tables = (_tab(data, "bkg_val", bkg_events), _tab(data, "sig_val", None))
    metrics = dev_metrics(s_bkg, s_sig, *tables)
    np.savez(out_dir / "dev_scores.npz", bkg_val=s_bkg.astype(np.float32), sig_val=s_sig.astype(np.float32))
    for k in keys:  # extra per-jet scores (e.g. the KL term of a VAE) from the same forward pass
        r = _save_dev(
            f"{name}+{k}",
            o_bkg[k].reshape(-1, 2),
            o_sig[k].reshape(-1, 2),
            data,
            {"source_run": name, "train_minutes": 0, "config": cfg},
            tables=tables,
        )
        log(f"[dev:{name}+{k}] {headline(r['metrics'])}")
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
    np.savez(
        dev_dir(name) / "dev_scores.npz",
        bkg_val=np.asarray(s_bkg, dtype=np.float32),
        sig_val=np.asarray(s_sig, dtype=np.float32),
    )
    return res


def dev_ensemble(names: list[str], out_name: str, log=print) -> dict[str, Any]:
    """Rank ensemble of dev entries (several seeds, or a VAE's reconstruction and KL scores): per jet, the
    average of the entries' background-validation CDF values, evaluated like a dev run (as
    :func:`gaehep.train.run.ensemble_variant` does for test runs)."""
    from gaehep.train.run import _ecdf

    per = []
    for n in names:
        with np.load(dev_dir(n) / "dev_scores.npz") as f:
            per.append((f["bkg_val"].astype(np.float64), f["sig_val"].astype(np.float64)))
    if len({b.shape for b, _ in per}) != 1:
        raise ValueError(f"dev entries scored on different background events: {names}")
    acc_b, acc_s = np.zeros_like(per[0][0]), np.zeros_like(per[0][1])
    for b, s in per:
        ref = np.sort(b.ravel())
        acc_b += _ecdf(ref, b)
        acc_s += _ecdf(ref, s)
    res = _save_dev(
        out_name,
        acc_b / len(per),
        acc_s / len(per),
        Data(n_const=50),
        {"source_runs": list(names), "posthoc": "rank_ensemble", "train_minutes": 0},
    )
    log(f"[dev:{out_name}] {headline(res['metrics'])}")
    return res


def dev_kme(
    out_name: str,
    bank_jets: int = 100_000,
    n_pca: int = 64,
    k: int = 10,
    dim_per_scale: int = 128,
    sigmas: tuple[float, ...] | None = None,
    bank_mjj: tuple[float, float] | None = None,
    log=print,
) -> dict[str, Any]:
    """Training-free scores in the energy-MMD geometry (:mod:`gaehep.eval.kme`): kNN distance, whitened kNN
    distance and Mahalanobis distance of the random-Fourier kernel mean embedding to a bank of background-
    *training* jets (optionally restricted to an m_jj window).  Three dev entries ``<out_name>+knn`` etc."""
    from gaehep.eval.kme import EmbeddingScorer, KernelEmbedding, embed_pool
    from gaehep.losses.mmd import DEFAULT_SIGMAS

    data = Data(n_const=50)
    scaler = data.scaler()
    emb = KernelEmbedding(sigmas or DEFAULT_SIGMAS, dim_per_scale, seed=0)
    bank_pool = data.pool("bkg_train", max_events=bank_jets // 2, mjj=bank_mjj)
    scorer = EmbeddingScorer(embed_pool(bank_pool, scaler, emb), n_pca=n_pca, k=k)
    tables = (_tab(data, "bkg_val", None), _tab(data, "sig_val", None))
    sc_b = scorer.scores(embed_pool(data.pool("bkg_val"), scaler, emb))
    sc_s = scorer.scores(embed_pool(data.pool("sig_val"), scaler, emb))
    res = {}
    for key in sc_b:
        res[key] = _save_dev(
            f"{out_name}+{key}",
            sc_b[key].reshape(-1, 2),
            sc_s[key].reshape(-1, 2),
            data,
            {"posthoc": "kme", "bank_jets": len(bank_pool), "n_pca": n_pca, "k": k, "train_minutes": 0},
            tables=tables,
        )
        log(f"[dev:{out_name}+{key}] {headline(res[key]['metrics'])}")
    return res


def dev_profile(
    name: str,
    out_name: str | None = None,
    train_jets: int = 100_000,
    n_components: int = 6,
    log=print,
) -> dict[str, Any]:
    """Density of the *error profile*: for a trained dev energy-flow model, the per-kernel-scale MMD errors
    (and optionally the latent mean and the KL term) of a jet form a short vector; a Gaussian mixture fitted
    to that vector on background-training jets gives -log p as a label-free score that flags jets whose error
    across angular scales is unusual even when the total reconstruction error is moderate.  Entries
    ``<out>+prof_mmd`` (5 log-errors), ``+prof_mmdkl`` (adds log KL), ``+prof_all`` (adds the 4 latent
    means), each as a GMM density; ``+prof_maha`` is the Mahalanobis distance of the 5 log-errors."""
    from sklearn.mixture import GaussianMixture

    from gaehep.losses.mmd import energy_mmd

    out_name = out_name or name
    cfg = yaml.safe_load((dev_dir(name) / "config.yaml").read_text())
    data = Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    scaler = data.scaler()
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(dev_dir(name) / "model.pt", weights_only=True, map_location="cpu"))
    model.eval()
    torch.set_num_threads(1)
    sig = tuple(model.sigmas)

    @torch.no_grad()
    def feats(pool) -> np.ndarray:
        rows = []
        for b in iterate_batches(pool, scaler, 1024):
            out = model(b)
            xy, w = model.target_flow(b)
            cols = [energy_mmd(out["xy"], out["w"], xy, w, (s,)) for s in sig]
            cols.append(model.losses(b, out)["kl"])
            cols.extend(out["z"].unbind(-1))
            rows.append(torch.stack(cols, dim=1).numpy())
        return np.concatenate(rows).astype(np.float64)

    ns = len(sig)
    f_tr = feats(data.pool("bkg_train", max_events=train_jets // 2))
    f_b, f_s = feats(data.pool("bkg_val")), feats(data.pool("sig_val"))

    def prep(f: np.ndarray, cols: list[int]) -> np.ndarray:
        x = f[:, cols].copy()
        for j, c in enumerate(cols):
            if c <= ns:  # errors and KL are positive and heavy-tailed: log scale
                x[:, j] = np.log(np.maximum(x[:, j], 1e-9))
        return x

    sets = {
        "prof_mmd": list(range(ns)),
        "prof_mmdkl": list(range(ns + 1)),
        "prof_all": list(range(f_tr.shape[1])),
    }
    tables = (_tab(data, "bkg_val", None), _tab(data, "sig_val", None))
    res = {}
    for key, cols in sets.items():
        x_tr = prep(f_tr, cols)
        mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
        gm = GaussianMixture(n_components, covariance_type="full", random_state=0, reg_covar=1e-4).fit(
            (x_tr - mu) / sd
        )
        sc = [-gm.score_samples((prep(f, cols) - mu) / sd) for f in (f_b, f_s)]
        res[key] = _save_dev(
            f"{out_name}+{key}",
            sc[0].reshape(-1, 2),
            sc[1].reshape(-1, 2),
            data,
            {"posthoc": "error_profile_gmm", "source_run": name, "train_minutes": 0},
            tables=tables,
        )
        log(f"[dev:{out_name}+{key}] {headline(res[key]['metrics'])}")
        if key == "prof_mmd":  # Mahalanobis distance of the log-error vector
            xs = [(prep(f, cols) - mu) / sd for f in (f_tr, f_b, f_s)]
            cov_inv = np.linalg.inv(np.cov(xs[0].T) + 1e-6 * np.eye(len(cols)))
            md = [np.einsum("ij,jk,ik->i", x, cov_inv, x) for x in xs[1:]]
            res["prof_maha"] = _save_dev(
                f"{out_name}+prof_maha",
                md[0].reshape(-1, 2),
                md[1].reshape(-1, 2),
                data,
                {"posthoc": "error_profile_mahalanobis", "source_run": name, "train_minutes": 0},
                tables=tables,
            )
            log(f"[dev:{out_name}+prof_maha] {headline(res['prof_maha']['metrics'])}")
    return res


def dev_rescore(root: Path | None = None, log=print) -> None:
    """Recompute the metrics of every dev entry that has saved validation scores (after metric changes)."""
    root = Path(root) if root else results_dir() / "dev"
    data = Data(n_const=50)
    tables = (_tab(data, "bkg_val", None), _tab(data, "sig_val", None))
    for p in sorted(root.glob("*/dev_scores.npz")):
        mp = p.parent / "dev_metrics.json"
        if not mp.exists():
            continue
        d = load_json(mp)
        with np.load(p) as f:
            sb, ss = f["bkg_val"], f["sig_val"]
        if len(sb) != len(tables[0]["mjj"]):
            log(f"[dev-rescore] skip {p.parent.name}: scored on a background subset")
            continue
        d["metrics"] = dev_metrics(sb, ss, *tables)
        save_json(d, mp)
        log(f"[dev-rescore] {p.parent.name}: {headline(d['metrics'])}")


def dev_eval_run(run: str, log=print) -> dict[str, Any]:
    """Dev metrics of a fully trained run in ``results/runs/<run>`` (bkg_val scores reused, sig_val scored)."""
    from gaehep.train.run import _run_dir

    src = _run_dir(run)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    data = Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    scaler = data.scaler()
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(src / "model.pt", weights_only=True, map_location="cpu"))
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


def _dev_latents(name: str, data: Data, cfg: dict, train_events: int) -> dict[str, np.ndarray]:
    """Jet latents and (log m, log pT) contexts of bkg_train (subset), bkg_val and sig_val; cached."""
    cache = dev_dir(name) / "dev_latents.npz"
    if cache.exists():
        with np.load(cache) as f:
            return {k: f[k] for k in f.files}
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(dev_dir(name) / "model.pt", weights_only=True))
    torch.set_num_threads(TrainConfig.from_dict(cfg.get("train")).threads)
    scaler = data.scaler()
    out = {}
    for split, cap in (("bkg_train", train_events), ("bkg_val", None), ("sig_val", None)):
        pool = data.pool(split, cap)
        o = score_pool(model, pool, scaler, with_embed=True)
        if "z" not in o:
            raise ValueError(f"model {cfg['model']} has no graph-level latent")
        out[f"z_{split}"], out[f"c_{split}"] = o["z"], _jet_context(pool)
    np.savez(cache, **out)
    return out


def _event_latents(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Concatenated latents of the two jets of each event, lighter jet first (jet-order invariant)."""
    d = z.shape[-1]
    z, logm = z.reshape(-1, 2, d), c[:, 0].reshape(-1, 2)
    o = np.argsort(logm, axis=1, kind="stable")
    return np.take_along_axis(z, o[..., None], 1).reshape(len(z), 2 * d)


def dev_flow(
    name: str,
    train_events: int = 50_000,
    epochs: int = 30,
    kinds: tuple[str, ...] = ("flow", "cflow", "eflow"),
    log=print,
) -> list[dict[str, Any]]:
    """Latent-density variants of dev run ``name`` (flows fitted on background-train latents):

    * ``+flow``  per-jet -log p(z);
    * ``+cflow`` per-jet -log p(z | log m, log pT) (mass-decorrelated substructure anomaly);
    * ``+eflow`` event-level -log p(z_light, z_heavy): joint density of both jets' latents (an event score,
      stored as identical values for the two jets).
    """
    from gaehep.models.latent_flow import LatentFlow

    cfg = yaml.safe_load((dev_dir(name) / "config.yaml").read_text())
    data = Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    L = _dev_latents(name, data, cfg, train_events)
    lat = {s: L[f"z_{s}"] for s in ("bkg_train", "bkg_val", "sig_val")}
    ctx = {s: L[f"c_{s}"] for s in ("bkg_train", "bkg_val", "sig_val")}
    results = []
    for kind in kinds:
        suffix = f"+{kind}"
        if kind == "eflow":
            x = {s: _event_latents(lat[s], ctx[s]) for s in lat}
            c = {s: None for s in lat}
        else:
            x = lat
            c = ctx if kind == "cflow" else {s: None for s in lat}
        vsel = np.random.default_rng(0).choice(
            len(x["bkg_val"]), min(40_000, len(x["bkg_val"])), replace=False
        )
        flow = LatentFlow(x["bkg_train"].shape[-1], seed=0, context_dim=2 if kind == "cflow" else 0).fit(
            x["bkg_train"],
            x["bkg_val"][vsel],
            epochs=epochs,
            patience=5,
            c_train=c["bkg_train"],
            c_val=None if c["bkg_val"] is None else c["bkg_val"][vsel],
        )
        s_bkg = flow.nll(x["bkg_val"], context=c["bkg_val"])
        s_sig = flow.nll(x["sig_val"], context=c["sig_val"])
        if kind == "eflow":
            s_bkg, s_sig = np.repeat(s_bkg[:, None], 2, 1), np.repeat(s_sig[:, None], 2, 1)
        s_bkg, s_sig = s_bkg.reshape(-1, 2), s_sig.reshape(-1, 2)
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
    dev_oracle_density(data, log=log)


ORACLE_SETS = {
    "m": (("logm",), 60),
    "m,t21": (("logm", "t21"), 40),
    "m,t21,t32": (("logm", "t21", "t32"), 20),
    "m,pt,t21": (("logm", "logpt", "t21"), 20),
}


def _jet_obs(st, ev: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        t21 = np.where(st["tau1"][ev] > 0, st["tau2"][ev] / st["tau1"][ev], 1.0)
        t32 = np.where(st["tau2"][ev] > 0, st["tau3"][ev] / st["tau2"][ev], 1.0)
    return {
        "logm": np.log(np.maximum(st["m"][ev], 1.0)),
        "logpt": np.log(st["pt"][ev]),
        "t21": np.nan_to_num(t21, nan=1.0),
        "t32": np.nan_to_num(t32, nan=1.0),
    }


def dev_oracle_density(data: Data | None = None, log=print) -> None:
    """Reference: per-jet score -log p_QCD(observables), p from a (smoothed) histogram of background-training
    jets.  Measures how far *pure* density-based anomaly detection gets with ideal low-dimensional jet
    features (no "heavier / more prongs = anomalous" prior)."""
    data = data or Data(n_const=50)
    st = data.store("rnd")
    tr, bv, sv = (_jet_obs(st, data.events(s)) for s in ("bkg_train", "bkg_val", "sig_val"))
    for name, (keys, bins) in ORACLE_SETS.items():
        x = np.stack([tr[k].ravel() for k in keys], 1)
        lo, hi = np.quantile(x, 0.0005, 0), np.quantile(x, 0.9995, 0)
        edges = [np.linspace(a, b, bins + 1) for a, b in zip(lo, hi, strict=True)]
        h, _ = np.histogramdd(x, bins=edges)
        p = (h + 0.5) / (h.sum() + 0.5 * h.size)

        def score(d, keys=keys, edges=edges, p=p):
            idx = tuple(
                np.clip(np.searchsorted(e, d[k], side="right") - 1, 0, len(e) - 2)
                for k, e in zip(keys, edges, strict=True)
            )
            return -np.log(p[idx])

        res = _save_dev(
            f"oracle_density_{name}",
            score(bv),
            score(sv),
            data,
            {"oracle_density": list(keys), "train_minutes": 0},
        )
        log(f"[dev:oracle_density_{name}] {headline(res['metrics'])}")


def _plot_dev(runs: list[dict], path: Path) -> None:
    """Signal-region event AUC (max / sum / min / prod aggregation) of every dev run, sorted by the best."""
    from gaehep.eval import style

    style.apply()
    import matplotlib.pyplot as plt

    aggs = (("max", "o"), ("sum", "s"), ("min", "^"), ("prod", "D"))
    rows = []
    for d in runs:
        m = d["metrics"]
        rows.append((d["name"], [m.get(f"SR/{a}", {}).get("auc", np.nan) for a, _ in aggs]))
    if not rows:
        return
    rows.sort(key=lambda r: np.nanmax(r[1]))
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rows) + 1.4))
    y = np.arange(len(rows))
    for j, (agg, mk) in enumerate(aggs):
        ax.scatter([r[1][j] for r in rows], y, marker=mk, s=24, label=agg, zorder=3)
    for i, (name, _) in enumerate(rows):
        ref = name.startswith(("baseline_", "oracle_"))
        ax.axhspan(i - 0.5, i + 0.5, color="0.93" if ref else "white", zorder=0)
    ax.axvline(0.5, color="grey", lw=1, ls=":")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlabel("event AUC in the SR (2-prong signal validation split)")
    ax.legend(title="event score", loc="lower right", fontsize=9)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def dev_report(root: Path | None = None) -> str:
    root = Path(root) if root else results_dir() / "dev"
    lines = [
        "| dev run | SR max AUC/maxSIC | SR sum AUC/maxSIC | SR min AUC/maxSIC | SR prod AUC/maxSIC | "
        "SR jet AUC | heavy | light | ρ(m) | ρ(n) | ρ(τ21) | min |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(m, agg):
        e = m.get(f"SR/{agg}")
        return "–" if e is None else f"{e['auc']:.3f} / {e['max_sic']:.2f}"

    for p in sorted(root.glob("*/dev_metrics.json")):
        d = load_json(p)
        m = d["metrics"]
        lines.append(
            f"| {d['name']} | {cell(m, 'max')} | {cell(m, 'sum')} | {cell(m, 'min')} | {cell(m, 'prod')} | "
            f"{m['SR/jet']['auc']:.3f} | {m['SR/heavy_jet']['auc']:.3f} | {m['SR/light_jet']['auc']:.3f} | "
            f"{m['bkg_spearman']['m']:+.2f} | {m['bkg_spearman']['n_const']:+.2f} | "
            f"{m['bkg_spearman']['tau21']:+.2f} | {d.get('train_minutes') or 0:.0f} |"
        )
    md = "\n".join(lines)
    # the figure shows the design-study entries; hyper-parameter sweep entries have their own tables/figures
    # (scripts/hp_study.py, <prefix>_sweep.md) and would make it hundreds of rows tall
    _plot_dev(
        [
            d
            for d in (load_json(p) for p in sorted(root.glob("*/dev_metrics.json")))
            if not d["name"].startswith(SWEEP_PREFIXES)
        ],
        root / "summary.png",
    )
    (root / "summary.md").write_text(
        "# Development runs (validation splits: bkg_val vs 2-prong sig_val; never the test splits)\n\n"
        "AUC / max SIC in the signal region (3.3-3.7 TeV) for event scores max/sum/min over the two jets and "
        "prod = log F_B(s_1) + log F_B(s_2) (F_B = background CDF of jet scores), "
        "jet-level AUC, heavier/lighter jet AUC, and Spearman correlations of the per-jet score with "
        "m_j, multiplicity and τ21 on background; last column = training minutes (2 CPU cores).\n\n"
        + md
        + "\n"
    )
    return md


def promote(dev_run: str, name: str, log=print) -> dict[str, Any]:
    """Evaluate a dev-protocol model on the *test* splits: copy its weights and history to
    ``results/runs/<name>`` and run the score + evaluate stages with the dev run's exact config.

    Used for the final models, whose training budget (like every other design choice) was selected on the
    validation splits."""
    import shutil

    from gaehep.train.run import _run_dir, run

    src, dst = dev_dir(dev_run), _run_dir(name)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("model.pt", "history.json"):
        shutil.copy2(src / f, dst / f)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    cfg["promoted_from_dev_run"] = dev_run
    return run(cfg, name, stages=("score", "evaluate"), log=log)
