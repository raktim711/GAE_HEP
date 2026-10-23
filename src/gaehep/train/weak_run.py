"""Weak-supervision study on top of a trained run's permutation-invariant jet latents (see eval/weak.py)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import yaml

from gaehep.config import run_metadata, save_json
from gaehep.eval import weak
from gaehep.models import build_model
from gaehep.train.loop import TrainConfig, score_pool
from gaehep.train.run import SPLIT_SAMPLE, Data, _run_dir

KEYS = ("mjj", "m", "tau1", "tau2")


def _sr_events(data: Data, split: str) -> np.ndarray:
    st = data.store(SPLIT_SAMPLE[split])
    ev = data.events(split)
    return ev[weak.sr_mask(st["mjj"][ev])]


def sr_latents(name: str, data: Data, splits=("bkg_train", "sig_train", "bkg_test", "sig_test"), log=print):
    """Per-jet latents (n_SR_events, 2, d) and anomaly scores (n_SR_events, 2) of the SR events of ``splits``
    (cached in latents_sr.npz as ``<split>`` and ``<split>:score``)."""
    src = _run_dir(name)
    cache = src / "latents_sr.npz"
    keys = [*splits, *(f"{s}:score" for s in splits)]
    if cache.exists():
        with np.load(cache) as f:
            if all(k in f.files for k in keys):
                return {k: f[k] for k in keys}
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(src / "model.pt", weights_only=True))
    torch.set_num_threads(TrainConfig.from_dict(cfg.get("train")).threads)
    scaler = data.scaler()
    out = {}
    for s in splits:
        ev = _sr_events(data, s)
        pool = data.store(SPLIT_SAMPLE[s]).pool(ev, data.n_const)
        o = score_pool(model, pool, scaler, with_embed=True)
        out[s] = o["z"].reshape(len(ev), 2, -1).astype(np.float32)
        out[f"{s}:score"] = o["score"].reshape(len(ev), 2).astype(np.float32)
        log(f"[{name}] SR latents {s}: {len(ev)} events")
    np.savez(cache, **out)
    return out


def weak_run(
    name: str,
    n_models: int = 5,
    n_inj: tuple[int, ...] = (150, 300, 600, 1200, 3000),
    reps: tuple[int, ...] = (0, 1, 2),
    log=print,
) -> dict[str, Any]:
    src = _run_dir(name)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    data = Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    splits = ("bkg_train", "sig_train", "bkg_test", "sig_test")
    tabs = {}
    for s in splits:
        st = data.store(SPLIT_SAMPLE[s])
        ev = _sr_events(data, s)
        tabs[s] = {k: st[k][ev] for k in KEYS}
    lat = sr_latents(name, data, splits, log=log)
    feats = {
        "hl4": {s: weak.hl4(tabs[s]) for s in splits},
        "score+m": {
            s: weak.latent_features(tabs[s], lat[f"{s}:score"][..., None], with_mass=True) for s in splits
        },
        "latent": {s: weak.latent_features(tabs[s], lat[s]) for s in splits},
        "latent+m": {s: weak.latent_features(tabs[s], lat[s], with_mass=True) for s in splits},
    }
    res = weak.run_iad(feats, n_inj=n_inj, reps=reps, n_models=n_models, log=log)
    settings = {"n_models": n_models, "n_inj": list(n_inj), "reps": list(reps)}
    save_json({"name": name, "settings": settings, "results": res, **run_metadata()}, src / "weak.json")
    from gaehep.eval.plots import plot_weak

    plot_weak(res, src / "figures" / "weak_iad.png", title=f"IAD on {name} latents (SR)")
    return res
