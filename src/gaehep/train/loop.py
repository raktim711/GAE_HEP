"""Generic training loop for :class:`~gaehep.models.base.JetModel`.

* AdamW + linear warm-up + cosine decay.
* Each epoch draws a fresh random subset of ``jets_per_epoch`` training jets (deterministic per epoch
  through ``default_rng([seed, epoch])``, so a resumed run sees the same data).
* Model selection = lowest **background validation loss** (labels are never used for unsupervised models);
  the best weights are deep-copied (the AIMS code stored a live ``state_dict`` reference instead).
* A checkpoint is written after every epoch and training stops cleanly when the next epoch would exceed the
  time budget (cloud commands are limited to ~2 h); calling :func:`fit` again resumes.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gaehep.config import save_json, seed_everything
from gaehep.data.dataset import JetPool, Scaler, iterate_batches
from gaehep.models.base import JetModel


@dataclass
class TrainConfig:
    epochs: int = 30
    jets_per_epoch: int = 200_000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    min_delta: float = 1e-3  # relative improvement needed to count as better
    val_jets: int = 50_000
    time_budget_min: float = 100.0
    seed: int = 1
    grad_clip: float = 5.0
    warmup_frac: float = 0.02
    prepare_batches: int = 20
    threads: int = 2
    device: str = "auto"  # "auto" -> cuda if available, else cpu

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        unknown = set(d or {}) - names
        if unknown:
            raise KeyError(f"unknown train options: {sorted(unknown)}")
        return cls(**(d or {}))


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _model_device(model: torch.nn.Module) -> torch.device:
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def _lr_lambda(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return f


@torch.no_grad()
def mean_loss(
    model: JetModel, pool: JetPool, scaler: Scaler, idx: np.ndarray, batch_size: int | None = None
) -> float:
    model.eval()
    batch_size = batch_size or getattr(model, "eval_batch_size", 1024)
    dev = _model_device(model)
    tot, n = 0.0, 0
    sub = pool.subset(idx)
    for b in iterate_batches(sub, scaler, batch_size):
        b = b.to(dev)
        tot += float(model.losses(b)["total"].sum())
        n += len(b)
    return tot / max(n, 1)


def fit(
    model: JetModel,
    train_pool: JetPool,
    val_pool: JetPool,
    scaler: Scaler,
    cfg: TrainConfig,
    run_dir: Path,
    log=print,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    model.to(device)

    n_train = min(cfg.jets_per_epoch, len(train_pool))
    steps_per_epoch = math.ceil(n_train / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(total_steps, int(cfg.warmup_frac * total_steps))
    )
    val_idx = np.sort(np.random.default_rng(12345).permutation(len(val_pool))[: cfg.val_jets])

    ckpt_path = run_dir / "checkpoint.pt"
    state: dict[str, Any] = {"epoch": 0, "best": float("inf"), "bad": 0, "history": [], "best_state": None}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        state = ck["state"]
        log(f"resumed from epoch {state['epoch']} (best val {state['best']:.5g})")
    else:
        prep_rng = np.random.default_rng([cfg.seed, 10**6])
        model.prepare(
            b.to(device)
            for b in iterate_batches(
                train_pool,
                scaler,
                cfg.batch_size,
                shuffle=True,
                rng=prep_rng,
                limit=cfg.prepare_batches * cfg.batch_size,
            )
        )

    t_start = time.time()
    status = "done"
    while state["epoch"] < cfg.epochs and state.get("stop_reason") != "early_stop":
        ep = state["epoch"]
        t_ep = time.time()
        model.train()
        ep_rng = np.random.default_rng([cfg.seed, ep])
        tot, n = 0.0, 0
        for b in iterate_batches(train_pool, scaler, cfg.batch_size, shuffle=True, rng=ep_rng, limit=n_train):
            b = b.to(device)
            loss = model.losses(b)["total"].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(b), n + len(b)
        val = mean_loss(model, val_pool, scaler, val_idx)
        best = state["best"]
        improved = (not math.isfinite(best)) or val < best - abs(best) * cfg.min_delta
        if improved:
            state["best"], state["bad"] = val, 0
            state["best_state"] = copy.deepcopy(model.state_dict())
        else:
            state["bad"] += 1
        dt = time.time() - t_ep
        state["history"].append(
            {
                "epoch": ep,
                "train_loss": tot / max(n, 1),
                "val_loss": val,
                "lr": opt.param_groups[0]["lr"],
                "seconds": dt,
            }
        )
        state["epoch"] = ep + 1
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "state": state,
            },
            run_dir / "checkpoint.tmp",
        )
        (run_dir / "checkpoint.tmp").replace(ckpt_path)
        log(
            f"epoch {ep:3d}  train {tot / max(n, 1):.5g}  val {val:.5g}  {'*' if improved else ' '}  {dt:.0f}s"
        )
        if state["bad"] >= cfg.patience:
            log(f"early stopping after {state['bad']} epochs without improvement")
            state["stop_reason"] = "early_stop"
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "state": state,
                },
                run_dir / "checkpoint.tmp",
            )
            (run_dir / "checkpoint.tmp").replace(ckpt_path)
            break
        elapsed = time.time() - t_start
        if state["epoch"] < cfg.epochs and elapsed + dt > cfg.time_budget_min * 60:
            status = "paused"
            log(f"time budget reached after {elapsed / 60:.1f} min; rerun to resume")
            break

    summary = {
        "status": status,
        "epochs_done": state["epoch"],
        "best_val_loss": state["best"],
        "history": state["history"],
        "train_config": asdict(cfg),
    }
    if status == "done":
        model.load_state_dict(state["best_state"])
        # CPU tensors, so that GPU-trained weights load anywhere
        torch.save({k: v.detach().cpu() for k, v in state["best_state"].items()}, run_dir / "model.pt")
    save_json(summary, run_dir / "history.json")
    return summary


@torch.no_grad()
def score_pool(
    model: JetModel,
    pool: JetPool,
    scaler: Scaler,
    batch_size: int | None = None,
    with_embed: bool = False,
    keys: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Per-jet anomaly scores (and optionally graph latents and the extra per-jet loss terms ``keys``) for
    every jet in ``pool`` (row order kept).  ``batch_size`` defaults to the model's ``eval_batch_size``
    (1024; smaller for encoders with dense pairwise messages)."""
    model.eval()
    batch_size = batch_size or getattr(model, "eval_batch_size", 1024)
    dev = _model_device(model)
    scores, embeds = [], []
    extra: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for b in iterate_batches(pool, scaler, batch_size):
        b = b.to(dev)
        if keys:  # one forward pass for the score, the extra terms and (optionally) the latent
            out = model(b)
            res = model.losses(b, out)
            s = res["score"]
            for k in keys:
                extra[k].append(res[k].float().cpu().numpy())
            if with_embed:
                embeds.append(out["z"].float().cpu().numpy())
        elif with_embed:
            s, z = model.score_and_embed(b)
            if z is not None:
                embeds.append(z.float().cpu().numpy())
        else:
            s = model.score(b)
        scores.append(s.float().cpu().numpy())
    out_np = {"score": np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)}
    if with_embed and embeds:
        out_np["z"] = np.concatenate(embeds)
    for k, v in extra.items():
        out_np[k] = np.concatenate(v)
    return out_np
