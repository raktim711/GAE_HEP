"""Input versus reconstruction in the (deta, dphi) plane for the set / energy-flow autoencoders.

Marker area is proportional to the constituent's pT fraction (input) or to the decoder's weight
(reconstruction), so the pictures show what the anomaly score compares.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from gaehep.config import SR_MJJ
from gaehep.data.dataset import RECON_IDX, JetPool, Scaler, make_batch
from gaehep.eval import style
from gaehep.models import build_model

style.apply()
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def reconstruct(model, pool: JetPool, scaler: Scaler, idx: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    b = make_batch(pool, idx, scaler)
    out = model(b)
    score = model.losses(b, out)["score"].numpy()
    pt = (b.raw[..., 0] * b.mask).numpy()
    res = {
        "in_xy": b.coords.numpy(),
        "in_w": pt / pt.sum(1, keepdims=True),
        "mask": b.mask.numpy(),
        "score": score,
    }
    if "xy" in out:  # energy-flow AE: points + weights
        res["out_xy"], res["out_w"] = out["xy"].numpy(), out["w"].numpy()
    elif "recon" in out:  # set AE: standardised (deta, dphi, log pT) -> raw
        mean = scaler.node_mean[list(RECON_IDX)]
        std = scaler.node_std[list(RECON_IDX)]
        r = out["recon"].numpy() * std + mean
        n = b.n.clamp(max=r.shape[1]).numpy()
        w = np.exp(r[..., 2]) * (np.arange(r.shape[1])[None, :] < n[:, None])
        res["out_xy"], res["out_w"] = r[..., :2], w / w.sum(1, keepdims=True)
    else:
        raise ValueError("model has no set reconstruction")
    return res


def plot_reconstructions(rec: dict[str, np.ndarray], titles: list[str], path: Path, ncols: int = 4) -> Path:
    n = len(titles)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.8 * ncols, 4.1 * nrows), squeeze=False, layout="constrained"
    )
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        m = rec["mask"][i]
        ax.scatter(
            *rec["in_xy"][i, m].T, s=2000 * rec["in_w"][i, m], c="tab:blue", alpha=0.6, lw=0, label="input"
        )
        ax.scatter(
            *rec["out_xy"][i].T,
            s=2000 * rec["out_w"][i],
            facecolors="none",
            edgecolors="tab:red",
            lw=1.2,
            label="reconstruction",
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(f"{titles[i]}\nscore {rec['score'][i]:.3g}", fontsize=11)
        ax.set_xlabel(r"$\Delta\eta$")
        ax.set_ylabel(r"$\Delta\phi$")
        if i == 0:
            ax.legend(loc="lower left", fontsize=9, markerscale=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def recon_figure(run_dir: Path, n_per_class: int = 4, seed: int = 0) -> Path:
    """Heavier jet of random SR events: background (bkg_val) and 2-prong signal (sig_val)."""
    from gaehep.train.run import Data

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    data = Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    scaler = data.scaler()
    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    rng = np.random.default_rng(seed)
    st = data.store("rnd")
    picks, titles = [], []
    for split, lab in (("bkg_val", "QCD"), ("sig_val", "signal")):
        ev = data.events(split)
        sr = ev[(st["mjj"][ev] >= SR_MJJ[0]) & (st["mjj"][ev] <= SR_MJJ[1])]
        chosen = np.sort(rng.choice(sr, n_per_class, replace=False))
        heavy = np.argmax(st["m"][chosen], axis=1)
        pool = st.pool(chosen, data.n_const)
        rows = 2 * np.arange(n_per_class) + heavy  # event-major pool rows
        picks.append((pool, rows))
        titles += [f"{lab}, m$_j$ = {st['m'][e, h]:.0f} GeV" for e, h in zip(chosen, heavy, strict=True)]
    recs = [reconstruct(model, p, scaler, r) for p, r in picks]
    rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    return plot_reconstructions(rec, titles, run_dir / "figures" / "reconstructions.png")
