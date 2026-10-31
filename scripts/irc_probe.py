"""How much do the saved models' anomaly scores move under collinear splittings and soft emissions?

Usage: python scripts/irc_probe.py <run> [<run> ...] [--out FILE]
(runs are <results>/runs/<name>: baseline hp_final_s1, EWMP lt_ewmp_s1; <results> = GAEHEP_RESULTS, default
<repo>/results; the jets are read from GAEHEP_DATA).  Jet-level SR AUC (background-validation vs 2-prong
signal-validation jets) before/after, and the Spearman rank correlation of the background scores before/after;
the numbers are merged into ``--out`` (default <results>/irc_probe.json).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from gaehep.config import results_dir
from gaehep.data.dataset import JetPool
from gaehep.models import build_model
from gaehep.train.loop import score_pool
from gaehep.train.run import Data

torch.set_num_threads(1)
SR = (3300.0, 3700.0)


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


def main(names: list[str], out: Path | None = None) -> None:
    runs = results_dir() / "runs"
    out = out or results_dir() / "irc_probe.json"
    data = Data(n_const=50)
    scaler = data.scaler()
    bkg = data.pool("bkg_val", 8000, mjj=SR)
    sig = data.pool("sig_val", 8000, mjj=SR)
    res: dict[str, dict] = {}
    for name in names:
        cfg = yaml.safe_load((runs / name / "config.yaml").read_text())
        model = build_model(cfg["model"], **cfg.get("model_args", {}))
        model.load_state_dict(torch.load(runs / name / "model.pt", weights_only=True, map_location="cpu"))
        model.eval()

        def sc(p, model=model):
            return score_pool(model, p, scaler, batch_size=128)["score"]

        s0b, s0s = sc(bkg), sc(sig)
        y = np.r_[np.zeros(len(s0b)), np.ones(len(s0s))]
        base = roc_auc_score(y, np.r_[s0b, s0s])
        res[name] = {"auc": float(base)}
        print(f"{name:14s} unperturbed jet AUC {base:.3f}")
        for label, fn in [
            ("split top-1", lambda p: split_top(p, 1)),
            ("split top-5", lambda p: split_top(p, 5)),
            ("split all", lambda p: split_top(p, None)),
            ("10 soft (0.2% pT each)", lambda p: add_soft(p, 10, 0.002)),
            ("10 soft (1% pT each)", lambda p: add_soft(p, 10, 0.01)),
        ]:
            sb, ss = sc(fn(bkg)), sc(fn(sig))
            auc = roc_auc_score(y, np.r_[sb, ss])
            rho = spearmanr(s0b, sb).statistic
            thr0, thr1 = np.quantile(s0b, 0.99), np.quantile(sb, 0.99)
            flip = float(((s0b > thr0) != (sb > thr1)).mean() * 100)
            res[name][label] = {"auc": float(auc), "rho": float(rho), "flip_pct_at_1pct": flip}
            print(
                f"   {label:24s} AUC {auc:.3f}  (d = {auc - base:+.3f})  Spearman(bkg) {rho:.4f}  top-1% membership flips {flip:.2f}% of jets"
            )
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prev, indent=1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", metavar="RUN", help="run name under <results>/runs")
    p.add_argument(
        "--out",
        type=Path,
        default=results_dir() / "irc_probe.json",
        help="JSON file the results are merged into (default: %(default)s)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args.runs, args.out)
