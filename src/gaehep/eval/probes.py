"""What do the jet latents know?  Linear and MLP regressions of jet observables on a run's latents.

``gaehep probes <run>`` reads ``<results>/runs/<run>/latents.npz`` and uses the latents of the 20 000 jets
(``2 * n_events``, jet-pair order) of the first 10 000 ``bkg_val`` events.  Targets, standardised inside each
training fold: m_J, pT_J, tau21, tau32, n_const (the jet tables) and the wedge, triangle and square EFPs (hadronic
measure, beta = 1, from the stored constituents; ``gaehep.eval.efp``).  Jets with an undefined tau21 or tau32
are left out of that target's regression.

Models (inputs standardised inside each fold): ridge regression (alpha = 1) and an MLP with hidden layers
(64, 64), ``random_state=20201``.  5-fold ``KFold(shuffle=True, random_state=20201)``.  ``probes.json`` in the run
directory holds, per target and model, the R-squared on the held-out fold, its mean and its sample standard
deviation (ddof = 1) over the folds.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gaehep.config import results_dir, run_metadata, save_json

SEED = 20201
N_EVENTS = 10_000
N_FOLDS = 5
TARGETS = ("m_j", "pt_j", "tau21", "tau32", "n_const", "efp_wedge", "efp_triangle", "efp_square")


def jet_targets(tab: dict[str, np.ndarray], const: np.ndarray) -> dict[str, np.ndarray]:
    """Probe targets of the jets of the events in ``tab`` (event-major, jets 0 and 1) with constituents
    ``const`` (2 * n_events, N, 3); NaN where undefined."""
    from gaehep.eval.efp import probe_efps

    with np.errstate(invalid="ignore", divide="ignore"):
        t1, t2, t3 = (np.asarray(tab[k], dtype=np.float64).ravel() for k in ("tau1", "tau2", "tau3"))
        tau21 = np.where(t1 > 0, t2 / t1, np.nan)
        tau32 = np.where(t2 > 0, t3 / t2, np.nan)
    out = {
        "m_j": np.asarray(tab["m"], dtype=np.float64).ravel(),
        "pt_j": np.asarray(tab["pt"], dtype=np.float64).ravel(),
        "tau21": tau21,
        "tau32": tau32,
        "n_const": np.asarray(tab["n_const"], dtype=np.float64).ravel(),
    }
    out.update(probe_efps(const))
    return out


def _models() -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), random_state=SEED)),
    }


def cv_r2(z: np.ndarray, y: np.ndarray, model_name: str, n_folds: int = N_FOLDS) -> list[float]:
    """Held-out R-squared of ``y`` regressed on ``z`` in each of the folds (targets standardised per fold)."""
    folds = []
    for tr, te in KFold(n_folds, shuffle=True, random_state=SEED).split(z):
        reg = TransformedTargetRegressor(regressor=_models()[model_name], transformer=StandardScaler())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            reg.fit(z[tr], y[tr])
        folds.append(float(r2_score(y[te], reg.predict(z[te]))))
    return folds


def probe(
    name: str, data=None, n_events: int = N_EVENTS, n_folds: int = N_FOLDS, log=print
) -> dict[str, Any]:
    """Run the probes on the latents of run ``name`` and write ``probes.json`` into its directory."""
    from gaehep.train.run import SPLIT_SAMPLE, Data

    run_dir: Path = results_dir() / "runs" / name
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) if (run_dir / "config.yaml").exists() else {}
    with np.load(run_dir / "latents.npz") as f:
        lat = f["bkg_val"][:n_events]
    n_events = len(lat)
    z = lat.reshape(2 * n_events, -1).astype(np.float64)
    data = data or Data(n_const=int(cfg.get("data", {}).get("n_const", 50)))
    events = data.events("bkg_val")[:n_events]
    store = data.store(SPLIT_SAMPLE["bkg_val"])
    tab = {k: store[k][events] for k in ("m", "pt", "n_const", "tau1", "tau2", "tau3")}
    targets = jet_targets(tab, store.pool(events, data.n_const).const)
    res: dict[str, Any] = {}
    for t in TARGETS:
        ok = np.isfinite(targets[t])
        res[t] = {"n_jets": int(ok.sum())}
        for m in ("ridge", "mlp"):
            folds = cv_r2(z[ok], targets[t][ok], m, n_folds)
            res[t][m] = {
                "r2_mean": float(np.mean(folds)),
                "r2_std": float(np.std(folds, ddof=1)) if len(folds) > 1 else float("nan"),
                "r2_folds": folds,
            }
        log(f"[{name}] {t:13s} R2 ridge {res[t]['ridge']['r2_mean']:.3f}  mlp {res[t]['mlp']['r2_mean']:.3f}")
    out = {
        "name": name,
        "n_events": n_events,
        "n_jets": 2 * n_events,
        "latent_dim": z.shape[1],
        "n_folds": n_folds,
        "std_ddof": 1,
        "targets": res,
        **run_metadata(),
    }
    save_json(out, run_dir / "probes.json")
    return out
