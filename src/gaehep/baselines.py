"""Non-graph baselines of the study (docs/STUDY_PROTOCOL.md, section 6) and their scorer registry.

Three baselines give a per-jet score (larger = more anomalous) without a graph model:

* ``observable`` - one of m_J, n_const, -tau21, -tau32 (``gaehep obs-baselines``; runs ``study_obs_*``);
* ``hl_density`` - -log p of a zuko NSF fitted to the standardised jet observables
  (log m_J, log pT_J, tau21, tau32, log n_const) of ``bkg_train`` (``gaehep hl-density --seed s``;
  runs ``study_hldensity_s<s>``);
* ``efp_knn`` - the mean distance to the k nearest of a fixed sample of background-training jets in
  standardised energy-flow-polynomial space (``gaehep efp-knn``; run ``study_efpknn``).

Each run directory holds ``config.yaml`` (with ``model``), ``scores.npz`` (``bkg_val``, ``bkg_test``,
``sig_test``, ``sig3_test``) and ``metrics.json``, like a model run.  ``SCORERS`` maps a config's ``model`` to a
function ``(run_dir, data, split) -> (n_events, 2) float64`` that scores any split ``Data`` knows, which is how
``gaehep score-sample`` scores them on the black boxes.  All three only read jet tables and EFPs, so they are
invariant to the order of the constituents and to zero-pT padding (EFPs are symmetric sums with pT weights).

Choices the protocol leaves open, fixed here:

* hl-density: a jet whose tau21 or tau32 is undefined (missing or zero tau) is left out of the training set;
  when it is scored, its undefined features take the training mean (0 after standardisation).  m_J and n_const
  are floored at 1 inside the logarithm, as in ``dataset.jet_level_features``.
* efp-knn: log(max(EFP, 1e-12)), then standardisation with the reference set's mean and standard deviation;
  Euclidean distances in float64.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from gaehep.config import load_json, results_dir, save_json
from gaehep.eval.evaluate import observable_scores
from gaehep.models.latent_flow import LatentFlow

if TYPE_CHECKING:  # pragma: no cover
    from gaehep.train.run import Data

Scorer = Callable[[Path, "Data", str], np.ndarray]

# study run name -> key of evaluate.observable_scores
OBSERVABLES = {
    "study_obs_mj": "m_j",
    "study_obs_nconst": "n_const",
    "study_obs_tau21": "-tau21",
    "study_obs_tau32": "-tau32",
}
HL_FEATURES = ("log_m", "log_pt", "tau21", "tau32", "log_n_const")
FLOW_TRANSFORMS, FLOW_HIDDEN = 4, (128, 128)  # LatentFlow's defaults
FLOW_FIT = {"epochs": 60, "batch_size": 1024, "lr": 1e-3, "patience": 8}  # LatentFlow.fit's defaults
KNN_K, KNN_N_REF, KNN_REF_SEED = 10, 100_000, 20201
N_CONST = 50
REFERENCE_FILE = "reference.npz"


def _run_dir(name: str) -> Path:
    return results_dir() / "runs" / name


def _config(run_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((Path(run_dir) / "config.yaml").read_text())


def _pairs(x: np.ndarray, n_events: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(n_events, 2)


# ---------------------------------------------------------------------------------------- observables
def observable_scorer(run_dir: Path, data: Data, split: str) -> np.ndarray:
    """Scores of the single observable named in the run's config (``observable``)."""
    obs = _config(run_dir)["observable"]
    tab = data.table(split)
    return _pairs(observable_scores(tab)[obs], len(tab["m"]))


def run_obs_baselines(data: Data | None = None, n_boot: int = 30, log=print) -> dict[str, dict[str, Any]]:
    """The four single-observable baselines as run directories ``study_obs_*``."""
    from gaehep.train.run import SCORE_SPLITS, Data, evaluate_run

    data = data or Data(n_const=N_CONST)
    out = {}
    for name, obs in OBSERVABLES.items():
        run_dir = _run_dir(name)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"model": "observable", "observable": obs, "data": {"n_const": data.n_const}}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        scores = {
            s: observable_scorer(run_dir, data, s).astype(np.float32) for s in SCORE_SPLITS if data.has(s)
        }
        np.savez(run_dir / "scores.npz", **scores)
        out[name] = evaluate_run(name, data=data, cfg=cfg, scores=scores, n_boot=n_boot)
        log(f"[{name}] {obs}: scored {', '.join(scores)}")
    return out


# ------------------------------------------------------------------------------------- hl-density flow
def hl_features(tab: Mapping[str, np.ndarray]) -> np.ndarray:
    """(n_events, 2, 5) float64 (log m_J, log pT_J, tau21, tau32, log n_const); NaN where undefined."""
    m, pt, n = (np.asarray(tab[k], dtype=np.float64) for k in ("m", "pt", "n_const"))
    t1, t2, t3 = (np.asarray(tab[k], dtype=np.float64) for k in ("tau1", "tau2", "tau3"))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau21 = np.where(t1 > 0, t2 / t1, np.nan)
        tau32 = np.where(t2 > 0, t3 / t2, np.nan)
        log_pt = np.log(np.where(pt > 0, pt, np.nan))
    feats = np.stack([np.log(np.maximum(m, 1.0)), log_pt, tau21, tau32, np.log(np.maximum(n, 1.0))], axis=-1)
    return np.where(np.isfinite(feats), feats, np.nan)


def _save_flow(flow: LatentFlow, fill: np.ndarray, path: Path) -> None:
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in flow.flow.state_dict().items()},
            "mean": torch.from_numpy(flow.mean),
            "std": torch.from_numpy(flow.std),
            "fill": torch.from_numpy(np.asarray(fill, dtype=np.float32)),
            "features": list(HL_FEATURES),
            "transforms": FLOW_TRANSFORMS,
            "hidden": list(FLOW_HIDDEN),
        },
        path,
    )


def _load_flow(path: Path) -> tuple[LatentFlow, np.ndarray]:
    d = torch.load(path, weights_only=True, map_location="cpu")
    flow = LatentFlow(len(d["features"]), transforms=int(d["transforms"]), hidden=tuple(d["hidden"]))
    flow.flow.load_state_dict(d["state_dict"])
    flow.mean, flow.std = d["mean"].numpy(), d["std"].numpy()
    return flow, d["fill"].numpy().astype(np.float64)


def hl_nll(flow: LatentFlow, fill: np.ndarray, tab: Mapping[str, np.ndarray]) -> np.ndarray:
    """(n_events, 2) float64 -log p; undefined features are replaced by the training means ``fill``."""
    feats = hl_features(tab)
    n = feats.shape[0]
    x = np.where(np.isfinite(feats), feats, fill).reshape(-1, feats.shape[-1])
    return _pairs(flow.nll(x.astype(np.float32)), n)


def hl_density_scorer(run_dir: Path, data: Data, split: str) -> np.ndarray:
    flow, fill = _load_flow(Path(run_dir) / "model.pt")
    return hl_nll(flow, fill, data.table(split))


def run_hl_density(
    seed: int,
    data: Data | None = None,
    name: str | None = None,
    epochs: int = FLOW_FIT["epochs"],
    threads: int = 1,
    n_boot: int = 30,
    log=print,
) -> dict[str, Any]:
    """Fit the jet-observable flow on ``bkg_train`` (early stopping on the ``bkg_val`` NLL), score the test
    splits and write the run directory ``study_hldensity_s<seed>`` (a finished run is only rescored)."""
    from gaehep.train.run import SCORE_SPLITS, Data, evaluate_run

    name = name or f"study_hldensity_s{seed}"
    run_dir = _run_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = data or Data(n_const=N_CONST)
    torch.set_num_threads(threads)
    cfg = {
        "model": "hl_density",
        "seed": seed,
        "features": list(HL_FEATURES),
        "flow": {"transforms": FLOW_TRANSFORMS, "hidden": list(FLOW_HIDDEN)},
        "fit": {**FLOW_FIT, "epochs": epochs, "seed": seed},
        "data": {"n_const": data.n_const},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    hist_path = run_dir / "history.json"
    if (
        (run_dir / "model.pt").exists()
        and hist_path.exists()
        and load_json(hist_path).get("status") == "done"
    ):
        log(f"[{name}] already trained; rescoring")
    else:
        t0 = time.time()
        train, val = (
            hl_features(data.table("bkg_train")).reshape(-1, 5),
            hl_features(data.table("bkg_val")).reshape(-1, 5),
        )
        train, val = train[np.isfinite(train).all(1)], val[np.isfinite(val).all(1)]
        flow = LatentFlow(len(HL_FEATURES), transforms=FLOW_TRANSFORMS, hidden=FLOW_HIDDEN, seed=seed)
        flow.fit(
            train.astype(np.float32), val.astype(np.float32), seed=seed, **{**FLOW_FIT, "epochs": epochs}
        )
        best = float(flow.nll(val.astype(np.float32)).mean())  # the restored best weights
        _save_flow(flow, train.mean(0), run_dir / "model.pt")
        save_json(
            {
                "status": "done",
                "epochs_done": len(flow.history["val_nll"]),
                "best_val_loss": best,
                "history": [
                    {"epoch": i + 1, "train_loss": a, "val_loss": b}
                    for i, (a, b) in enumerate(
                        zip(flow.history["train_nll"], flow.history["val_nll"], strict=True)
                    )
                ],
                "n_train_jets": len(train),
                "n_val_jets": len(val),
                "train_minutes": (time.time() - t0) / 60,
            },
            hist_path,
        )
        log(f"[{name}] flow fitted: {len(flow.history['val_nll'])} epochs, best val NLL {best:.4f}")
    scores = {}
    for s in SCORE_SPLITS:
        if data.has(s):
            scores[s] = hl_density_scorer(run_dir, data, s).astype(np.float32)
    np.savez(run_dir / "scores.npz", **scores)
    return evaluate_run(name, data=data, cfg=cfg, scores=scores, n_boot=n_boot)


# ------------------------------------------------------------------------------------------ EFP k-NN
def efp_features(const: np.ndarray) -> np.ndarray:
    """log-EFPs of the prime graphs with d <= 4 for constituents (n, N, 3): (n, 20) float64."""
    from gaehep.eval.efp import log_efps, prime_efps

    return log_efps(prime_efps(const))


def draw_reference_jets(data: Data, n_ref: int, seed: int) -> np.ndarray:
    """Constituents ``(n_ref, N, 3)`` of ``n_ref`` ``bkg_train`` jets drawn without replacement with
    ``default_rng(seed)`` from all jets of the split (event-major order: event e, jet j -> 2e + j)."""
    events = data.events("bkg_train")
    n_jets = 2 * len(events)
    if n_ref > n_jets:
        raise ValueError(f"reference set of {n_ref} jets, but bkg_train has {n_jets}")
    flat = np.sort(np.random.default_rng(seed).choice(n_jets, n_ref, replace=False))
    ev, jt = events[flat // 2], flat % 2
    const = data.store("rnd").const
    out = np.empty((n_ref, data.n_const, 3), dtype=np.float32)
    for s in range(0, n_ref, 50_000):
        out[s : s + 50_000] = const[ev[s : s + 50_000], jt[s : s + 50_000], : data.n_const, :]
    return out


def standardise_reference(feats: np.ndarray) -> dict[str, np.ndarray]:
    mean, std = feats.mean(0), np.maximum(feats.std(0), 1e-6)
    return {"ref": (feats - mean) / std, "mean": mean, "std": std}


def knn_mean_distance(queries: np.ndarray, ref: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    """Mean Euclidean distance of every query to its k nearest rows of ``ref`` (float64, ``(n,)``).

    The neighbours are found with the fast matrix-product distances; their distances are then recomputed
    from the coordinate differences, so that exact matches give exactly 0 (the product form has ~1e-8 noise)."""
    r = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float64))
    q = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float64))
    if k > len(r):
        raise ValueError(f"k = {k} neighbours from a reference set of {len(r)}")
    out = np.empty(len(q), dtype=np.float64)
    for s in range(0, len(q), chunk):
        qc = q[s : s + chunk]
        idx = torch.topk(torch.cdist(qc, r), k, dim=1, largest=False).indices
        out[s : s + chunk] = (qc[:, None, :] - r[idx]).norm(dim=-1).mean(1).numpy()
    return out


def _reference(run_dir: Path, data: Data, cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(run_dir) / REFERENCE_FILE
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    ref = standardise_reference(
        efp_features(draw_reference_jets(data, int(cfg["n_ref"]), int(cfg["ref_seed"])))
    )
    np.savez(path.with_suffix(".tmp.npz"), **ref)
    path.with_suffix(".tmp.npz").replace(path)
    return ref


def efp_knn_scorer(run_dir: Path, data: Data, split: str) -> np.ndarray:
    """Mean distance to the k nearest reference jets; the reference set is cached in ``reference.npz``
    of the run directory and rebuilt from ``bkg_train`` when that file is missing."""
    cfg = _config(run_dir)
    n_const = int(cfg.get("data", {}).get("n_const", N_CONST))
    if data.n_const != n_const:
        raise ValueError(
            f"efp_knn run {run_dir} was built with {n_const} constituents, Data has {data.n_const}"
        )
    ref = _reference(run_dir, data, cfg)
    pool = data.pool(split)
    x = (efp_features(pool.const) - ref["mean"]) / ref["std"]
    n_events = len(pool) // 2
    data.release(split)
    return _pairs(knn_mean_distance(x, ref["ref"], int(cfg["k"])), n_events)


def run_efp_knn(
    data: Data | None = None,
    name: str = "study_efpknn",
    k: int = KNN_K,
    n_ref: int = KNN_N_REF,
    ref_seed: int = KNN_REF_SEED,
    threads: int = 1,
    n_boot: int = 30,
    log=print,
) -> dict[str, Any]:
    """Training-free EFP k-NN baseline: write the run directory ``study_efpknn``."""
    from gaehep.eval.efp import PRIME_SET
    from gaehep.train.run import SCORE_SPLITS, Data, evaluate_run

    data = data or Data(n_const=N_CONST)
    torch.set_num_threads(threads)
    run_dir = _run_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model": "efp_knn",
        "k": k,
        "n_ref": n_ref,
        "ref_seed": ref_seed,
        "efp": {"set": list(PRIME_SET), "measure": "hadr", "beta": 1, "normed": True, "log_floor": 1e-12},
        "data": {"n_const": data.n_const},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    scores = {}
    for s in SCORE_SPLITS:
        if data.has(s):
            scores[s] = efp_knn_scorer(run_dir, data, s).astype(np.float32)
            log(f"[{name}] scored {s}: {scores[s].shape[0]} events")
    np.savez(run_dir / "scores.npz", **scores)
    return evaluate_run(name, data=data, cfg=cfg, scores=scores, n_boot=n_boot)


SCORERS: dict[str, Scorer] = {
    "observable": observable_scorer,
    "hl_density": hl_density_scorer,
    "efp_knn": efp_knn_scorer,
}
