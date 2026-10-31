"""Score a trained run on an LHC Olympics 2020 black box and evaluate it (docs/STUDY_PROTOCOL.md, sections 5
and 7).

``score_sample(run, "bb1", subsample=True)`` (``gaehep score-sample <run> --sample bb1 --subsample``) scores
every event of the box's split with at least two jets (``bb<i>_test_sub`` with ``subsample``, else
``bb<i>_test``), resumably per chunk of events, and writes two files into the run directory:

``bb<i>_scores.npz``     ``jet_scores`` (n, 2) float32, ``event_index`` (n,) int64 (row in the box's event
                         table), ``mjj`` float32, ``label`` int8 and ``subsample`` (a bool scalar);
``bb<i>_metrics.json``   ``name, sample, subsample, splits_sha256, git, time, metrics``.

Model runs are scored with ``build_model`` and the run's ``model.pt``; baseline runs (a ``config.yaml`` whose
``model`` is a key of ``gaehep.baselines.SCORERS``: ``observable``, ``hl_density``, ``efp_knn``) with that
function.  The event score is the run's: ``prod`` = log F_B(s_1) + log F_B(s_2) with F_B the empirical CDF of
the jet scores on the run's ``bkg_val`` (for a run trained on BB1 that is BB1's validation part); labels never
enter F_B.

``metrics`` of BB1 and BB3 holds ``event``, ``jet`` and ``sculpting`` as in a run's ``metrics.json``.  Events
are keyed ``"<signal>/<region>"``: the signal is ``bb1``, or for BB3 ``bb3`` (both modes), ``bb3_dijet`` and
``bb3_trijet`` (the modes are the labels whose counts in the master key are 1200 and 2000), and the region is
``window`` (BB1 3.6-4.0 TeV, BB3 4.0-4.4 TeV) or ``full``.  Signal is judged against the background events of
the box in the same region.  BB2 is background only: its ``metrics`` hold ``ks`` (the Kolmogorov-Smirnov
distance of the box's score distribution to the R&D ``bkg_test`` one, per aggregation and for single jets) and
``bumphunt`` (the sideband-fit bump hunt, no injected signal, in the three BB2 windows).
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gaehep.config import (
    BB2_WINDOWS,
    BLACKBOX_SAMPLES,
    BLACKBOX_WINDOWS,
    run_metadata,
    save_json,
)
from gaehep.eval.bumphunt import search
from gaehep.eval.evaluate import AGGREGATIONS, aggregate, jet_reference, region_mask
from gaehep.eval.metrics import DEFAULT_MIN_BKG, bootstrap, summarize
from gaehep.eval.sculpting import sculpting_jsd

CHUNK = 50_000  # events per scoring chunk
# BB3's two signal modes, identified by their event counts in the master key
BB3_MODE_COUNTS: dict[str, int] = {"dijet": 1200, "trijet": 2000}
BASELINE_MODELS = (
    "observable",
    "hl_density",
    "efp_knn",
)  # `model` values of baseline runs (protocol, section 6)
BUMPHUNT_EFFS = (1.0, 0.1, 0.02)  # no cut, and cuts keeping 10 % and 2 % of the sideband events


# ----------------------------------------------------------------------------------------- signals
def mode_labels(label: np.ndarray) -> dict[str, int]:
    """BB3: the master-key label (1 or 2) of the ``dijet`` and ``trijet`` modes, from the label counts over
    the whole box (:data:`BB3_MODE_COUNTS`)."""
    counts = {lab: int(np.sum(label == lab)) for lab in (1, 2)}
    out = {}
    for mode, n in BB3_MODE_COUNTS.items():
        found = [lab for lab, c in counts.items() if c == n]
        if len(found) != 1:
            raise ValueError(
                f"BB3 modes cannot be told apart by their counts: {counts}, expected {BB3_MODE_COUNTS}"
            )
        out[mode] = found[0]
    return out


def signal_masks(sample: str, label: np.ndarray, label_all: np.ndarray) -> dict[str, np.ndarray]:
    """Signal name -> mask of its events among ``label`` (the scored events).  ``label_all`` are the labels of
    the whole box, which identify the BB3 modes."""
    if sample == "bb2":
        return {}
    sig = {sample: label > 0}
    if sample == "bb3":
        modes = mode_labels(label_all)
        sig.update({f"bb3_{mode}": label == lab for mode, lab in modes.items()})
    return sig


# ----------------------------------------------------------------------------------------- metrics
def _summarize(sig: np.ndarray, bkg: np.ndarray, min_bkg: int) -> dict[str, float]:
    """``summarize``, or NaNs with the counts if a class is empty (e.g. no signal in the window)."""
    if len(sig) == 0 or len(bkg) == 0:
        keys = summarize(np.array([1.0, 0.0]), np.array([0.0, 1.0]), min_bkg=1)
        return {k: float("nan") for k in keys} | {"n_sig": len(sig), "n_bkg": len(bkg)}
    return summarize(sig, bkg, min_bkg)


def evaluate_signal_box(
    sample: str,
    jet_scores: np.ndarray,
    mjj: np.ndarray,
    label: np.ndarray,
    ref: np.ndarray,
    label_all: np.ndarray | None = None,
    m: np.ndarray | None = None,
    n_boot: int = 30,
    min_bkg: int = DEFAULT_MIN_BKG,
) -> dict[str, Any]:
    """``event`` / ``jet`` / ``sculpting`` metrics of BB1 or BB3 (see the module docstring).  ``ref``: sorted
    background jet scores (F_B); ``m``: (n, 2) jet masses for the sculpting of the jet mass (optional)."""
    window = BLACKBOX_WINDOWS[sample]
    sigs = signal_masks(sample, label, label if label_all is None else label_all)
    bkg = label == 0
    res: dict[str, Any] = {"event": {}, "jet": {}, "sculpting": {}}
    for agg in AGGREGATIONS:
        ev = aggregate(jet_scores, agg, ref)
        res["event"][agg] = {}
        for name, sm in sigs.items():
            for region in ("window", "full"):
                rm = region_mask(mjj, region, window)
                entry = _summarize(ev[sm & rm], ev[bkg & rm], min_bkg)
                if n_boot and agg in ("max", "prod") and entry["n_sig"] and entry["n_bkg"]:
                    entry["bootstrap"] = bootstrap(ev[sm & rm], ev[bkg & rm], n_boot=n_boot, min_bkg=min_bkg)
                res["event"][agg][f"{name}/{region}"] = entry
        res["sculpting"][agg] = {"mjj": sculpting_jsd(ev[bkg], mjj[bkg])}
        if m is not None:
            res["sculpting"][agg]["mj_max"] = sculpting_jsd(ev[bkg], np.nanmax(m[bkg], axis=1))
    for name, sm in sigs.items():
        res["jet"][name] = _summarize(jet_scores[sm].ravel(), jet_scores[bkg].ravel(), min_bkg)
    return res


def evaluate_null_box(
    jet_scores: np.ndarray,
    mjj: np.ndarray,
    ref: np.ndarray,
    rnd_jet_scores: np.ndarray,
    windows: tuple[tuple[float, float], ...] = BB2_WINDOWS,
    effs: tuple[float, ...] = BUMPHUNT_EFFS,
) -> dict[str, Any]:
    """BB2 (background only): ``ks`` and ``bumphunt`` (see the module docstring).  ``rnd_jet_scores``: the
    run's R&D ``bkg_test`` jet scores.  The bump hunt is :func:`gaehep.eval.bumphunt.search` on the ``prod``
    event score with a sideband-defined cut keeping a fraction ``eff`` of the sideband events."""
    from scipy.stats import ks_2samp

    ks = {
        agg: float(ks_2samp(aggregate(jet_scores, agg, ref), aggregate(rnd_jet_scores, agg, ref)).statistic)
        for agg in AGGREGATIONS
    }
    ks["jet"] = float(ks_2samp(jet_scores.ravel(), rnd_jet_scores.ravel()).statistic)
    score = aggregate(jet_scores, "prod", ref)
    rows = []
    for win in windows:
        for eff in effs:
            r = search(mjj.astype(np.float64), score, eff, sr=win)
            rows.append({"window": list(win), "agg": "prod", "n_inj": 0, **r})
    return {"ks": ks, "bumphunt": rows}


# ----------------------------------------------------------------------------------------- scoring
def baseline_scorers() -> Mapping[str, Callable[..., np.ndarray]]:
    """``gaehep.baselines.SCORERS``, or {} while that module (written in parallel) does not exist."""
    try:
        from gaehep.baselines import SCORERS
    except ModuleNotFoundError as err:
        if err.name != "gaehep.baselines":
            raise
        return {}
    return SCORERS


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def _chunked_scores(
    events: np.ndarray,
    chunk: int,
    part_dir: Path,
    signature: str,
    score_chunk: Callable[[np.ndarray], np.ndarray],
    log,
) -> np.ndarray:
    """(n, 2) scores of ``events``, scored ``chunk`` events at a time with every finished chunk kept in
    ``part_dir`` so that a restarted job resumes.  ``signature`` identifies what produced the chunks."""
    part_dir.mkdir(parents=True, exist_ok=True)
    meta = part_dir / "signature.txt"
    sig = f"{signature} n={len(events)} chunk={chunk} first={events[0] if len(events) else -1} last={events[-1] if len(events) else -1}"
    if meta.exists() and meta.read_text() != sig:
        for f in part_dir.glob("chunk_*.npy"):
            f.unlink()
    meta.write_text(sig)
    out = []
    n_chunks = -(-len(events) // chunk)
    for i in range(n_chunks):
        path = part_dir / f"chunk_{i:05d}.npy"
        if path.exists():
            out.append(np.load(path))
            continue
        s = np.asarray(score_chunk(events[i * chunk : (i + 1) * chunk]), dtype=np.float32).reshape(-1, 2)
        _save_npy(path, s)
        out.append(s)
        log(f"  chunk {i + 1}/{n_chunks} scored ({min((i + 1) * chunk, len(events))}/{len(events)} events)")
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)


def _up_to_date(path: Path, events: np.ndarray, subsample: bool, deps: list[Path]) -> bool:
    """The saved scores belong to these events and are newer than the run's model / scores."""
    if not path.exists():
        return False
    try:
        with np.load(path) as f:
            same = bool(f["subsample"]) == subsample and np.array_equal(f["event_index"], events)
    except (OSError, KeyError, ValueError):
        return False
    return same and all(path.stat().st_mtime_ns >= d.stat().st_mtime_ns for d in deps if d.exists())


def score_sample(
    run: str,
    sample: str,
    subsample: bool = False,
    chunk: int = CHUNK,
    n_boot: int = 30,
    rescore: bool = False,
    data=None,
    log=print,
) -> dict[str, Any]:
    """Score run ``run`` on black box ``sample`` and write ``bb<i>_scores.npz`` / ``bb<i>_metrics.json`` into
    its run directory (see the module docstring).  Returns the contents of the metrics file."""
    from gaehep.train.run import Data, _run_dir

    if sample not in BLACKBOX_SAMPLES:
        raise ValueError(f"sample must be one of {BLACKBOX_SAMPLES}, not {sample!r}")
    run_dir = _run_dir(run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    data = data or Data.from_config(cfg.get("data"))
    split = f"{sample}_test_sub" if subsample else f"{sample}_test"
    if not data.has(split):
        raise FileNotFoundError(f"no split {split!r} in {data.root}: run `gaehep splits --sample {sample}`")
    events = data.events(split)
    store = data.store(sample)
    out_npz = run_dir / f"{sample}_scores.npz"
    tag = f"{sample}{'_sub' if subsample else ''}"

    model_name = cfg["model"]
    scorers = baseline_scorers()
    if model_name in BASELINE_MODELS and model_name not in scorers:
        raise RuntimeError(
            f"{run} is a {model_name!r} baseline run, but gaehep.baselines.SCORERS does not provide it"
        )
    if model_name in scorers:
        deps = [run_dir / "scores.npz"]
    else:
        deps = [run_dir / "model.pt"]
    log(f"[{run}] {split}: {len(events)} events")
    if not rescore and _up_to_date(out_npz, events, subsample, deps):
        log(f"[{run}] {out_npz.name} is up to date; recomputing the metrics only")
        with np.load(out_npz) as f:
            jet_scores = f["jet_scores"]
    elif model_name in scorers:  # a baseline scores the whole split in one call
        jet_scores = np.asarray(scorers[model_name](run_dir, data, split), dtype=np.float32)
        if jet_scores.shape != (len(events), 2):
            raise ValueError(
                f"baseline {model_name!r} returned shape {jet_scores.shape}, expected {(len(events), 2)}"
            )
    else:
        jet_scores = _score_model(run_dir, cfg, data, sample, events, chunk, run_dir / f"_partial_{tag}", log)
    jet_scores = np.asarray(jet_scores, dtype=np.float32)
    if not np.isfinite(jet_scores).all():
        raise ValueError(f"{run}: non-finite scores on {split}")

    label_all = store["label"]
    mjj = store["mjj"][events].astype(np.float32)
    label = label_all[events].astype(np.int8)
    tmp = out_npz.with_name(out_npz.stem + ".tmp.npz")
    np.savez(
        tmp,
        jet_scores=jet_scores,
        event_index=events.astype(np.int64),
        mjj=mjj,
        label=label,
        subsample=np.bool_(subsample),
    )
    tmp.replace(out_npz)
    shutil.rmtree(run_dir / f"_partial_{tag}", ignore_errors=True)  # only needed to resume an interrupted job

    with np.load(run_dir / "scores.npz") as f:
        if "bkg_val" not in f.files:
            raise KeyError(
                f"{run_dir / 'scores.npz'} has no 'bkg_val' scores (F_B): score the run's splits first"
            )
        ref = jet_reference({"bkg_val": f["bkg_val"]})
        rnd_bkg_test = f["bkg_test"] if "bkg_test" in f.files else None
    if sample == "bb2":
        if rnd_bkg_test is None:
            raise KeyError(
                f"{run_dir / 'scores.npz'} has no 'bkg_test' scores (reference of the KS distance)"
            )
        metrics = evaluate_null_box(jet_scores, mjj, ref, rnd_bkg_test)
    else:
        metrics = evaluate_signal_box(
            sample, jet_scores, mjj, label, ref, label_all=label_all, m=store["m"][events], n_boot=n_boot
        )
    meta = run_metadata()
    out = {
        "name": run,
        "sample": sample,
        "subsample": bool(subsample),
        "splits_sha256": meta.get("splits_sha256"),
        "git": meta["git"],
        "time": meta["time"],
        "metrics": metrics,
    }
    save_json(out, run_dir / f"{sample}_metrics.json")
    _log_summary(log, run, sample, metrics)
    return out


def _log_summary(log, run: str, sample: str, metrics: Mapping[str, Any]) -> None:
    if sample == "bb2":
        log(
            f"[{run}] bb2 KS distance to R&D bkg_test: "
            + ", ".join(f"{k} {v:.4f}" for k, v in metrics["ks"].items())
        )
        return
    for key, e in metrics["event"]["prod"].items():
        log(f"[{run}] {sample} prod {key}: {e['n_sig']} sig / {e['n_bkg']} bkg  AUC {e['auc']:.4f}")


def _score_model(
    run_dir: Path, cfg: dict[str, Any], data, sample: str, events: np.ndarray, chunk: int, part_dir: Path, log
) -> np.ndarray:
    import torch

    from gaehep.models import build_model
    from gaehep.train.loop import TrainConfig, resolve_device, score_pool

    model = build_model(cfg["model"], **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True, map_location="cpu"))
    tcfg = TrainConfig.from_dict(cfg.get("train"))
    torch.set_num_threads(tcfg.threads)
    model.to(resolve_device(tcfg.device))
    scaler = data.scaler()
    store = data.store(sample)

    def score_chunk(ev: np.ndarray) -> np.ndarray:
        pool = store.pool(ev, data.n_const)
        return score_pool(model, pool, scaler)["score"].reshape(-1, 2)

    signature = f"model {_file_sha256(run_dir / 'model.pt')}"
    return _chunked_scores(events, chunk, part_dir, signature, score_chunk, log)
