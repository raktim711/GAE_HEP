"""Metrics and figures for arbitrary per-event anomaly scores, independent of the LHCO tables of a run.

``evaluate_score_arrays(score, label, mjj)`` reports, for signal (``label == 1``) vs background
(``label == 0``):

* ``full``: AUC, max SIC (cuts keeping >= ``min_bkg`` background events) and its signal efficiency,
  1/eps_B and SIC at eps_S = 0.2 / 0.3 / 0.5 (``metrics.summarize``), plus bootstrap mean / std / 16-84%
  interval of AUC and max SIC (``metrics.bootstrap``);
* ``SR`` (only when ``mjj`` is given): the same for the events with ``sr[0] <= mjj <= sr[1]``;
* ``sculpting`` (only when ``mjj`` is given): JSD of the background m_jj distribution before / after a cut
  keeping the top 10% / 1% of background (``sculpting.sculpting_jsd``, all background events).

The result holds only str / bool / int / float / list values and is exactly what ``metrics.json``
contains.  1/eps_B and SIC are ``inf`` when no background passes a cut, and max SIC is ``nan`` when no cut
keeps ``min_bkg`` background events (as in ``metrics``); ``json`` writes these as ``Infinity`` / ``NaN``,
like the other results files.  Scores are sorted per class before any computation, so the result does not
depend on the order of the events (the bootstrap resamples positions).

Degenerate inputs raise ``ValueError``: non-finite scores, labels other than 0/1, arrays of different
lengths, and no signal or no background event - over the full range or, when ``mjj`` is given, inside
the SR (usually a sign of m_jj in TeV instead of GeV, or a wrong ``sr``).

``load_score_file`` reads the arrays from a ``.npz`` file or a ``.csv`` file with a header line; it backs
``gaehep eval-scores``.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gaehep.config import SR_MJJ, save_json
from gaehep.eval import plots
from gaehep.eval.metrics import DEFAULT_MIN_BKG, bootstrap, summarize
from gaehep.eval.sculpting import sculpting_jsd


def _region(sig: np.ndarray, bkg: np.ndarray, where: str, n_boot: int, seed: int, min_bkg: int) -> dict:
    if len(sig) == 0 or len(bkg) == 0:
        raise ValueError(
            f"{where}: {len(sig)} signal and {len(bkg)} background events; need at least one of each"
        )
    entry: dict[str, Any] = summarize(sig, bkg, min_bkg)
    if n_boot:
        entry["bootstrap"] = bootstrap(sig, bkg, n_boot=n_boot, seed=seed, min_bkg=min_bkg)
    return entry


def evaluate_score_arrays(
    score: np.ndarray,
    label: np.ndarray,
    mjj: np.ndarray | None = None,
    *,
    sr: tuple[float, float] = SR_MJJ,
    n_boot: int = 30,
    min_bkg: int = DEFAULT_MIN_BKG,
    higher_is_signal: bool = True,
    seed: int = 0,
    name: str = "scores",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Metrics of per-event ``score`` for ``label`` (1 = signal, 0 = background); see the module docstring.

    ``higher_is_signal=False`` evaluates ``-score``.  ``n_boot=0`` skips the bootstrap; ``seed`` seeds it.
    ``name`` goes into the result and the figure titles.  With ``out_dir``, also writes ``metrics.json``
    (the returned dict) and ``rejection.png``, ``sic.png``, ``score_hist.png`` there.
    """
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label)
    if score.ndim != 1 or label.shape != score.shape:
        raise ValueError(f"score and label must be 1-d of equal length, got {score.shape} and {label.shape}")
    if not np.all(np.isfinite(score)):
        raise ValueError("non-finite scores")
    if not np.all(np.isin(label, (0, 1))):
        raise ValueError("labels must be 0 (background) or 1 (signal)")
    is_sig = label.astype(bool)
    if not higher_is_signal:
        score = -score
    lo, hi = (float(v) for v in sr)
    res: dict[str, Any] = {
        "name": name,
        "higher_is_signal": bool(higher_is_signal),
        "n_boot": int(n_boot),
        "min_bkg": int(min_bkg),
    }
    sig, bkg = np.sort(score[is_sig]), np.sort(score[~is_sig])
    res["full"] = _region(sig, bkg, "full range", n_boot, seed, min_bkg)
    curves = {f"{name} (full)" if mjj is not None else name: (sig, bkg)}
    if mjj is not None:
        mjj = np.asarray(mjj, dtype=np.float64)
        if mjj.shape != score.shape:
            raise ValueError(f"mjj must have the shape of score {score.shape}, got {mjj.shape}")
        in_sr = (mjj >= lo) & (mjj <= hi)  # NaN m_jj -> outside
        s_sr, b_sr = np.sort(score[is_sig & in_sr]), np.sort(score[~is_sig & in_sr])
        res["sr"] = [lo, hi]
        res["SR"] = _region(s_sr, b_sr, f"SR m_jj in [{lo:g}, {hi:g}] GeV", n_boot, seed, min_bkg)
        res["sculpting"] = {"mjj": sculpting_jsd(score[~is_sig], mjj[~is_sig])}
        curves[f"{name} (SR)"] = (s_sr, b_sr)
    if out_dir is not None:
        out = Path(out_dir)
        save_json(res, out / "metrics.json")
        title = name if higher_is_signal else f"{name} (lower score = signal)"
        plots.plot_rejection(curves, out / "rejection.png", title=title)
        plots.plot_sic(curves, out / "sic.png", title=title, min_bkg=min_bkg)
        plots.plot_score_hist(
            {"background": bkg, "signal": sig},
            out / "score_hist.png",
            xlabel="score" if higher_is_signal else "-score",
            title=title,
        )
    return res


def load_score_file(
    path: str | Path, keys: Sequence[str], optional: Sequence[str] = ()
) -> dict[str, np.ndarray]:
    """Columns ``keys`` (required) and ``optional`` (skipped if absent) of a ``.npz`` or header-line ``.csv``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            cols = {k: np.asarray(f[k]) for k in f.files}
    elif suffix == ".csv":
        with open(path, newline="") as fh:
            header = [h.strip() for h in next(csv.reader(fh), [])]
        if not header:
            raise ValueError(f"{path}: empty file, expected a header line")
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64)
        if data.size == 0:
            raise ValueError(f"{path}: no data rows")
        if data.shape[1] != len(header):
            raise ValueError(f"{path}: {len(header)} header names but {data.shape[1]} columns")
        cols = {h: data[:, i] for i, h in enumerate(header)}
    else:
        raise ValueError(f"{path}: unsupported file type {suffix!r} (expected .npz or .csv)")
    missing = [k for k in keys if k not in cols]
    if missing:
        raise ValueError(f"{path}: no column(s) {missing}; available: {sorted(cols)}")
    return {k: cols[k] for k in (*keys, *optional) if k in cols}
