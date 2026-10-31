"""Study table and significance contrasts (docs/STUDY_PROTOCOL.md, sections 2 and 7).

``gaehep study-table <prefix>`` collects, for the runs ``<results>/runs/<name>`` whose names start with
``<prefix>``, the files ``metrics.json``, ``bb<i>_metrics.json``, ``probes.json``, ``irc_probe.json``,
``history.json`` and ``config.yaml`` (all optional), and writes to ``<results>/study/``

* ``<prefix>_table.md`` and ``<prefix>_table.csv``: one row per configuration, with the mean +- standard
  deviation over seeds and the value of the 5-seed rank ensemble;
* ``<prefix>_contrasts.json``: the protocol's significance rule for every ordered pair of configurations.

Configurations.  A run ``<config>_s<seed>`` is one seed of ``<config>``; ``<config>_ens`` is its rank
ensemble; any other name is a configuration of one unseeded run (``study_efpknn``, ``study_obs_*``).  The label
of a configuration is its name without ``<prefix>_``.  Names containing ``+`` (extra scores of a model) are
skipped.

Where the numbers come from (the protocol's table of fields; ``prod`` event scores):

=========  ================================================================  ==========
signal     file and field                                                    primary
=========  ================================================================  ==========
rd2        ``metrics.json``  ``metrics.event.prod["sig_test/<region>"]``     SR
rd3        ``metrics.json``  ``metrics.event.prod["sig3_test/<region>"]``    SR
bb1        ``bb1_metrics.json`` ``metrics.event.prod["bb1/<region>"]``       window
bb3        ``bb3_metrics.json`` ``metrics.event.prod["bb3/<region>"]``       window
=========  ================================================================  ==========

with metrics ``auc``, ``max_sic`` and ``rej@0.3`` (1/eps_B at eps_S = 0.3), the sculpting JSD
``metrics.sculpting.prod.mjj["jsd@0.01"]`` and the background validation loss ``history.json: best_val_loss``.
Values that are missing or not finite are left out of a mean.  A run is *flagged* when its ``best_val_loss``
exceeds 1.5 x the median of its configuration (when that median is positive); flagged runs stay in every mean.

Significance (A - B on one signal and one metric): ``|delta| > 2 SE`` with delta the difference of the seed
means and ``SE = sqrt(s_A^2 / n_A + s_B^2 / n_B)`` (sample standard deviations, ddof = 1); undetermined (null)
with fewer than two seeds in either configuration.  On BB1 and BB3 a second requirement is that the paired
event bootstrap of the difference of the two rank ensembles (``N_BOOT`` = 200 resamples of the box's events,
``default_rng(20201)``, the same events for both ensembles, 95 % percentile interval, events restricted to the
box's window after resampling) excludes zero.  The ensembles are rebuilt from the seed runs' ``bb<i>_scores.npz``
(keys ``jet_scores``, ``event_index``, ``mjj``, ``label``) and ``scores.npz`` (``bkg_val``) exactly as
``gaehep ensemble`` does it, then scored with the ``prod`` aggregation.  ``significant`` is true if both
requirements hold, false if one fails, and null if one is undetermined.  The bootstrap is skipped (with a
reason in ``meta.bootstrap_unavailable``) for configurations whose score files are missing or inconsistent and
for runs trained on a black box (``data.train_sample`` other than R&D), whose F_B is not ``scores.npz``'s.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gaehep.config import load_json, results_dir, save_json
from gaehep.eval.evaluate import aggregate
from gaehep.eval.metrics import DEFAULT_MIN_BKG, roc
from gaehep.eval.probes import TARGETS as PROBE_TARGETS

N_BOOT = 200
BOOT_SEED = 20201
FLAG_FACTOR = 1.5
RUN_FILES = ("metrics.json", "probes.json", "irc_probe.json", "history.json")


@dataclass(frozen=True)
class Signal:
    short: str
    label: str
    source: str | None  # None: metrics.json; else the black box, bb<i>_metrics.json
    key: str  # the signal in "<key>/<region>"
    primary: str  # SR or window
    sig_labels: tuple[int, ...] = ()  # BB label values that count as signal
    window: tuple[float, float] | None = None  # GeV, protocol section 5


SIGNALS = (
    Signal("rd2", "R&D 2-prong (tuned)", None, "sig_test", "SR"),
    Signal("rd3", "R&D 3-prong", None, "sig3_test", "SR"),
    Signal("bb1", "BB1", "bb1", "bb1", "window", (1,), (3600.0, 4000.0)),
    Signal("bb3", "BB3", "bb3", "bb3", "window", (1, 2), (4000.0, 4400.0)),
)
METRICS = (
    ("auc", "AUC", "{:.3f}"),
    ("max_sic", "max SIC", "{:.2f}"),
    ("rej@0.3", "1/ε_B at ε_S = 0.3", "{:.1f}"),
)
IRC_LABELS = (
    "split top-1",
    "split top-5",
    "split all",
    "10 soft (0.2% pT each)",
    "10 soft (1% pT each)",
    "random re-order",
)


# ------------------------------------------------------------------------------------------- loading
def _json(path: Path) -> Any:
    try:
        return load_json(path)
    except (OSError, ValueError):
        return None


@dataclass
class Run:
    name: str
    dir: Path
    metrics: dict | None = None
    bb: dict[str, dict] = field(default_factory=dict)
    probes: dict | None = None
    irc: dict | None = None
    history: dict | None = None
    config: dict = field(default_factory=dict)

    @property
    def best_val_loss(self) -> float | None:
        return _finite((self.history or {}).get("best_val_loss"))


def _finite(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_run(path: Path) -> Run | None:
    """The run's files (missing or unreadable ones are skipped); None if it has none of them."""
    run = Run(path.name, path)
    run.metrics, run.probes, run.irc, run.history = (_json(path / f) for f in RUN_FILES)
    run.bb = {p.name.split("_")[0]: j for p in sorted(path.glob("bb*_metrics.json")) if (j := _json(p))}
    try:
        run.config = yaml.safe_load((path / "config.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        run.config = {}
    if not (run.metrics or run.bb or run.probes or run.irc or run.history):
        return None
    return run


_SEED = re.compile(r"^(?P<cfg>.+)_s(?P<seed>\d+)$")


@dataclass
class Config:
    key: str
    label: str
    runs: list[Run] = field(default_factory=list)  # seeds in seed order, or the one unseeded run
    ens: Run | None = None
    flagged: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        for r in self.runs:
            cfg = r.config
            return str((cfg.get("model_args") or {}).get("encoder") or cfg.get("model") or "")
        return ""


def collect(prefix: str, root: Path | None = None) -> list[Config]:
    """The configurations of the runs under ``<root>/runs`` (default ``results_dir()``) named ``<prefix>*``."""
    base = (Path(root) if root else results_dir()) / "runs"
    groups: dict[str, Config] = {}
    seeds: dict[str, list[tuple[int, Run]]] = {}
    for d in sorted(base.glob(f"{prefix}*")):
        if not d.is_dir() or "+" in d.name or (run := load_run(d)) is None:
            continue
        if d.name.endswith("_ens"):
            key, kind = d.name[:-4], "ens"
        elif m := _SEED.match(d.name):
            key, kind = m["cfg"], "seed"
        else:
            key, kind = d.name, "single"
        cfg = groups.setdefault(key, Config(key, key[len(prefix) :].lstrip("_") or key))
        if kind == "ens":
            cfg.ens = run
        elif kind == "seed":
            seeds.setdefault(key, []).append((int(_SEED.match(d.name)["seed"]), run))
        else:
            cfg.runs.append(run)
    for key, lst in seeds.items():
        groups[key].runs = [r for _, r in sorted(lst, key=lambda t: t[0])] + groups[key].runs
    for cfg in groups.values():
        cfg.flagged = flagged_runs(cfg.runs)
    return sorted((c for c in groups.values() if c.runs or c.ens), key=lambda c: c.key)


def flagged_runs(runs: list[Run]) -> list[str]:
    """Names of the runs whose best validation loss exceeds 1.5 x the median of the configuration."""
    vals = [(r.name, r.best_val_loss) for r in runs if r.best_val_loss is not None]
    if len(vals) < 2:
        return []
    med = float(np.median([v for _, v in vals]))
    return [n for n, v in vals if med > 0 and v > FLAG_FACTOR * med]


# ------------------------------------------------------------------------------------------ columns
@dataclass(frozen=True)
class Col:
    key: str
    label: str
    fmt: str
    get: Callable[[Run], float | None]


def _event_getter(sig: Signal, region: str, metric: str) -> Callable[[Run], float | None]:
    def get(run: Run) -> float | None:
        src = run.metrics if sig.source is None else run.bb.get(sig.source)
        try:
            return _finite(src["metrics"]["event"]["prod"][f"{sig.key}/{region}"][metric])
        except (KeyError, TypeError):
            return None

    return get


def _jsd(run: Run) -> float | None:
    try:
        return _finite(run.metrics["metrics"]["sculpting"]["prod"]["mjj"]["jsd@0.01"])
    except (KeyError, TypeError):
        return None


def _probe_getter(target: str, model: str) -> Callable[[Run], float | None]:
    def get(run: Run) -> float | None:
        try:
            return _finite(run.probes["targets"][target][model]["r2_mean"])
        except (KeyError, TypeError):
            return None

    return get


def _irc_getter(label: str, field_: str) -> Callable[[Run], float | None]:
    def get(run: Run) -> float | None:
        try:
            if field_ == "d_auc":
                return _finite(run.irc[label]["auc"] - run.irc["auc"])
            return _finite(run.irc[label][field_])
        except (KeyError, TypeError):
            return None

    return get


def _tie_fraction(run: Run) -> float | None:
    try:
        return _finite(run.irc["knn_ties"]["bkg"]["jet_fraction"])
    except (KeyError, TypeError):
        return None


def _columns() -> dict[str, Col]:
    cols: dict[str, Col] = {}
    for sig in SIGNALS:
        for region in (sig.primary, "full"):
            for metric, mlabel, fmt in METRICS:
                key = f"{sig.short}_{region}_{metric}"
                cols[key] = Col(
                    key, f"{mlabel}, {sig.label} ({region})", fmt, _event_getter(sig, region, metric)
                )
    cols["jsd"] = Col("jsd", "JSD of the R&D background m_jj at ε_B = 1 %", "{:.4f}", _jsd)
    cols["val_loss"] = Col("val_loss", "background validation loss", "{:.4g}", lambda r: r.best_val_loss)
    for model in ("ridge", "mlp"):
        for t in PROBE_TARGETS:
            key = f"probe_{model}_{t}"
            cols[key] = Col(key, f"R² {model}: {t}", "{:.3f}", _probe_getter(t, model))
    for lab in IRC_LABELS:
        cols[f"irc_rho_{lab}"] = Col(
            f"irc_rho_{lab}", f"Spearman ρ, {lab}", "{:.4f}", _irc_getter(lab, "rho")
        )
        cols[f"irc_dauc_{lab}"] = Col(
            f"irc_dauc_{lab}", f"Δ jet AUC, {lab}", "{:+.4f}", _irc_getter(lab, "d_auc")
        )
    cols["irc_ties"] = Col("irc_ties", "fraction of jets with a k-NN tie", "{:.4f}", _tie_fraction)
    return cols


COLUMNS = _columns()


def values(cfg: Config, col: Col) -> list[float]:
    return [v for r in cfg.runs if (v := col.get(r)) is not None]


def ens_value(cfg: Config, col: Col) -> float | None:
    return None if cfg.ens is None else col.get(cfg.ens)


def mean_std(v: list[float]) -> tuple[float | None, float | None]:
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None


# --------------------------------------------------------------------------------------------- tables
def _cell(cfg: Config, col: Col) -> str:
    m, s = mean_std(values(cfg, col))
    e = ens_value(cfg, col)
    if m is None and e is None:
        return "–"
    txt = (
        "–" if m is None else col.fmt.format(m) + ("" if s is None else " ± " + col.fmt.format(s).lstrip("+"))
    )
    return txt if e is None else f"{txt} [{col.fmt.format(e)}]"


def _md_table(configs: list[Config], keys: list[str], heads: list[str]) -> list[str]:
    lines = [
        "| configuration | runs | flagged | " + " | ".join(heads) + " |",
        "|---|--:|--:|" + "|".join(["---"] * len(heads)) + "|",
    ]
    for c in configs:
        cells = " | ".join(_cell(c, COLUMNS[k]) for k in keys)
        lines.append(f"| {c.label} | {len(c.runs)} | {len(c.flagged)} | {cells} |")
    return lines


def render_markdown(prefix: str, configs: list[Config]) -> str:
    out = [
        f"# Study table: `{prefix}`",
        "",
        "Each cell is the mean ± the sample standard deviation over the seeds, with the value of the rank ensemble "
        "in brackets (`prod` event score). *runs* counts the seed runs found; *flagged* counts runs whose best "
        "validation loss is above 1.5 × the median of their configuration (they stay in every mean). "
        "A dash means that no file with that number was found. The R&D 2-prong signal is tuned and never evidence.",
        "",
    ]
    for region_kind, tag in (("primary", "in the signal window"), ("full", "over the full m_jj range")):
        for metric, mlabel, _ in METRICS:
            keys = [
                f"{s.short}_{s.primary if region_kind == 'primary' else 'full'}_{metric}" for s in SIGNALS
            ]
            heads = [f"{s.label}, {s.primary if region_kind == 'primary' else 'full'}" for s in SIGNALS]
            out += [f"## {mlabel} {tag}", "", *_md_table(configs, keys, heads), ""]
    out += [
        "## Sculpting and training loss",
        "",
        *_md_table(
            configs, ["jsd", "val_loss"], ["JSD at ε_B = 1 % (R&D bkg_test)", "background validation loss"]
        ),
        "",
    ]
    for model in ("mlp", "ridge"):
        keys = [f"probe_{model}_{t}" for t in PROBE_TARGETS]
        out += [
            f"## Latent probes ({model}, R², 5-fold)",
            "",
            *_md_table(configs, keys, list(PROBE_TARGETS)),
            "",
        ]
    keys = [f"irc_{f}_{lab}" for lab in IRC_LABELS for f in ("rho", "dauc")] + ["irc_ties"]
    heads = [f"{'ρ' if f == 'rho' else 'Δ AUC'}: {lab}" for lab in IRC_LABELS for f in ("rho", "dauc")]
    out += ["## IRC probe", "", *_md_table(configs, keys, [*heads, "k-NN tie fraction"]), ""]
    flagged = [n for c in configs for n in c.flagged]
    out += ["## Flagged runs", "", ", ".join(f"`{n}`" for n in flagged) if flagged else "None.", ""]
    return "\n".join(out)


def csv_rows(configs: list[Config]) -> tuple[list[str], list[list[Any]]]:
    header = ["configuration", "label", "kind", "n_runs", "n_flagged", "has_ensemble"]
    for k in COLUMNS:
        header += [f"{k}_mean", f"{k}_std", f"{k}_ens"]
    rows = []
    for c in configs:
        row: list[Any] = [c.key, c.label, c.kind, len(c.runs), len(c.flagged), int(c.ens is not None)]
        for col in COLUMNS.values():
            m, s = mean_std(values(c, col))
            row += [m, s, ens_value(c, col)]
        rows.append(["" if v is None else v for v in row])
    return header, rows


# ----------------------------------------------------------------------------------------- contrasts
def seed_contrast(a: list[float], b: list[float]) -> dict[str, Any]:
    """The seed part of the protocol's rule for A - B."""
    out: dict[str, Any] = {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "delta": float(np.mean(a) - np.mean(b)),
        "se": None,
        "significant_seed": None,
    }
    if len(a) > 1 and len(b) > 1:
        out["se"] = float(math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
        out["significant_seed"] = bool(abs(out["delta"]) > 2 * out["se"])
    return out


def combine(seed_sig: bool | None, boot_excludes_zero: bool | None) -> bool | None:
    """Both requirements must hold: false if one fails, null if one is undetermined."""
    if seed_sig is False or boot_excludes_zero is False:
        return False
    if seed_sig is True and boot_excludes_zero is True:
        return True
    return None


def _ecdf(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:  # as in train.run._ecdf
    return np.searchsorted(ref_sorted, x, side="right") / max(len(ref_sorted), 1)


def ensemble_event_scores(runs: list[Run], box: str) -> dict[str, np.ndarray] | str:
    """``prod`` event scores of the rank ensemble of ``runs`` on a black box, from the runs' ``<box>_scores.npz``
    and ``scores.npz``; or the reason why they cannot be built."""
    acc_jet = acc_val = ref_event = None
    for r in runs:
        bb_path, val_path = r.dir / f"{box}_scores.npz", r.dir / "scores.npz"
        try:
            with np.load(bb_path) as f:
                bb = {k: f[k] for k in ("jet_scores", "event_index", "mjj", "label")}
            with np.load(val_path) as f:
                val = f["bkg_val"]
        except (OSError, KeyError, ValueError) as err:
            return f"{r.name}: cannot read {bb_path.name} / {val_path.name} ({type(err).__name__})"
        if ref_event is None:
            ref_event = bb
        elif not all(np.array_equal(bb[k], ref_event[k]) for k in ("event_index", "label", "mjj")):
            return f"{r.name}: {bb_path.name} lists different events than {runs[0].name}"
        ref = np.sort(val.ravel())
        pj, pv = _ecdf(ref, bb["jet_scores"]), _ecdf(ref, val)
        acc_jet = pj if acc_jet is None else acc_jet + pj
        acc_val = pv if acc_val is None else acc_val + pv
    if ref_event is None:
        return "no runs"
    ens_jet = (acc_jet / len(runs)).astype(np.float32)
    ens_val = (acc_val / len(runs)).astype(np.float32)
    ref = np.sort(ens_val.astype(np.float64).ravel())
    return {
        "score": aggregate(ens_jet, "prod", ref),
        "label": ref_event["label"],
        "mjj": ref_event["mjj"],
    }


def roc_metrics(
    sig: np.ndarray, bkg: np.ndarray, min_bkg: int = DEFAULT_MIN_BKG
) -> tuple[float, float, float]:
    """(AUC, max SIC, 1/eps_B at eps_S = 0.3) from one ROC curve (as ``metrics.summarize``)."""
    if len(sig) == 0 or len(bkg) == 0:
        return math.nan, math.nan, math.nan
    fpr, tpr, _ = roc(sig, bkg)
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    ok = fpr * len(bkg) >= min_bkg
    sic = float(np.max(tpr[ok] / np.sqrt(fpr[ok]))) if ok.any() else math.nan
    i = min(int(np.searchsorted(tpr, 0.3, side="left")), len(tpr) - 1)
    return auc, sic, (math.inf if fpr[i] == 0 else float(1.0 / fpr[i]))


def paired_bootstrap(
    boxes: dict[str, dict[str, np.ndarray]], sig: Signal, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per configuration: the (AUC, max SIC, 1/eps_B@0.3) of its ensemble on the box's window events, observed
    (3,) and on each of ``n_boot`` resamples of the box's events (n_boot, 3); every configuration sees the
    same resamples.  ``boxes`` maps a label to the output of :func:`ensemble_event_scores` (same events)."""
    first = next(iter(boxes.values()))
    label, mjj = first["label"], first["mjj"]
    lo, hi = sig.window
    in_win = (mjj >= lo) & (mjj <= hi)
    is_sig, is_bkg = np.isin(label, sig.sig_labels), label == 0
    n = len(label)

    def metrics(sel_s: np.ndarray, sel_b: np.ndarray) -> dict[str, tuple[float, float, float]]:
        return {k: roc_metrics(v["score"][sel_s], v["score"][sel_b]) for k, v in boxes.items()}

    w = np.flatnonzero(in_win)
    obs = metrics(w[is_sig[w]], w[is_bkg[w]])
    boot = {k: np.full((n_boot, 3), np.nan) for k in boxes}
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        idx = idx[in_win[idx]]
        for k, m in metrics(idx[is_sig[idx]], idx[is_bkg[idx]]).items():
            boot[k][b] = m
    return {k: np.asarray(v) for k, v in obs.items()}, boot


def _bootstrap_inputs(configs: list[Config]) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, str]]]:
    """box -> config label -> ensemble event scores; and the reasons for those left out."""
    ok: dict[str, dict[str, dict]] = {}
    why: dict[str, dict[str, str]] = {}
    for sig in (s for s in SIGNALS if s.source):
        for c in configs:
            cfg0 = c.runs[0].config if c.runs else {}
            if (cfg0.get("data") or {}).get("train_sample", "rnd") not in (None, "rnd"):
                why.setdefault(sig.short, {})[c.label] = (
                    "trained on a black box: F_B is not scores.npz's bkg_val"
                )
                continue
            if not c.runs:
                continue
            res = ensemble_event_scores(c.runs, sig.source)
            if isinstance(res, str):
                why.setdefault(sig.short, {})[c.label] = res
            else:
                ok.setdefault(sig.short, {})[c.label] = res
        # the configurations must list the same events; keep the largest group of identical ones
        group = ok.get(sig.short, {})
        if len(group) > 1:
            ref = next(iter(group.values()))
            for lab in [
                k
                for k, v in group.items()
                if not (np.array_equal(v["label"], ref["label"]) and np.array_equal(v["mjj"], ref["mjj"]))
            ]:
                why.setdefault(sig.short, {})[lab] = "lists different events than the other configurations"
                del group[lab]
    return ok, why


def contrasts(prefix: str, configs: list[Config], n_boot: int = N_BOOT, log=print) -> dict[str, Any]:
    """Every ordered pair (A, B) of configurations x signal x metric, as ``contrasts[A - B][signal][metric]``."""
    boxes, why = _bootstrap_inputs(configs)
    boots: dict[str, tuple[dict, dict]] = {}
    for sig in (s for s in SIGNALS if s.source):
        group = boxes.get(sig.short, {})
        if len(group) > 1:
            log(f"[study-table] bootstrap on {sig.short}: {len(group)} configurations, {n_boot} resamples")
            boots[sig.short] = paired_bootstrap(group, sig, n_boot)
    metric_idx = {m: i for i, (m, _, _) in enumerate(METRICS)}
    out: dict[str, Any] = {}
    for a in configs:
        for b in configs:
            if a is b:
                continue
            per: dict[str, Any] = {}
            for sig in SIGNALS:
                for metric, _, _ in METRICS:
                    col = COLUMNS[f"{sig.short}_{sig.primary}_{metric}"]
                    va, vb = values(a, col), values(b, col)
                    if not va or not vb:
                        continue
                    e = seed_contrast(va, vb)
                    excl = None
                    if sig.source:
                        e["bootstrap"] = None
                        if (
                            sig.short in boots
                            and a.label in boots[sig.short][0]
                            and b.label in boots[sig.short][0]
                        ):
                            obs, bt = boots[sig.short]
                            i = metric_idx[metric]
                            with np.errstate(invalid="ignore"):  # inf - inf: no background passes in either
                                d = bt[a.label][:, i] - bt[b.label][:, i]
                            d = d[np.isfinite(d)]
                            if len(d) >= n_boot // 2:
                                lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
                                excl = bool(lo > 0 or hi < 0)
                                e["bootstrap"] = {
                                    "delta_ensemble": float(obs[a.label][i] - obs[b.label][i]),
                                    "lo": lo,
                                    "hi": hi,
                                    "excludes_zero": excl,
                                    "n_boot": n_boot,
                                    "n_finite": len(d),
                                }
                    e["significant"] = (
                        combine(e["significant_seed"], excl) if sig.source else e["significant_seed"]
                    )
                    per.setdefault(f"{sig.key}/{sig.primary}", {})[metric] = e
            va, vb = values(a, COLUMNS["jsd"]), values(b, COLUMNS["jsd"])
            if va and vb:
                e = seed_contrast(va, vb)
                e["significant"] = e["significant_seed"]
                per["bkg_test/sculpting"] = {"jsd@0.01": e}
            if per:
                out[f"{a.label} - {b.label}"] = per
    return {
        "meta": {
            "prefix": prefix,
            "rule": "A - B is significant if |delta| > 2 SE, SE = sqrt(s_A^2/n_A + s_B^2/n_B) (ddof = 1); on BB1/BB3 "
            "also the paired event bootstrap of the ensemble difference must exclude zero",
            "bootstrap": {
                "n_boot": n_boot,
                "seed": BOOT_SEED,
                "interval": [2.5, 97.5],
                "units": "events of the box",
            },
            "windows_gev": {s.short: list(s.window) for s in SIGNALS if s.window},
            "metrics": [m for m, _, _ in METRICS] + ["jsd@0.01"],
            "configurations": {
                c.label: {"key": c.key, "n_runs": len(c.runs), "flagged": c.flagged} for c in configs
            },
            "bootstrap_unavailable": why,
        },
        "contrasts": out,
    }


# ------------------------------------------------------------------------------------------- driver
def study_table(
    prefix: str, root: Path | None = None, out_dir: Path | None = None, n_boot: int = N_BOOT, log=print
) -> dict[str, Path]:
    """Write ``<prefix>_table.md`` / ``.csv`` / ``_contrasts.json`` (default ``<results>/study``)."""
    configs = collect(prefix, root)
    if not configs:
        raise FileNotFoundError(
            f"no runs named {prefix}* with result files under {(root or results_dir()) / 'runs'}"
        )
    out_dir = Path(out_dir) if out_dir else results_dir() / "study"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{prefix}_{k}" for k in ("table.md", "table.csv", "contrasts.json")}
    paths["table.md"].write_text(render_markdown(prefix, configs))
    header, rows = csv_rows(configs)
    with open(paths["table.csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    save_json(contrasts(prefix, configs, n_boot, log=log), paths["contrasts.json"])
    log(f"[study-table] {len(configs)} configurations -> {out_dir}")
    return paths
