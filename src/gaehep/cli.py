"""Command-line entry point: ``gaehep <command> ...`` (see ``gaehep -h``).

``gaehep eval-scores FILE --out DIR`` evaluates any per-event scores, without the LHCO tables of a run:
FILE is a ``.npz`` or a ``.csv`` with a header line holding a score column (``--score-key``, default
``score``), a 0/1 label column (``--label-key``, 1 = signal, default ``label``) and optionally an m_jj
column in GeV (``--mjj-key``, default ``mjj``; if the file has no such column, or ``--mjj-key ''``, only
full-range metrics are computed).  With m_jj, metrics are also given inside the SR (``--sr``, default
3300,3700) together with the m_jj sculpting JSD.  Larger scores must be more signal-like.  DIR receives
``metrics.json`` (``gaehep.eval.score_file.evaluate_score_arrays``) and ``rejection.png``, ``sic.png``,
``score_hist.png``; ``--name`` (default: FILE's stem) labels the figures and ``metrics.json``, and
``--n-boot`` sets the bootstrap resamples (0 = none).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


class _CliLoader(yaml.SafeLoader):
    """PyYAML's safe YAML 1.1 typing plus YAML 1.2 floats, for values typed on the command line.

    YAML 1.1 floats need a '.' and a signed exponent, so ``1e-3``, ``1.0e3`` and ``-.5`` would load as
    strings.  The added resolver accepts a YAML 1.2 float containing a '.' or an exponent ('09' stays a
    string).  It runs after PyYAML's own resolvers, so everything they type keeps its type, and like
    them it only sees plain scalars: a quoted ``'1e-3'`` is still a string.
    """


_CliLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^(?=.*[.eE])[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?$"),
    list("-+.0123456789"),
)


def _yaml(text: str) -> Any:
    """A YAML value given on the command line (``--set`` values, ``--grid``)."""
    return yaml.load(text, Loader=_CliLoader)


def _parse_set(item: str) -> tuple[str, Any]:
    """One ``--set key=value`` -> (dotted key, YAML-typed value); ``key=`` gives None."""
    key, eq, raw = item.partition("=")
    if not eq or not key:
        raise SystemExit(f"--set expects key=value, got {item!r}")
    return key, _yaml(raw)


def _set(cfg: dict[str, Any], dotted: str) -> None:
    """Apply one ``--set a.b.c=value`` to a config, creating the nested dicts that are missing."""
    key, val = _parse_set(dotted)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = val


def _log(*args: object) -> None:
    print(*args, flush=True)


def _eval_scores(a: argparse.Namespace) -> int:
    from gaehep.config import SR_MJJ
    from gaehep.eval.score_file import evaluate_score_arrays, load_score_file

    try:
        lo, hi = SR_MJJ if a.sr is None else (float(x) for x in a.sr.split(","))
    except ValueError:
        raise SystemExit(f"--sr expects lo,hi in GeV, got {a.sr!r}") from None
    if not lo < hi:
        raise SystemExit(f"--sr expects lo < hi, got {a.sr!r}")
    name = a.name or a.file.stem
    try:
        cols = load_score_file(a.file, (a.score_key, a.label_key), optional=(a.mjj_key,) if a.mjj_key else ())
        if a.mjj_key and a.mjj_key not in cols:
            _log(f"[{name}] no {a.mjj_key!r} column in {a.file}: full-range metrics only")
        res = evaluate_score_arrays(
            cols[a.score_key],
            cols[a.label_key],
            cols.get(a.mjj_key),
            sr=(lo, hi),
            n_boot=a.n_boot,
            name=name,
            out_dir=a.out,
        )
    except (OSError, ValueError) as err:
        raise SystemExit(f"eval-scores: {err}") from None
    for region in ("full", "SR"):
        if region in res:
            r = res[region]
            _log(
                f"[{name}] {region}: {r['n_sig']} sig / {r['n_bkg']} bkg  AUC {r['auc']:.4f}  "
                f"max SIC {r['max_sic']:.2f} (eps_S {r['eff_s_at_max_sic']:.2f})  1/eps_B@0.3 {r['rej@0.3']:.3g}"
            )
    print(a.out / "metrics.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        prog="gaehep", description="Graph autoencoders for LHC Olympics anomaly detection"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("download", "preprocess", "tables", "splits"):
        sub.add_parser(name, add_help=False)
    t = sub.add_parser("train", help="train (+score +evaluate) a model from a YAML config")
    t.add_argument("config", type=Path)
    t.add_argument("--name", default=None)
    t.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=5")
    t.add_argument("--stages", default="train,score,evaluate")
    d = sub.add_parser("dev", help="fast development run: train + metrics on validation splits only")
    d.add_argument("config", type=Path)
    d.add_argument("--name", default=None)
    d.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=5")
    d.add_argument("--bkg-events", type=int, default=None, help="background-validation events to score")
    sub.add_parser("dev-report", help="table of all dev runs -> results/dev/summary.md")
    sub.add_parser("dev-rescore", help="recompute dev metrics from saved validation scores")
    de = sub.add_parser("dev-eval", help="dev metrics (validation splits) of a fully trained run")
    de.add_argument("run")
    sub.add_parser("dev-baselines", help="single-observable references under the dev protocol")
    pr = sub.add_parser("promote", help="score + evaluate a dev-protocol model on the test splits")
    pr.add_argument("dev_run")
    pr.add_argument("name")
    rc = sub.add_parser("recon", help="input vs reconstruction figure of a (dev) run directory")
    rc.add_argument("run_dir", type=Path)
    dk = sub.add_parser(
        "dev-kme", help="training-free kernel-embedding kNN / Mahalanobis scores (dev entries)"
    )
    dk.add_argument("name")
    dk.add_argument("--bank-jets", type=int, default=100_000)
    dk.add_argument("--n-pca", type=int, default=64)
    dk.add_argument("-k", type=int, default=10)
    dk.add_argument("--dim", type=int, default=128, help="random Fourier features per kernel scale")
    dk.add_argument("--bank-mjj", default=None, help="restrict the bank to an m_jj window, e.g. 3300,3700")
    dpf = sub.add_parser("dev-profile", help="GMM / Mahalanobis density of a dev model's error profile")
    dpf.add_argument("run")
    dpf.add_argument("--out", default=None)
    dpf.add_argument("--components", type=int, default=6)
    dn = sub.add_parser("dev-ensemble", help="rank ensemble of dev entries (seeds, or rec + KL scores)")
    dn.add_argument("runs", nargs="+")
    dn.add_argument("--out", required=True)
    df = sub.add_parser("dev-flow", help="latent-flow variants of a dev run")
    df.add_argument("run")
    df.add_argument("--kinds", default="flow,cflow,eflow", help="comma list of flow, cflow, eflow")
    for name in ("evaluate", "flow", "qn"):
        s = sub.add_parser(name)
        s.add_argument("run")
        if name == "flow":
            s.add_argument("--conditional", action="store_true", help="p(z | log m_j, log pT_j)")
    sw = sub.add_parser("sweep", help="dev-protocol grid sweep -> seed confirmation -> test evaluation")
    sw.add_argument("config", type=Path)
    sw.add_argument("--prefix", default="sw")
    sw.add_argument("--device", default="auto")
    sw.add_argument("--threads", type=int, default=1)
    sw.add_argument("--budget-hours", type=float, default=8.0)
    sw.add_argument("--grid", default=None, help='YAML dict, e.g. "{latent_dim: [4], beta: [0.001], ...}"')
    sw.add_argument("--top-k", type=int, default=3)
    sw.add_argument("--seeds", type=int, default=3)
    sw.add_argument("--final-seeds", type=int, default=5)
    sw.add_argument(
        "--score",
        default="rec",
        choices=["rec", "ens", "best"],
        help="rank by the reconstruction score, the rec+KL rank ensemble, or the better of the two",
    )
    sw.add_argument("--set", action="append", default=[], help="fixed override, e.g. data.n_const=100")
    co = sub.add_parser("collapse", help="node-embedding collapse of the legacy GAE on real jets")
    co.add_argument("--dev-runs", nargs="*", default=[])
    wk = sub.add_parser("weak", help="idealised weak supervision (IAD) on a run's jet latents vs HL4")
    wk.add_argument("run")
    wk.add_argument("--n-models", type=int, default=5)
    wk.add_argument("--n-inj", default="150,300,600,1200,3000", help="injected signal events in the SR")
    wk.add_argument("--reps", type=int, default=3)
    bh = sub.add_parser("bumphunt", help="sideband-fit bump hunt using a run's event scores")
    bh.add_argument("run")
    bh.add_argument("--agg", default="max", choices=["max", "min", "sum", "prod"])
    bh.add_argument("--mode", default="fixed", choices=["fixed", "qr"], help="fixed or m_jj-dependent cut")
    e = sub.add_parser("ensemble")
    e.add_argument("runs", nargs="+")
    e.add_argument("--out", required=True)
    sub.add_parser("baselines")
    cmp_ = sub.add_parser("compare", help="overlay ROC/SIC curves of several runs")
    cmp_.add_argument("runs", nargs="+")
    cmp_.add_argument("--out", type=Path, required=True)
    cmp_.add_argument("--agg", default="max")
    cmp_.add_argument("--region", default="SR")
    sub.add_parser("report")
    es = sub.add_parser("eval-scores", help="metrics + figures of per-event scores in a .npz / .csv file")
    es.add_argument("file", type=Path)
    es.add_argument("--out", type=Path, required=True)
    es.add_argument("--score-key", default="score")
    es.add_argument("--label-key", default="label", help="0 = background, 1 = signal")
    es.add_argument("--mjj-key", default="mjj", help="m_jj [GeV]; skipped if absent from the file")
    es.add_argument("--name", default=None, help="figure titles and metrics name (default: file stem)")
    es.add_argument("--sr", default=None, help="SR m_jj window lo,hi [GeV] (default: 3300,3700)")
    es.add_argument("--n-boot", type=int, default=30)

    if argv and argv[0] in ("download", "preprocess", "tables", "splits"):
        mod = {
            "download": "gaehep.data.download",
            "preprocess": "gaehep.data.preprocess",
            "tables": "gaehep.data.tables",
            "splits": "gaehep.data.splits",
        }[argv[0]]
        import importlib

        return importlib.import_module(mod).main(argv[1:])

    a = p.parse_args(argv)
    if a.cmd == "eval-scores":  # before importing the training code (torch): this needs only numpy
        return _eval_scores(a)
    from gaehep.train import run as R

    if a.cmd == "train":
        cfg = yaml.safe_load(a.config.read_text())
        for s in a.set:
            _set(cfg, s)
        name = a.name or a.config.stem
        summary = R.run(cfg, name, stages=tuple(a.stages.split(",")), log=_log)
        print(f"[{name}] status: {summary.get('status')}")
    elif a.cmd == "dev":
        from gaehep.train.dev import DEV_TRAIN, dev_run

        cfg = yaml.safe_load(a.config.read_text())
        cfg.setdefault("train", {}).update(DEV_TRAIN)
        for s in a.set:
            _set(cfg, s)
        res = dev_run(cfg, a.name or a.config.stem, bkg_events=a.bkg_events, log=_log)
        print(f"[dev:{a.name or a.config.stem}] status: {res.get('status', 'done')}")
    elif a.cmd == "dev-eval":
        from gaehep.train.dev import dev_eval_run

        dev_eval_run(a.run, log=_log)
    elif a.cmd == "dev-kme":
        from gaehep.train.dev import dev_kme

        win = tuple(float(x) for x in a.bank_mjj.split(",")) if a.bank_mjj else None
        dev_kme(
            a.name, bank_jets=a.bank_jets, n_pca=a.n_pca, k=a.k, dim_per_scale=a.dim, bank_mjj=win, log=_log
        )
    elif a.cmd == "dev-profile":
        from gaehep.train.dev import dev_profile

        dev_profile(a.run, a.out, n_components=a.components, log=_log)
    elif a.cmd == "dev-ensemble":
        from gaehep.train.dev import dev_ensemble

        dev_ensemble(a.runs, a.out, log=_log)
    elif a.cmd == "dev-flow":
        from gaehep.train.dev import dev_flow

        dev_flow(a.run, kinds=tuple(a.kinds.split(",")), log=_log)
    elif a.cmd == "recon":
        from gaehep.eval.recon import recon_figure

        print(recon_figure(a.run_dir))
    elif a.cmd == "promote":
        from gaehep.train.dev import promote

        promote(a.dev_run, a.name, log=_log)
    elif a.cmd == "dev-baselines":
        from gaehep.train.dev import dev_baselines

        dev_baselines(log=_log)
    elif a.cmd == "dev-rescore":
        from gaehep.train.dev import dev_rescore

        dev_rescore(log=_log)
    elif a.cmd == "dev-report":
        from gaehep.train.dev import dev_report

        print(dev_report())
    elif a.cmd == "evaluate":
        R.evaluate_run(a.run)
    elif a.cmd == "flow":
        R.latent_flow_variant(a.run, conditional=a.conditional)
    elif a.cmd == "qn":
        R.quantile_normalised_variant(a.run)
    elif a.cmd == "sweep":
        from gaehep.train.sweep import sweep

        fixed = {"train.device": a.device, "train.threads": a.threads}
        fixed.update(_parse_set(item) for item in a.set)
        sweep(
            str(a.config),
            prefix=a.prefix,
            grid=_yaml(a.grid) if a.grid else None,
            overrides=fixed,
            budget_hours=a.budget_hours,
            top_k=a.top_k,
            n_seeds=a.seeds,
            final_seeds=a.final_seeds,
            score=a.score,
            log=_log,
        )
    elif a.cmd == "collapse":
        from gaehep.eval.collapse import collapse_study

        collapse_study(dev_runs=tuple(a.dev_runs), log=_log)
    elif a.cmd == "weak":
        from gaehep.train.weak_run import weak_run

        weak_run(
            a.run,
            n_models=a.n_models,
            n_inj=tuple(int(x) for x in a.n_inj.split(",")),
            reps=tuple(range(a.reps)),
            log=_log,
        )
    elif a.cmd == "bumphunt":
        R.bumphunt_run(a.run, agg=a.agg, mode=a.mode)
    elif a.cmd == "ensemble":
        R.ensemble_variant(a.runs, a.out)
    elif a.cmd == "baselines":
        R.baselines()
    elif a.cmd == "compare":
        from gaehep.eval.report import compare

        for p_ in compare(a.runs, a.out, agg=a.agg, region=a.region):
            print(p_)
    elif a.cmd == "report":
        from gaehep.eval.report import main as report

        print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
