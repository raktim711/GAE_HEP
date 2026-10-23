"""Command-line entry point: ``gaehep <command> ...`` (see ``gaehep -h``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def _set(cfg: dict[str, Any], dotted: str) -> None:
    key, _, raw = dotted.partition("=")
    if not _:
        raise SystemExit(f"--set expects key=value, got {dotted!r}")
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def _log(*args: object) -> None:
    print(*args, flush=True)


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
    for name in ("evaluate", "flow", "qn"):
        s = sub.add_parser(name)
        s.add_argument("run")
    bh = sub.add_parser("bumphunt", help="sideband-fit bump hunt using a run's event scores")
    bh.add_argument("run")
    bh.add_argument("--agg", default="max", choices=["max", "min", "sum"])
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
    from gaehep.train import run as R

    if a.cmd == "train":
        cfg = yaml.safe_load(a.config.read_text())
        for s in a.set:
            _set(cfg, s)
        name = a.name or a.config.stem
        summary = R.run(cfg, name, stages=tuple(a.stages.split(",")), log=_log)
        print(f"[{name}] status: {summary.get('status')}")
    elif a.cmd == "dev":
        from gaehep.train.dev import dev_run

        cfg = yaml.safe_load(a.config.read_text())
        for s in a.set:
            _set(cfg, s)
        res = dev_run(cfg, a.name or a.config.stem, bkg_events=a.bkg_events, log=_log)
        print(f"[dev:{a.name or a.config.stem}] status: {res.get('status', 'done')}")
    elif a.cmd == "dev-report":
        from gaehep.train.dev import dev_report

        print(dev_report())
    elif a.cmd == "evaluate":
        R.evaluate_run(a.run)
    elif a.cmd == "flow":
        R.latent_flow_variant(a.run)
    elif a.cmd == "qn":
        R.quantile_normalised_variant(a.run)
    elif a.cmd == "bumphunt":
        R.bumphunt_run(a.run, agg=a.agg)
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
