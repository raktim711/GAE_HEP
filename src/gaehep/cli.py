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
    for name in ("evaluate", "flow", "qn"):
        s = sub.add_parser(name)
        s.add_argument("run")
    e = sub.add_parser("ensemble")
    e.add_argument("runs", nargs="+")
    e.add_argument("--out", required=True)
    sub.add_parser("baselines")
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
        summary = R.run(cfg, name, stages=tuple(a.stages.split(",")))
        print(f"[{name}] status: {summary.get('status')}")
    elif a.cmd == "evaluate":
        R.evaluate_run(a.run)
    elif a.cmd == "flow":
        R.latent_flow_variant(a.run)
    elif a.cmd == "qn":
        R.quantile_normalised_variant(a.run)
    elif a.cmd == "ensemble":
        R.ensemble_variant(a.runs, a.out)
    elif a.cmd == "baselines":
        R.baselines()
    elif a.cmd == "report":
        from gaehep.eval.report import main as report

        print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
