"""Paths, YAML configs and small run-time helpers shared by all modules.

Data live outside the repository (default ``/home/claude/data/lhco``); override with ``GAEHEP_DATA``.
Results (metrics JSON, figures, small model files) live in ``<repo>/results`` unless ``GAEHEP_RESULTS`` is set.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Signal region used throughout the LHC Olympics literature (TeV -> GeV).
SR_MJJ = (3300.0, 3700.0)

# Sample identifiers used in processed files and split tables.
SAMPLES = ("rnd", "qqq")  # rnd: QCD background + 2-prong W' signal; qqq: 3-prong W' signal only


def data_dir() -> Path:
    return Path(os.environ.get("GAEHEP_DATA", "/home/claude/data/lhco"))


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def results_dir() -> Path:
    return Path(os.environ.get("GAEHEP_RESULTS", REPO_ROOT / "results"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)
    tmp.replace(path)


def load_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (if importable)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def run_metadata() -> dict[str, Any]:
    """Provenance stamped into every results file."""
    meta: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git": git_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host": platform.node(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["torch_threads"] = torch.get_num_threads()
    except ImportError:  # pragma: no cover
        pass
    return meta


class Timer:
    """Context manager measuring wall time: ``with Timer() as t: ...; t.seconds``."""

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start
