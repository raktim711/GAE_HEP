"""Fake processed data + an empty results directory, exported through the environment for one test module."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from gaehep.data.fake import make_fake


@contextlib.contextmanager
def fake_env(root: Path, n_bkg: int = 600, n_sig: int = 150, n_max: int = 50):
    """Write the synthetic LHCO-like dataset under ``root/data`` and point GAEHEP_DATA / GAEHEP_RESULTS at it
    (and at ``root/results``) until the context exits."""
    make_fake(root / "data", n_bkg, n_sig, n_max)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GAEHEP_DATA", str(root / "data"))
        mp.setenv("GAEHEP_RESULTS", str(root / "results"))
        yield root
