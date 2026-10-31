"""Small synthetic LHCO black boxes: raw pandas-HDF5 event files (700 particles x (pT, eta, phi), no label
column) and master keys (one float label per line), with a physical marker of the label in every event so
that tests can check the labels are aligned with the right rows.

The marker is the number of constituents per jet: background 5, label 1 12, label 2 8.  Background jets have
pT 1300-2300 GeV (m_jj 2.6-4.6 TeV); label-1 events are back-to-back dijets at m_jj ~ ``mass`` and label-2
events are three-jet events with a low m_jj of the two leading jets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaehep.data.preprocess import MASTER_KEYS, N_PARTICLES, RAW_FILES

K_CONST = {0: 5, 1: 12, 2: 8}


def _jet(rng: np.random.Generator, pt: float, eta: float, phi: float, k: int) -> np.ndarray:
    frac = rng.dirichlet(np.full(k, 3.0))
    d = rng.normal(0.0, 0.04, (k, 2))
    return np.stack([pt * frac, eta + d[:, 0], (phi + d[:, 1] + np.pi) % (2 * np.pi) - np.pi], axis=1)


def _event(rng: np.random.Generator, kind: str, label: int, mass: float) -> np.ndarray:
    phi = rng.uniform(-np.pi, np.pi)
    k = K_CONST[label]
    jets = []
    if kind == "dijet":
        pt = rng.normal(mass / 2.0, 20.0) if label else rng.uniform(1300.0, 2300.0)
        jets = [(pt, 0.2, phi), (pt * rng.uniform(0.97, 1.0), -0.2, phi + np.pi)]
    elif kind == "trijet":
        jets = [(1250.0, 0.0, phi), (1150.0, 0.1, phi + 2 * np.pi / 3), (600.0, -0.1, phi + 4 * np.pi / 3)]
    elif kind == "one_jet":
        jets = [(1500.0, 0.0, phi)]
    elif kind != "empty":
        raise ValueError(kind)
    rows = np.concatenate([_jet(rng, pt, eta, ph, k) for pt, eta, ph in jets]) if jets else np.zeros((0, 3))
    ev = np.zeros((N_PARTICLES, 3))
    ev[: len(rows)] = rows
    return ev


def write_box(
    raw: Path,
    sample: str,
    n_bkg: int,
    signals: dict[int, int] | None = None,
    mass: float = 3800.0,
    n_one_jet: int = 0,
    n_empty: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Write ``raw/<events file of sample>`` (and the master key, for bb1 and bb3) with events in a random
    order.  ``signals``: {label: n_events}.  Returns the labels in file order (int8)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    kinds = [("dijet", 0)] * n_bkg + [("one_jet", 0)] * n_one_jet + [("empty", 0)] * n_empty
    for lab, n in (signals or {}).items():
        kinds += [("dijet" if lab == 1 else "trijet", lab)] * n
    order = rng.permutation(len(kinds))
    events = np.stack([_event(rng, *kinds[i], mass) for i in order])
    labels = np.array([kinds[i][1] for i in order], dtype=np.int8)
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events.reshape(len(events), -1)).to_hdf(
        raw / RAW_FILES[sample], key="df", mode="w", complevel=1, complib="blosc"
    )
    if sample in MASTER_KEYS:
        (raw / MASTER_KEYS[sample]).write_text("".join(f"{float(lab)}\n" for lab in labels))
    return labels
