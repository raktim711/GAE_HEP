"""Energy-flow polynomials (EFPs) of jets, from the ``energyflow`` package.

Every EFP here uses the hadronic measure with beta = 1 and pT-fraction weights (``normed=True``): for a
multigraph G on N vertices,

    EFP_G = sum_{i_1 ... i_N} z_{i_1} ... z_{i_N} prod_{(k, l) in G} DeltaR_{i_k i_l},   z_i = pT_i / sum_j pT_j.

They are infrared- and collinear-safe, and exactly invariant under permutations of the constituents (the sum
is symmetric) and under zero-pT padding (z = 0 removes a constituent).  Constituents are the stored
``(pT, eta, phi)`` rows, treated as massless, so eta stands in for the rapidity.

* :func:`prime_efps` - the prime (connected) graphs with d <= 4 edges, ``EFPSet('d<=4', 'p==1')``, minus the
  empty graph, whose EFP is the constant 1 (20 columns).  The k-NN baseline (``gaehep efp-knn``) uses it.
* :func:`probe_efps` - the wedge, triangle and square that ``gaehep probes`` regresses on.
"""

from __future__ import annotations

from functools import cache

import numpy as np

PRIME_SET = ("d<=4", "p==1")
PROBE_GRAPHS: dict[str, list[tuple[int, int]]] = {
    "efp_wedge": [(0, 1), (1, 2)],
    "efp_triangle": [(0, 1), (1, 2), (2, 0)],
    "efp_square": [(0, 1), (1, 2), (2, 3), (3, 0)],
}
LOG_FLOOR = 1e-12  # EFPs are >= 0 and vanish for jets of one constituent; log(max(EFP, floor))


def _ef():
    try:
        import energyflow
    except ImportError as err:  # pragma: no cover - energyflow is a declared dependency
        raise ImportError(
            "the EFP baselines and probes need the 'energyflow' package (pip install energyflow)"
        ) from err
    return energyflow


@cache
def _prime_set():
    ef = _ef()
    efpset = ef.EFPSet(*PRIME_SET, measure="hadr", beta=1, normed=True)
    keep = [i for i, g in enumerate(efpset.graphs()) if len(g) > 0]
    return efpset, keep


def prime_graphs() -> list[list[tuple[int, int]]]:
    """Edge lists of the graphs behind the columns of :func:`prime_efps`."""
    efpset, keep = _prime_set()
    graphs = efpset.graphs()
    return [[tuple(e) for e in graphs[i]] for i in keep]


def _check(const: np.ndarray) -> np.ndarray:
    const = np.asarray(const)
    if const.ndim != 3 or const.shape[-1] != 3:
        raise ValueError(
            f"constituents must have shape (n_jets, n_nodes, 3) = (pT, eta, phi), got {const.shape}"
        )
    return const.astype(np.float64, copy=False)


def prime_efps(const: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """EFPs of the prime graphs with d <= 4: ``(n_jets, 20)`` float64 for constituents ``(n_jets, n_nodes, 3)``."""
    const = _check(const)
    efpset, keep = _prime_set()
    out = np.empty((len(const), len(keep)), dtype=np.float64)
    for s in range(0, len(const), chunk):
        out[s : s + chunk] = efpset.batch_compute(const[s : s + chunk], n_jobs=1)[:, keep]
    return out


@cache
def _probe_efp(name: str):
    return _ef().EFP(PROBE_GRAPHS[name], measure="hadr", beta=1, normed=True)


def probe_efps(const: np.ndarray, chunk: int = 4096) -> dict[str, np.ndarray]:
    """The three probe EFPs (wedge, triangle, square) of every jet: name -> ``(n_jets,)`` float64."""
    const = _check(const)
    return {
        name: np.concatenate(
            [
                _probe_efp(name).batch_compute(const[s : s + chunk], n_jobs=1)
                for s in range(0, len(const), chunk)
            ]
        )
        if len(const)
        else np.zeros(0)
        for name in PROBE_GRAPHS
    }


def log_efps(x: np.ndarray) -> np.ndarray:
    """log(EFP), floored at ``LOG_FLOOR`` so that jets of one constituent stay finite."""
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.maximum(np.nan_to_num(x, nan=0.0), LOG_FLOOR))
