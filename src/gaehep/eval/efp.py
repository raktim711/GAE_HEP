"""Energy-flow polynomials (EFPs) of jets, in numpy.

Every EFP here uses the hadronic measure with beta = 1 and pT-fraction weights: for a multigraph G on N
vertices,

    EFP_G = sum_{i_1 ... i_N} z_{i_1} ... z_{i_N} prod_{(k, l) in G} DeltaR_{i_k i_l},   z_i = pT_i / sum_j pT_j,

with DeltaR_ij = hypot(eta_i - eta_j, wrap(phi_i - phi_j)).  They are infrared- and collinear-safe, and
exactly invariant under permutations of the constituents (the sum is symmetric) and under zero-pT padding
(z = 0 removes a constituent).  Constituents are the stored ``(pT, eta, phi)`` rows, treated as massless, so eta
stands in for the rapidity.

* :func:`prime_efps` - the 20 prime (connected) graphs with 1 <= d <= 4 edges, i.e. the graphs of
  ``energyflow.EFPSet('d<=4', 'p==1')`` without the empty graph (whose EFP is the constant 1).  The k-NN
  baseline (``gaehep efp-knn``) uses them.
* :func:`probe_efps` - the wedge, triangle and square that ``gaehep probes`` regresses on.

This is a numpy implementation rather than a call into the ``energyflow`` package: ``energyflow`` needs
``wasserstein``, whose releases either require numpy < 2 or are built against numpy 1.x, so it does not import
with the pinned numpy 2.x on Python 3.11 and 3.12.  The graph list below is energyflow 1.4.0's, and
``tests/test_baselines.py`` checks both the list and the values against the package whenever it is importable.
"""

from __future__ import annotations

import string

import numpy as np

Graph = list[tuple[int, int]]

PRIME_SET = ("d<=4", "p==1")  # the energyflow EFPSet selection that PRIME_GRAPHS reproduces

# energyflow.EFPSet('d<=4', 'p==1').graphs() (energyflow 1.4.0) without the empty graph; vertices are 0, 1, ...
PRIME_GRAPHS: list[Graph] = [
    [(0, 1)],
    [(0, 1), (0, 1)],
    [(0, 1), (0, 1), (0, 1)],
    [(0, 1), (0, 1), (0, 1), (0, 1)],
    [(0, 1), (0, 2)],
    [(0, 1), (0, 2), (0, 2)],
    [(0, 1), (0, 2), (0, 2), (0, 2)],
    [(0, 1), (0, 1), (0, 2), (0, 2)],
    [(0, 1), (0, 2), (1, 2)],
    [(0, 1), (0, 2), (0, 2), (1, 2)],
    [(0, 1), (0, 2), (0, 3)],
    [(0, 1), (0, 2), (0, 2), (0, 3)],
    [(0, 1), (0, 2), (1, 3)],
    [(0, 1), (0, 2), (0, 2), (1, 3)],
    [(0, 1), (0, 1), (0, 2), (1, 3)],
    [(0, 1), (0, 2), (1, 2), (0, 3)],
    [(0, 1), (0, 2), (1, 3), (2, 3)],
    [(0, 1), (0, 2), (0, 3), (0, 4)],
    [(0, 1), (0, 2), (0, 3), (1, 4)],
    [(0, 1), (0, 2), (1, 3), (2, 4)],
]
PROBE_GRAPHS: dict[str, Graph] = {
    "efp_wedge": [(0, 1), (1, 2)],
    "efp_triangle": [(0, 1), (1, 2), (2, 0)],
    "efp_square": [(0, 1), (1, 2), (2, 3), (3, 0)],
}
LOG_FLOOR = 1e-12  # EFPs are >= 0 and vanish for jets of one constituent; log(max(EFP, floor))
CHUNK = 2048  # jets per einsum call (a chunk of Delta R matrices is chunk * N^2 doubles)


def prime_graphs() -> list[Graph]:
    """Edge lists of the graphs behind the columns of :func:`prime_efps`."""
    return [list(g) for g in PRIME_GRAPHS]


def _check(const: np.ndarray) -> np.ndarray:
    const = np.asarray(const)
    if const.ndim != 3 or const.shape[-1] != 3:
        raise ValueError(
            f"constituents must have shape (n_jets, n_nodes, 3) = (pT, eta, phi), got {const.shape}"
        )
    return const.astype(np.float64, copy=False)


def _weights_and_distances(const: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pT fractions (B, N) and Delta R matrices (B, N, N) of constituents (B, N, 3)."""
    pt = const[..., 0]
    total = pt.sum(1, keepdims=True)
    z = np.divide(pt, total, out=np.zeros_like(pt), where=total > 0)
    deta = const[:, :, None, 1] - const[:, None, :, 1]
    dphi = np.abs(const[:, :, None, 2] - const[:, None, :, 2])
    dphi = np.pi - np.abs(np.pi - np.mod(dphi, 2 * np.pi))  # wrap into [0, pi]
    return z, np.hypot(deta, dphi)


def _contract(graph: Graph, z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """EFP of one graph for a batch: sum over all vertex assignments, one einsum."""
    n_vert = 1 + max(max(e) for e in graph)
    letters = string.ascii_lowercase[:n_vert]
    spec = ",".join(["B" + c for c in letters] + ["B" + letters[a] + letters[b] for a, b in graph])
    return np.einsum(f"{spec}->B", *([z] * n_vert), *([theta] * len(graph)), optimize="greedy")


def _efps(const: np.ndarray, graphs: list[Graph], chunk: int) -> np.ndarray:
    const = _check(const)
    out = np.empty((len(const), len(graphs)), dtype=np.float64)
    for s in range(0, len(const), chunk):
        z, theta = _weights_and_distances(const[s : s + chunk])
        for j, g in enumerate(graphs):
            out[s : s + chunk, j] = _contract(g, z, theta)
    return out


def prime_efps(const: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """EFPs of the prime graphs with d <= 4: ``(n_jets, 20)`` float64 for constituents ``(n_jets, n_nodes, 3)``."""
    return _efps(const, PRIME_GRAPHS, chunk)


def probe_efps(const: np.ndarray, chunk: int = CHUNK) -> dict[str, np.ndarray]:
    """The three probe EFPs (wedge, triangle, square) of every jet: name -> ``(n_jets,)`` float64."""
    x = _efps(const, list(PROBE_GRAPHS.values()), chunk)
    return {name: x[:, i] for i, name in enumerate(PROBE_GRAPHS)}


def log_efps(x: np.ndarray) -> np.ndarray:
    """log(EFP), floored at ``LOG_FLOOR`` so that jets of one constituent stay finite."""
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.maximum(np.nan_to_num(x, nan=0.0), LOG_FLOOR))
