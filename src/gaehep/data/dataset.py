"""Graph-ready jet batches built from the processed memory-mapped arrays.

A *jet graph* has one node per constituent (top-``n_const`` by pT, padded and masked).  Node features are
computed relative to the jet axis so that they are invariant to the jet direction:

``deta, dphi, log pT, log E, log(pT/pT_jet), log(E/E_jet), dR``

They are standardised with statistics from **background-training** jets only (see :func:`fit_scaler`).
No feature depends on the order in which constituents are stored, so every model built on top of these
batches can be permutation invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gaehep.config import load_json, processed_dir, save_json

NODE_FEATURES = ("deta", "dphi", "log_pt", "log_e", "log_ptrel", "log_erel", "dr")
JET_FEATURES = ("log_m", "log_pt", "log_n")
COORD_IDX = (0, 1)  # (deta, dphi) used for the first k-NN graph
RECON_IDX = (0, 1, 2)  # (deta, dphi, log pT) reconstructed by set decoders


def wrap_phi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def node_features(
    c: np.ndarray, jet_pt: np.ndarray, jet_eta: np.ndarray, jet_phi: np.ndarray, jet_e: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unstandardised) node features.

    c : (B, N, 3) constituents (pT, eta, phi) with pT = 0 for padding.  jet_* : (B,).
    Returns feats (B, N, 7) float32 (zero on padded nodes) and mask (B, N) bool.
    """
    pt = c[..., 0].astype(np.float64)
    eta = c[..., 1].astype(np.float64)
    phi = c[..., 2].astype(np.float64)
    mask = pt > 0
    deta = eta - jet_eta[:, None]
    dphi = wrap_phi(phi - jet_phi[:, None])
    e = pt * np.cosh(eta)
    log_pt = np.log(np.where(mask, pt, 1.0))
    log_e = np.log(np.where(mask, e, 1.0))
    log_ptrel = log_pt - np.log(jet_pt)[:, None]
    log_erel = log_e - np.log(jet_e)[:, None]
    dr = np.hypot(deta, dphi)
    feats = np.stack([deta, dphi, log_pt, log_e, log_ptrel, log_erel, dr], axis=-1).astype(np.float32)
    feats[~mask] = 0.0
    return feats, mask


def jet_level_features(m: np.ndarray, pt: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Raw jet-level features (log m, log pT, log n) used for conditioning and multiplicity heads."""
    return np.stack([np.log(np.maximum(m, 1.0)), np.log(pt), np.log(np.maximum(n, 1))], axis=-1).astype(
        np.float32
    )


# --------------------------------------------------------------------------------------------- storage
class JetStore:
    """Memory-mapped constituents plus the per-jet table of one processed sample (``rnd`` or ``qqq``)."""

    def __init__(self, sample: str, root: Path | None = None):
        d = (Path(root) if root else processed_dir()) / sample
        self.sample = sample
        self.const = np.load(d / "constituents.npy", mmap_mode="r")
        with np.load(d / "events.npz") as ev:
            self.table = {k: ev[k] for k in ev.files}
        self.n_events = len(self.const)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.table[key]

    def pool(
        self, events: np.ndarray, n_const: int, jets: tuple[int, ...] = (0, 1), with_labels: bool = True
    ) -> JetPool:
        """Load the constituents of ``events`` x ``jets`` into RAM (truncated to ``n_const`` nodes).  With
        ``with_labels=False`` the label column of the table is not read (the pool's labels are all 0)."""
        events = np.asarray(events, dtype=np.int64)
        ev = np.repeat(events, len(jets))
        jt = np.tile(np.asarray(jets, dtype=np.int64), len(events))
        order = np.argsort(ev, kind="stable")  # sequential memmap access
        const = np.empty((len(ev), n_const, 3), dtype=np.float32)
        step = 50_000
        for s in range(0, len(ev), step):
            sel = order[s : s + step]
            const[sel] = self.const[ev[sel], jt[sel], :n_const, :]
        t = self.table
        return JetPool(
            const=const,
            event=ev,
            jet=jt,
            pt=t["pt"][ev, jt].astype(np.float32),
            eta=t["eta"][ev, jt].astype(np.float32),
            phi=t["phi"][ev, jt].astype(np.float32),
            m=t["m"][ev, jt].astype(np.float32),
            e=t["e"][ev, jt].astype(np.float32),
            n_const=t["n_const"][ev, jt].astype(np.int32),
            label=t["label"][ev].astype(np.int8) if with_labels else np.zeros(len(ev), dtype=np.int8),
        )


@dataclass
class JetPool:
    """In-memory jets (one row per (event, jet) pair)."""

    const: np.ndarray  # (M, N, 3)
    event: np.ndarray  # (M,)
    jet: np.ndarray  # (M,) 0 = leading, 1 = sub-leading (pT order)
    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    m: np.ndarray
    e: np.ndarray
    n_const: np.ndarray  # untruncated multiplicity
    label: np.ndarray

    def __len__(self) -> int:
        return len(self.event)

    @property
    def n_nodes(self) -> int:
        return self.const.shape[1]

    def subset(self, idx: np.ndarray) -> JetPool:
        return JetPool(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})


# ---------------------------------------------------------------------------------------------- scaler
@dataclass
class Scaler:
    n_const: int
    node_mean: np.ndarray
    node_std: np.ndarray
    jet_mean: np.ndarray
    jet_std: np.ndarray

    def to_json(self, path: str | Path) -> None:
        save_json(
            {
                "n_const": self.n_const,
                "node_features": list(NODE_FEATURES),
                "node_mean": self.node_mean.tolist(),
                "node_std": self.node_std.tolist(),
                "jet_features": list(JET_FEATURES),
                "jet_mean": self.jet_mean.tolist(),
                "jet_std": self.jet_std.tolist(),
            },
            path,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Scaler:
        d = load_json(path)
        return cls(
            n_const=int(d["n_const"]),
            node_mean=np.asarray(d["node_mean"], dtype=np.float32),
            node_std=np.asarray(d["node_std"], dtype=np.float32),
            jet_mean=np.asarray(d["jet_mean"], dtype=np.float32),
            jet_std=np.asarray(d["jet_std"], dtype=np.float32),
        )


def fit_scaler(pool: JetPool, max_jets: int = 200_000, seed: int = 0) -> Scaler:
    """Mean/std of node features over real nodes of (a random subset of) background-training jets."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(max_jets, len(pool)), replace=False)
    sub = pool.subset(np.sort(idx))
    feats, mask = node_features(sub.const, sub.pt, sub.eta, sub.phi, sub.e)
    real = feats[mask].astype(np.float64)
    jf = jet_level_features(sub.m, sub.pt, sub.n_const).astype(np.float64)
    return Scaler(
        n_const=pool.n_nodes,
        node_mean=real.mean(0).astype(np.float32),
        node_std=np.maximum(real.std(0), 1e-6).astype(np.float32),
        jet_mean=jf.mean(0).astype(np.float32),
        jet_std=np.maximum(jf.std(0), 1e-6).astype(np.float32),
    )


# --------------------------------------------------------------------------------------------- batches
@dataclass
class JetBatch:
    x: torch.Tensor  # (B, N, F) standardised node features, zero on padding
    mask: torch.Tensor  # (B, N) bool
    coords: torch.Tensor  # (B, N, 2) raw (deta, dphi) for the first k-NN graph
    jet: torch.Tensor  # (B, 3) standardised (log m, log pT, log n)
    n: torch.Tensor  # (B,) number of real nodes (<= N)
    label: torch.Tensor  # (B,) truth label (never used by unsupervised losses)
    raw: torch.Tensor  # (B, N, 3) raw constituents (pT, eta, phi); used only by the legacy port

    def __len__(self) -> int:
        return self.x.shape[0]

    def to(self, device: torch.device | str) -> JetBatch:
        return JetBatch(**{k: getattr(self, k).to(device) for k in self.__dataclass_fields__})


def make_batch(pool: JetPool, idx: np.ndarray, scaler: Scaler) -> JetBatch:
    sub = pool.subset(idx)
    feats, mask = node_features(sub.const, sub.pt, sub.eta, sub.phi, sub.e)
    coords = feats[..., list(COORD_IDX)].copy()
    x = (feats - scaler.node_mean) / scaler.node_std
    x[~mask] = 0.0
    jf = (jet_level_features(sub.m, sub.pt, sub.n_const) - scaler.jet_mean) / scaler.jet_std
    return JetBatch(
        x=torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        mask=torch.from_numpy(mask),
        coords=torch.from_numpy(np.ascontiguousarray(coords, dtype=np.float32)),
        jet=torch.from_numpy(jf.astype(np.float32)),
        n=torch.from_numpy(mask.sum(1).astype(np.int64)),
        label=torch.from_numpy(sub.label.astype(np.int64)),
        raw=torch.from_numpy(np.ascontiguousarray(sub.const, dtype=np.float32)),
    )


def iterate_batches(
    pool: JetPool,
    scaler: Scaler,
    batch_size: int,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
    limit: int | None = None,
):
    """Yield :class:`JetBatch` objects over (a random subset of) the pool."""
    n = len(pool)
    if shuffle:
        rng = rng or np.random.default_rng()
        order = rng.permutation(n)
        if limit is not None:
            order = order[:limit]
    else:
        order = np.arange(n if limit is None else min(n, limit))
    for s in range(0, len(order), batch_size):
        yield make_batch(pool, order[s : s + batch_size], scaler)
