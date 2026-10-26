"""Energy-flow graph autoencoder ("M2e"): reconstruct the jet's pT-weighted energy flow.

Same permutation-invariant EdgeConv encoder and jet-level bottleneck as :class:`~gaehep.models.set_ae.SetAE`,
but the decoder emits ``n_out`` points in (deta, dphi) **with pT fractions** (softmax over the points),
and the loss / anomaly score is the energy-weighted multi-scale MMD between the input energy flow and its
reconstruction (:func:`gaehep.losses.mmd.energy_mmd`).  Soft constituents carry little weight, so the
score reflects the hard structure (prongs, their separation, i.e. mass and N-subjettiness-like
information) instead of soft multiplicity.  Order-free in both the input and the output sets.

``cond_dim > 0`` conditions the decoder on the jet's standardised (log m, log pT) (mass-decorrelated
variant): the latent then only needs to describe the energy flow *given* the jet mass.

Variants studied in RESULTS.md §11 (all off by default, which reproduces the earlier models exactly):

* ``loss`` = ``"mmd"`` (default), ``"nll"`` (energy-weighted cross-entropy under the decoder read as a Gaussian
  mixture with learned per-component widths; :func:`~gaehep.losses.mmd.energy_nll`) or ``"mmd+nll"``; the
  other term is still returned (``mmd`` / ``nll``) so both can be evaluated as scores in one pass.
* ``energy_power`` p: the energy flow is weighted by pT**p (normalised) instead of pT; p = 1 is IRC safe.
* ``align``: rotate/reflect (deta, dphi) into the jet's principal-axis frame (energy-weighted second moments;
  the signs fixed by the third moments), so that the latent does not spend a dimension on the orientation.
* ``encoder`` = ``"edgeconv"`` (default) or ``"ewmp"``: an infrared- and collinear-safe energy-weighted
  message-passing encoder on radius graphs (:mod:`gaehep.models.ewmp`; ``radii`` sets the balls).  ``ewmp_checkpoint`` recomputes the pairwise
  messages in the backward pass: ~30 % more time for a several-fold smaller training memory, needed at 100
  constituents per jet.

``variational=True`` turns the bottleneck into a Gaussian VAE latent (loss = MMD + beta * KL); the KL term
of a jet is a density-like anomaly score that does not reward "simple" jets the way reconstruction does.
``score`` selects the anomaly score: ``"recon"`` (MMD at z = mu), ``"kl"`` or ``"total"``.  All of them
are returned by :meth:`losses` (``rec``, ``kl``) so several scores can be evaluated in one pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gaehep.data.dataset import JetBatch
from gaehep.losses.mmd import DEFAULT_SIGMAS, energy_mmd, energy_nll
from gaehep.models.base import JetModel, mlp
from gaehep.models.edgeconv import EdgeConvEncoder
from gaehep.models.ewmp import EWMPEncoder


def principal_axis_frame(coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Express (deta, dphi) in the principal-axis frame of the energy flow (z = pT fractions, 0 on padding).

    A rotation by the angle of the major axis of the energy-weighted second-moment matrix, then a rotation by
    pi if the third moment along the major axis is negative and a reflection of the minor axis if its third
    moment is negative.  All steps use sums over the nodes, so the result is permutation- and padding-
    invariant; rotations and reflections preserve every pairwise distance, so the kernel losses are unchanged.
    """
    x, y = coords[..., 0], coords[..., 1]
    cxx, cyy, cxy = (z * x * x).sum(1), (z * y * y).sum(1), (z * x * y).sum(1)
    th = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    u, v = c * x + s * y, -s * x + c * y
    flip_u = torch.where((z * u.pow(3)).sum(1, keepdim=True) < 0, -1.0, 1.0)
    u, v = u * flip_u, v * flip_u
    flip_v = torch.where((z * v.pow(3)).sum(1, keepdim=True) < 0, -1.0, 1.0)
    return torch.stack([u, v * flip_v], dim=-1)


class EnergyFlowAE(JetModel):
    def __init__(
        self,
        in_dim: int = 7,
        hidden: Sequence[int] = (48, 48, 48),
        k: int = 12,
        latent_dim: int = 8,
        n_out: int = 32,
        dec_hidden: Sequence[int] = (256, 256),
        cond_dim: int = 0,
        aggr: str = "mean",
        dynamic: bool = True,
        sigmas: Sequence[float] = DEFAULT_SIGMAS,
        variational: bool = False,
        beta: float = 1e-3,
        score: str = "recon",
        loss: str = "mmd",
        energy_power: float = 1.0,
        align: bool = False,
        coord_scale: float = 0.234,
        encoder: str = "edgeconv",
        radii: Sequence[float] = (0.3,),
        min_sigma: float = 0.01,
        ewmp_checkpoint: bool = False,
    ):
        super().__init__()
        if loss not in ("mmd", "nll", "mmd+nll"):
            raise ValueError(f"loss={loss!r}: choose mmd, nll or mmd+nll")
        if encoder not in ("edgeconv", "ewmp"):
            raise ValueError(f"encoder={encoder!r}: choose edgeconv or ewmp")
        if score not in ("recon", "kl", "total") or (score != "recon" and not variational):
            raise ValueError(f"score={score!r} needs variational=True (or use 'recon')")
        self.n_out, self.cond_dim, self.sigmas = n_out, cond_dim, tuple(sigmas)
        self.variational, self.beta, self.score_kind = variational, beta, score
        self.loss_kind, self.energy_power, self.align = loss, float(energy_power), bool(align)
        self.coord_scale, self.min_sigma, self.irc = float(coord_scale), float(min_sigma), encoder == "ewmp"
        self.use_nll = "nll" in loss
        if self.irc:
            self.eval_batch_size = 256  # dense (B, N, N, C) pairwise messages: keep the scoring batches small
            self.encoder = EWMPEncoder(
                3, hidden, radii, checkpoint=ewmp_checkpoint
            )  # inputs: (deta, dphi, dR)
        else:
            self.encoder = EdgeConvEncoder(in_dim, hidden, k=k, dynamic=dynamic, aggr=aggr)
        self.to_latent = mlp(2 * self.encoder.out_dim, [128], latent_dim * (2 if variational else 1))
        self.n_chan = 4 if self.use_nll else 3  # (x, y, weight logit[, log sigma])
        self.decoder = mlp(latent_dim + cond_dim, dec_hidden, n_out * self.n_chan)

    def _frame(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Node features, (deta, dphi) and pT fractions in the frame used by the encoder and the loss."""
        pt = b.raw[..., 0] * b.mask
        z = pt / pt.sum(1, keepdim=True).clamp(min=1e-12)
        if not self.align:
            return b.x, b.coords, z
        coords = principal_axis_frame(b.coords, z) * b.mask[..., None]
        x = torch.cat([coords / self.coord_scale, b.x[..., 2:]], dim=-1)
        return x, coords, z

    def _encode(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, coords, z = self._frame(b)
        if self.irc:
            pooled = self.encoder.pool(self.encoder(x[..., [0, 1, 6]], b.mask, coords, z), z)
        else:
            pooled = self.encoder.pool(self.encoder(x, b.mask, coords), b.mask)
        h = self.to_latent(pooled)
        if not self.variational:
            return h, None
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar.clamp(-12.0, 8.0)

    def embed(self, b: JetBatch) -> torch.Tensor:
        return self._encode(b)[0]

    def forward(self, b: JetBatch) -> dict[str, torch.Tensor]:
        mu, logvar = self._encode(b)
        z = mu
        if self.variational and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        zc = torch.cat([z, b.jet[:, : self.cond_dim]], dim=-1) if self.cond_dim else z
        out = self.decoder(zc).view(len(b), self.n_out, self.n_chan)
        res = {"z": mu, "xy": out[..., :2], "w": torch.softmax(out[..., 2], dim=-1)}
        if self.use_nll:
            res["sigma"] = torch.exp(out[..., 3].clamp(-6.0, 1.0)).clamp(min=self.min_sigma)
        if logvar is not None:
            res["logvar"] = logvar
        return res

    def target_flow(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Input energy flow in the model's frame: (deta, dphi) and weights pT**p / sum pT**p over the
        (top-N) real constituents (p = ``energy_power`` = 1: the pT fractions)."""
        _, coords, z = self._frame(b)
        if self.energy_power == 1.0:
            return coords, z
        w = (b.raw[..., 0] * b.mask).clamp(min=0).pow(self.energy_power)
        return coords, w / w.sum(1, keepdim=True).clamp(min=1e-12)

    def score_and_embed(self, b: JetBatch) -> tuple[torch.Tensor, torch.Tensor]:
        out = self(b)
        return self.losses(b, out)["score"], out["z"]

    def losses(self, b: JetBatch, out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        out = self(b) if out is None else out
        xy, w = self.target_flow(b)
        mmd = energy_mmd(out["xy"], out["w"], xy, w, self.sigmas)
        res = {"mmd": mmd}
        if self.use_nll:
            res["nll"] = energy_nll(out["xy"], out["w"], out["sigma"], xy, w)
        rec = {"mmd": mmd, "nll": res.get("nll"), "mmd+nll": mmd + res.get("nll", mmd)}[self.loss_kind]
        res.update({"rec": rec, "total": rec})
        if self.variational:
            mu, logvar = out["z"], out["logvar"]
            res["kl"] = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1)
            res["total"] = rec + self.beta * res["kl"]
        res["score"] = {"recon": rec, "kl": res.get("kl"), "total": res["total"]}[self.score_kind]
        return res
