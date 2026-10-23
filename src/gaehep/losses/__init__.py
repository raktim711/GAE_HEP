"""Permutation-invariant set losses."""

from gaehep.losses.chamfer import masked_chamfer
from gaehep.losses.mmd import energy_mmd
from gaehep.losses.sinkhorn import masked_sinkhorn

__all__ = ["energy_mmd", "masked_chamfer", "masked_sinkhorn"]
