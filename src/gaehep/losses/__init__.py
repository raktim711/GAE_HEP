"""Permutation-invariant set losses."""

from gaehep.losses.chamfer import masked_chamfer
from gaehep.losses.sinkhorn import masked_sinkhorn

__all__ = ["masked_chamfer", "masked_sinkhorn"]
