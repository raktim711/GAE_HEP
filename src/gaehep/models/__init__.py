"""Model zoo.  All anomaly scores are permutation invariant in the jet constituents."""

from __future__ import annotations

from typing import Any

from gaehep.models.base import JetModel
from gaehep.models.classifier import EdgeConvClassifier
from gaehep.models.deep_svdd import DeepSVDD
from gaehep.models.ef_ae import EnergyFlowAE
from gaehep.models.node_gae import NodeGAE
from gaehep.models.rnd import RandomDistillation
from gaehep.models.set_ae import SetAE

REGISTRY: dict[str, type[JetModel]] = {
    "set_ae": SetAE,
    "ef_ae": EnergyFlowAE,
    "node_gae": NodeGAE,
    "deep_svdd": DeepSVDD,
    "rnd": RandomDistillation,
    "classifier": EdgeConvClassifier,
}


def build_model(name: str, **kwargs: Any) -> JetModel:
    if name == "legacy_gae":  # imported lazily: needs torch_geometric
        from gaehep.models.legacy_gae import LegacyGAE

        return LegacyGAE(**kwargs)
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {[*REGISTRY, 'legacy_gae']}")
    return REGISTRY[name](**kwargs)


__all__ = [
    "REGISTRY",
    "DeepSVDD",
    "EdgeConvClassifier",
    "EnergyFlowAE",
    "JetModel",
    "NodeGAE",
    "RandomDistillation",
    "SetAE",
    "build_model",
]
