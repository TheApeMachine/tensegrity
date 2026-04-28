"""
Primary V2 core: the unified energy landscape.

The old Morton-coded V1 API lives under ``tensegrity.legacy.v1``. Compatibility
modules remain at ``tensegrity.core.agent``, ``tensegrity.core.morton``, and
``tensegrity.core.blanket`` for migration, but they are no longer part of the
primary export surface.
"""

from tensegrity.engine.unified_field import (
    UnifiedField,
    HopfieldMemoryBank,
    EnergyDecomposition,
)
from tensegrity.engine.ngc import PredictiveCodingCircuit, LayerState
from tensegrity.engine.fhrr import (
    FHRREncoder,
    FHRRCodebook,
    SemanticFHRRCodebook,
    bind,
    bundle,
    unbind,
    permute,
)
from tensegrity.engine.causal_energy import (
    EnergyCausalArena,
    CausalEnergyTerm,
    TopologyMapper,
    TopologyMapping,
    VirtualParent,
)
from tensegrity.engine.scoring import ScoringBridge, NGCLogitsProcessor

__all__ = (
    "UnifiedField",
    "HopfieldMemoryBank",
    "EnergyDecomposition",
    "PredictiveCodingCircuit",
    "LayerState",
    "FHRREncoder",
    "FHRRCodebook",
    "SemanticFHRRCodebook",
    "bind",
    "bundle",
    "unbind",
    "permute",
    "EnergyCausalArena",
    "CausalEnergyTerm",
    "TopologyMapper",
    "TopologyMapping",
    "VirtualParent",
    "ScoringBridge",
    "NGCLogitsProcessor",
)
