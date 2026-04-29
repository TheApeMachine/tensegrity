"""
Unified cognitive engine: SBERT-native predictive coding, Hopfield memory,
FHRR compositional encoding, and energy-based causal competition.
"""

from tensegrity.engine.unified_field import UnifiedField, HopfieldMemoryBank, EnergyDecomposition
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
from tensegrity.engine.agent import CognitiveAgent, DEFAULT_MEDIATED_SCM_NAME

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
    "CognitiveAgent",
    "DEFAULT_MEDIATED_SCM_NAME",
)
