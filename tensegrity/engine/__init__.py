"""
Unified cognitive engine: compositional encoding, predictive coding, unified field,
semantic scoring, and optional energy-based causal competition.
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
from tensegrity.engine.causal_energy import EnergyCausalArena, CausalEnergyTerm
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
    "ScoringBridge",
    "NGCLogitsProcessor",
)
