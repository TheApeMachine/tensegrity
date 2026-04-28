"""
Tensegrity: a non-gradient cognitive architecture centered on a unified
energy landscape.

The primary engine is now the V2 ``UnifiedField`` stack:

    FHRR encoding -> hierarchical predictive coding -> Hopfield memory
    -> optional causal energy terms

Legacy V1 components remain importable from ``tensegrity.legacy.v1``:

    from tensegrity.legacy.v1 import TensegrityAgent, MortonEncoder

Top-level exports intentionally expose the unified field as the default
architecture. Deprecated V1 names are resolved lazily for migration only.
"""

from importlib import import_module
from typing import Any
import warnings

from tensegrity.engine import (
    UnifiedField,
    HopfieldMemoryBank,
    EnergyDecomposition,
    PredictiveCodingCircuit,
    LayerState,
    FHRREncoder,
    FHRRCodebook,
    SemanticFHRRCodebook,
    bind,
    bundle,
    unbind,
    permute,
    EnergyCausalArena,
    CausalEnergyTerm,
    TopologyMapper,
    TopologyMapping,
    VirtualParent,
    ScoringBridge,
    NGCLogitsProcessor,
)

__version__ = "0.1.0"

__all__ = (
    "__version__",
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

_LEGACY_EXPORTS = {
    "TensegrityAgent": ("tensegrity.legacy.v1.agent", "TensegrityAgent"),
    "MortonEncoder": ("tensegrity.legacy.v1.morton", "MortonEncoder"),
    "MarkovBlanket": ("tensegrity.legacy.v1.blanket", "MarkovBlanket"),
    "EpistemicMemory": ("tensegrity.memory.epistemic", "EpistemicMemory"),
    "EpisodicMemory": ("tensegrity.memory.episodic", "EpisodicMemory"),
    "AssociativeMemory": ("tensegrity.memory.associative", "AssociativeMemory"),
    "CausalArena": ("tensegrity.causal.arena", "CausalArena"),
    "StructuralCausalModel": ("tensegrity.causal.scm", "StructuralCausalModel"),
    "FreeEnergyEngine": ("tensegrity.inference.free_energy", "FreeEnergyEngine"),
    "BeliefPropagator": ("tensegrity.inference.belief_propagation", "BeliefPropagator"),
}


def __getattr__(name: str) -> Any:
    """Resolve deprecated top-level V1 names with an explicit migration warning."""
    target = _LEGACY_EXPORTS.get(name)

    if target is None:
        raise AttributeError(f"module 'tensegrity' has no attribute {name!r}")

    module_name, attr = target
    
    warnings.warn(
        f"tensegrity.{name} is not part of the primary V2 export surface. "
        f"Import {name} from {module_name} explicitly, or use "
        "tensegrity.UnifiedField for the unified energy engine.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    
    return value
