"""
Tensegrity: a non-gradient cognitive architecture operating in SBERT
embedding space.

V3 architecture:
    SBERT embedding → NGC predictive coding → Hopfield memory
    → causal energy terms → Bayesian belief integration

Primary exports:
    CognitiveAgent  — the complete agent (replaces V1 TensegrityAgent)
    UnifiedField    — SBERT-native NGC + Hopfield
    CanonicalPipeline — benchmark/chat entry point
"""

from tensegrity.engine.unified_field import (
    UnifiedField,
    HopfieldMemoryBank,
    EnergyDecomposition,
)
from tensegrity.engine.ngc import (
    PredictiveCodingCircuit,
    LayerState,
)
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
from tensegrity.engine.agent import (
    CognitiveAgent,
    DEFAULT_MEDIATED_SCM_NAME,
)
from tensegrity.causal.scm import StructuralCausalModel
from tensegrity.causal.arena import CausalArena
from tensegrity.inference.free_energy import FreeEnergyEngine
from tensegrity.inference.belief_propagation import BeliefPropagator
from tensegrity.memory.episodic import EpisodicMemory
from tensegrity.memory.epistemic import EpistemicMemory

__version__ = "0.3.0"

__all__ = (
    "__version__",
    # Agent
    "CognitiveAgent",
    "DEFAULT_MEDIATED_SCM_NAME",
    # Engine
    "UnifiedField",
    "HopfieldMemoryBank",
    "EnergyDecomposition",
    "PredictiveCodingCircuit",
    "LayerState",
    # FHRR
    "FHRREncoder",
    "FHRRCodebook",
    "SemanticFHRRCodebook",
    "bind",
    "bundle",
    "unbind",
    "permute",
    # Causal
    "EnergyCausalArena",
    "CausalEnergyTerm",
    "TopologyMapper",
    "TopologyMapping",
    "VirtualParent",
    "StructuralCausalModel",
    "CausalArena",
    # Inference
    "FreeEnergyEngine",
    "BeliefPropagator",
    # Memory
    "EpisodicMemory",
    "EpistemicMemory",
)
