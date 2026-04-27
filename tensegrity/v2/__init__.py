"""
Tensegrity v2: Unified Energy Architecture.

The v2 architecture replaces v1's flat inference with:
  - FHRR-RNS encoding (compositional hypervectors with semantic grounding)
  - NGC predictive coding (hierarchical prediction errors, replaces flat POMDP)
  - Unified energy landscape (perception + memory + causal in one functional)
  - CausalArenaV2 (energy-based model competition, replaces log-likelihood arena)
  - Sentence-level semantic scoring via sentence-transformers

Public API:
  UnifiedField     — The main cognitive engine (FHRR → NGC → Hopfield → Causal)
  V2ScoringBridge  — Bridge to benchmark harness (scores choices by semantic similarity + NGC energy)
  NGCLogitsProcessor — LogitsProcessor for LLM generation grafting
  CausalArenaV2    — Energy-based causal model competition
  FHRREncoder      — Compositional hypervector encoder (semantic=True by default)
  SemanticFHRRCodebook — FHRR codebook grounded in sentence-transformer embeddings
  PredictiveCodingCircuit — Hierarchical NGC circuit
"""

__version__ = "0.2.0"

from tensegrity.v2.field import UnifiedField, HopfieldMemoryBank, EnergyDecomposition
from tensegrity.v2.ngc import PredictiveCodingCircuit, LayerState
from tensegrity.v2.fhrr import (
    FHRREncoder, FHRRCodebook, SemanticFHRRCodebook,
    bind, bundle, unbind, permute,
)
from tensegrity.v2.causal_energy import CausalArenaV2, CausalEnergyTerm
from tensegrity.v2.graft import V2ScoringBridge, NGCLogitsProcessor
