"""
Tensegrity: A Non-Gradient Cognitive Architecture

Built on:
  - Friston's Free Energy Principle (variational free energy minimization via belief propagation)
  - Pearl's Causal Calculus (structural causal models, do-calculus, counterfactuals)
  - Markov blankets as computational boundaries
  - Bayesian belief updating (no backpropagation)
  - Zipf-distributed memory access (power-law priority)
  - Morton-coded modality-agnostic sensory input

The "tension" in this system comes from competing causal models that each try to explain
observations. Resolution is through Bayesian model comparison — the model that minimizes
variational free energy (maximizes evidence) wins each cycle. No gradient descent anywhere.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                    MARKOV BLANKET                        │
  │  ┌──────────┐                          ┌──────────┐    │
  │  │ SENSORY  │  Morton-coded input       │  ACTIVE  │    │
  │  │  STATES  │ ─────────────────────┐    │  STATES  │    │
  │  └──────────┘                      │    └────┬─────┘    │
  │       │                            │         │          │
  │       ▼                            │         │          │
  │  ┌──────────────────────────┐      │         │          │
  │  │     BELIEF STATES        │      │         │          │
  │  │  Q(s) over hidden states │◄─────┘         │          │
  │  │  Updated via VFE min     │                │          │
  │  └─────────┬────────────────┘                │          │
  │            │                                 │          │
  │            ▼                                 │          │
  │  ┌──────────────────────────┐                │          │
  │  │   CAUSAL ARENA           │                │          │
  │  │  M₁ vs M₂ vs ... vs Mₖ  │────────────────┘          │
  │  │  SCMs compete via F      │                           │
  │  └─────────┬────────────────┘                           │
  │            │                                            │
  │            ▼                                            │
  │  ┌──────────────────────────────────────┐               │
  │  │         MEMORY SYSTEMS               │               │
  │  │  Epistemic │ Episodic │ Associative  │               │
  │  │  (beliefs) │ (traces) │ (Hopfield)   │               │
  │  │  Zipf-weighted access priority       │               │
  │  └──────────────────────────────────────┘               │
  └─────────────────────────────────────────────────────────┘
"""

__version__ = "0.1.0"

from tensegrity.core.agent import TensegrityAgent
from tensegrity.core.morton import MortonEncoder
from tensegrity.core.blanket import MarkovBlanket
from tensegrity.memory.epistemic import EpistemicMemory
from tensegrity.memory.episodic import EpisodicMemory
from tensegrity.memory.associative import AssociativeMemory
from tensegrity.causal.arena import CausalArena
from tensegrity.causal.scm import StructuralCausalModel
from tensegrity.inference.free_energy import FreeEnergyEngine
from tensegrity.inference.belief_propagation import BeliefPropagator
