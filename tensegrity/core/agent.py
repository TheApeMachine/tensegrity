"""
TensegrityAgent: The complete cognitive architecture.

Integrates all components into a single agent that:
  1. Receives modality-agnostic observations (Morton-encoded)
  2. Updates beliefs via free energy minimization (no gradients)
  3. Maintains three memory systems (epistemic, episodic, associative)
  4. Runs competing causal models in the arena
  5. Selects actions that minimize expected free energy
  6. Generates epistemic actions to resolve model uncertainty

The name "Tensegrity" comes from the architectural principle where
structural integrity comes from the balance of tension and compression.
Here, the system's cognitive integrity comes from the tension between
competing causal models (compression = model evidence, tension = model
disagreement) balanced by the free energy principle.
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple
import logging

from tensegrity.core.morton import MortonEncoder
from tensegrity.core.blanket import MarkovBlanket
from tensegrity.memory.epistemic import EpistemicMemory
from tensegrity.memory.episodic import EpisodicMemory
from tensegrity.memory.associative import AssociativeMemory
from tensegrity.causal.arena import CausalArena
from tensegrity.causal.scm import StructuralCausalModel
from tensegrity.inference.free_energy import FreeEnergyEngine
from tensegrity.engine.unified_field import UnifiedField

logger = logging.getLogger(__name__)


class TensegrityAgent:
    """
    A non-gradient cognitive agent.
    
    The agent perceives the world through Morton-coded observations,
    maintains beliefs via Bayesian updates, resolves competing causal
    explanations in an adversarial arena, and acts to minimize
    expected free energy.
    
    No backpropagation. No gradient descent. No optimizer state.
    
    All learning is:
      - Dirichlet counting (epistemic memory)
      - Context drift (episodic memory)
      - Energy minimization via Hopfield dynamics (associative memory)
      - Bayesian model comparison (causal arena)
      - Fixed-point iteration (belief propagation)
    """
    
    def __init__(self, 
                 n_states: int = 16,
                 n_observations: int = 32,
                 n_actions: int = 4,
                 sensory_dims: int = 4,
                 sensory_bits: int = 8,
                 context_dim: int = 64,
                 associative_dim: int = 128,
                 planning_horizon: int = 3,
                 precision: float = 4.0,
                 zipf_exponent: float = 1.0):
        """
        Args:
            n_states: Number of hidden states in the generative model
            n_observations: Number of observation categories
            n_actions: Number of possible actions
            sensory_dims: Dimensionality of raw sensory input
            sensory_bits: Bits per dimension for Morton encoding
            context_dim: Dimensionality of episodic context vectors
            associative_dim: Dimensionality of associative memory patterns
            planning_horizon: How far ahead to plan
            precision: Inverse temperature for policy selection
            zipf_exponent: Controls power-law memory access
        """
        self.n_states = n_states
        self.n_obs = n_observations
        self.n_actions = n_actions
        
        # === SENSORY INTERFACE (Markov Blanket) ===
        self.encoder = MortonEncoder(n_dims=sensory_dims, bits_per_dim=sensory_bits)
        self.blanket = MarkovBlanket(
            encoder=self.encoder,
            n_sensory_channels=1,
            n_active_channels=1,
            observation_buffer_size=256
        )
        
        # === MEMORY SYSTEMS ===
        self.epistemic = EpistemicMemory(
            n_states=n_states,
            n_observations=n_observations,
            n_actions=n_actions,
            zipf_exponent=zipf_exponent
        )
        
        self.episodic = EpisodicMemory(
            context_dim=context_dim,
            capacity=10000,
            drift_rate=0.95,
            encoding_strength=0.3,
            zipf_exponent=zipf_exponent
        )
        
        self.associative = AssociativeMemory(
            pattern_dim=associative_dim,
            beta=precision,
            max_patterns=5000,
            zipf_exponent=zipf_exponent
        )
        
        # === INFERENCE ENGINE ===
        self.engine = FreeEnergyEngine(
            n_states=n_states,
            n_observations=n_observations,
            n_actions=n_actions,
            planning_horizon=planning_horizon,
            precision=precision,
            policy_depth=min(planning_horizon, 3)
        )
        
        # === CAUSAL ARENA ===
        self.arena = CausalArena(
            prior_concentration=1.0,
            falsification_threshold=-100.0,
            min_models=2
        )
        
        # === AGENT STATE ===
        self._step_count = 0
        self._total_surprise = 0.0
        self._total_free_energy = 0.0
        self._prev_belief_for_transition: Optional[np.ndarray] = None
        
        # Initialize with default competing models
        self._init_default_models()
        
        # Single perceptual substrate: FHRR → NGC → Hopfield (replaces parallel Morton-sense path).
        self.field = UnifiedField(
            obs_dim=256,
            hidden_dims=[128, 32],
            fhrr_dim=2048,
            hopfield_beta=0.01,
            ngc_settle_steps=20,
            ngc_learning_rate=0.005,
        )
    
    def _init_default_models(self):
        """
        Initialize the causal arena with default competing models.
        
        We start with two models that represent competing hypotheses
        about the causal structure of observations:
          Model A: "States cause observations directly" (simple)
          Model B: "States mediate between hidden causes and observations" (complex)
        """
        # Model A: Simple — direct state-observation link
        model_a = StructuralCausalModel(name="direct_causal")
        model_a.add_variable("state", n_values=self.n_states)
        model_a.add_variable("observation", n_values=self.n_obs, 
                           parents=["state"])
        
        # Model B: Mediated — hidden cause → state → observation
        model_b = StructuralCausalModel(name="mediated_causal")
        model_b.add_variable("cause", n_values=self.n_states)
        model_b.add_variable("state", n_values=self.n_states,
                           parents=["cause"])
        model_b.add_variable("observation", n_values=self.n_obs,
                           parents=["state"])
        
        self.arena.register_model(model_a)
        self.arena.register_model(model_b)
    
    def _morton_to_obs_index(self, morton_codes: np.ndarray) -> int:
        """Map Morton codes to observation index via modular hashing."""
        if isinstance(morton_codes, (int, np.integer)):
            return int(morton_codes) % self.n_obs
        # For multiple codes, hash the combination
        combined = 0
        for code in morton_codes:
            combined ^= int(code)
        return combined % self.n_obs
    
    def _obs_to_associative_pattern(self, observation: int,
                                     belief_state: np.ndarray) -> np.ndarray:
        """Project observation + belief into associative memory space."""
        rng = np.random.RandomState(observation)
        
        # Combine observation (one-hot) and belief state
        obs_vec = np.zeros(self.n_obs)
        obs_vec[observation] = 1.0
        combined = np.concatenate([obs_vec, belief_state])
        
        # Random projection to associative_dim
        W = rng.randn(self.associative.dim, len(combined)) / np.sqrt(len(combined))
        pattern = W @ combined
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern /= norm
        return pattern
    
    def perceive(self, raw_observation: np.ndarray) -> Dict[str, Any]:
        """
        One perception path: numeric vector → UnifiedField (FHRR / NGC / Hopfield)
        → discrete observation index → active inference engine → causal arena.

        Episodic and classical Hopfield associative traces are not written here;
        memory consolidation for this path lives inside UnifiedField.
        """
        self._step_count += 1
        raw = np.asarray(raw_observation, dtype=np.float64).ravel()

        cycle = self.field.observe(raw, input_type="numeric")
        obs_vec = cycle["observation"]
        decomp = cycle["energy"]
        surprise = float(decomp.surprise)

        # Deterministic discrete index for generative-model matrices
        v = np.arange(1, len(obs_vec) + 1, dtype=np.float64)
        obs_idx = int(np.abs(np.dot(obs_vec, v))) % max(self.n_obs, 1)

        A = self.epistemic.A
        B = self.epistemic.B
        C = self.epistemic.C
        D = self.epistemic.D
        log_A = self.epistemic.log_A

        inference_result = self.engine.step(obs_idx, A, B, C, D, log_A)
        q_states = inference_result["belief_state"]
        F = float(inference_result["free_energy"])

        self.epistemic.update_likelihood(obs_idx, q_states)
        if (self.engine.prev_action is not None
                and self._prev_belief_for_transition is not None):
            self.epistemic.update_transition(
                self._prev_belief_for_transition, q_states,
                self.engine.prev_action)
        self._prev_belief_for_transition = q_states.copy()

        causal_obs = {
            "state": int(np.argmax(q_states)),
            "observation": obs_idx,
        }
        if "mediated_causal" in self.arena.models:
            causal_obs["cause"] = int(np.argmax(q_states))

        arena_result = self.arena.compete(causal_obs)

        morton_codes = np.array([obs_idx], dtype=np.int64)
        self.blanket.surprise = surprise

        self._total_surprise += surprise
        self._total_free_energy += F

        return {
            "step": self._step_count,
            "morton_codes": morton_codes,
            "observation_index": obs_idx,
            "belief_state": q_states,
            "free_energy": F,
            "surprise": surprise,
            "action": inference_result["action"],
            "action_confidence": inference_result["action_confidence"],
            "arena": arena_result,
            "associative_energy": float(decomp.memory),
            "epistemic_value": self.engine.epistemic_value,
            "pragmatic_value": self.engine.pragmatic_value,
            "field_cycle": cycle,
        }
    
    def act(self) -> Dict[str, Any]:
        """
        Select and emit an action through the active boundary.
        
        Uses the policy posterior from the last perception step.
        Also checks if an epistemic action (experiment) would be more valuable.
        """
        # Check if an experiment would help resolve causal tension
        experiment = self.arena.suggest_experiment()
        
        # Compare epistemic value of experiment vs pragmatic action
        if (experiment['expected_info_gain'] > 0.1 and 
            self.arena.current_tension > 0.5):
            # Epistemic action: run an experiment to resolve tension
            return {
                'type': 'epistemic',
                'experiment': experiment,
                'reason': 'High causal tension — exploring to resolve',
                'tension': self.arena.current_tension,
            }
        
        # Pragmatic action: act to achieve preferences
        action, confidence = self.engine.select_action()
        action_dist = np.zeros(self.n_actions)
        for pi_idx, policy in enumerate(self.engine.policies):
            if len(policy) > 0:
                action_dist[policy[0]] += self.engine.q_policies[pi_idx]
        if action_dist.sum() > 0:
            action_dist /= action_dist.sum()
        
        selected = self.blanket.act(action_dist)
        
        return {
            'type': 'pragmatic',
            'action': selected,
            'confidence': confidence,
            'action_distribution': action_dist,
            'free_energy': self.engine.F_history[-1] if self.engine.F_history else None,
        }
    
    def experience_replay(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Replay past episodes to strengthen beliefs.
        
        This is the offline learning loop: re-process past observations
        through the epistemic memory to update Dirichlet parameters.
        Weighted by surprise — surprising experiences teach more.
        """
        episodes = self.episodic.replay(n_episodes)
        
        total_update = 0.0
        for ep in episodes:
            obs_idx = ep.metadata.get('obs_idx', 0)
            self.epistemic.update_likelihood(obs_idx, ep.belief_state)
            total_update += 1.0
        
        return {
            'episodes_replayed': len(episodes),
            'mean_surprise': np.mean([ep.surprise for ep in episodes]) if episodes else 0,
            'epistemic_entropy': self.epistemic.entropy(),
        }
    
    def introspect(self) -> Dict[str, Any]:
        """
        Full introspection: report on all system components.
        """
        return {
            'step': self._step_count,
            'average_surprise': self._total_surprise / max(self._step_count, 1),
            'average_free_energy': self._total_free_energy / max(self._step_count, 1),
            'inference': self.engine.statistics,
            'arena': self.arena.statistics,
            'epistemic_memory': {
                'entropy': self.epistemic.entropy(),
                'access_distribution': self.epistemic.get_access_distribution(),
            },
            'episodic_memory': self.episodic.statistics,
            'associative_memory': self.associative.statistics,
            'blanket': self.blanket.state,
            'tension_trajectory': self.arena.tension_history[-20:],
            'free_energy_trajectory': self.engine.F_history[-20:],
        }
    
    def add_causal_model(self, model: StructuralCausalModel):
        """Add a new competing causal model to the arena."""
        self.arena.register_model(model)
    
    def counterfactual(self, evidence: Dict[str, int],
                       intervention: Dict[str, int],
                       query: List[str]) -> Dict[str, Any]:
        """
        Ask: "What would have happened if we had done X instead?"
        
        Each competing model gives its own answer. Disagreement = tension.
        """
        return self.arena.counterfactual_comparison(evidence, intervention, query)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TensegrityAgent':
        """Create an agent from a configuration dictionary."""
        return cls(**config)
    
    def __repr__(self):
        return (f"TensegrityAgent(states={self.n_states}, obs={self.n_obs}, "
                f"actions={self.n_actions}, step={self._step_count}, "
                f"tension={self.arena.current_tension:.3f})")
