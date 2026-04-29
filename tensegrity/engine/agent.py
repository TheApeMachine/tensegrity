"""
CognitiveAgent — V3 clean agent without legacy V1 baggage.

Replaces TensegrityAgent (legacy/v1/agent.py). Composes:
  - UnifiedField (SBERT-native NGC + Hopfield)
  - FreeEnergyEngine (discrete active inference)
  - CausalArena (competing SCMs)
  - EpisodicMemory (cross-item recall)

No Morton codes. No MarkovBlanket. No associative memory random projections.
"""

import hashlib
import numpy as np
from typing import Optional, Dict, List, Any
import logging

from tensegrity.engine.unified_field import UnifiedField
from tensegrity.inference.free_energy import FreeEnergyEngine
from tensegrity.causal.arena import CausalArena
from tensegrity.causal.scm import StructuralCausalModel
from tensegrity.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)

DEFAULT_MEDIATED_SCM_NAME = "mediated_causal"


class CognitiveAgent:
    """
    V3 cognitive agent operating in SBERT embedding space.

    Provides the same interface that CognitiveController expects
    (field, perceive(), arena, episodic, experience_replay, n_states)
    without any V1 legacy code.
    """

    def __init__(
        self,
        n_states: int = 16,
        n_observations: int = 32,
        n_actions: int = 4,
        planning_horizon: int = 3,
        precision: float = 4.0,
        context_dim: int = 32,
        # UnifiedField parameters
        obs_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        fhrr_dim: int = 2048,
        hopfield_beta: float = 0.05,
        ngc_settle_steps: int = 20,
        ngc_learning_rate: float = 0.005,
        sbert_dim: Optional[int] = None,
        # Legacy compat: these are accepted but ignored
        sensory_dims: int = 4,
        sensory_bits: int = 4,
        associative_dim: int = 64,
    ):
        self.n_states = n_states
        self.n_obs = n_observations
        self.n_actions = n_actions

        # === Unified Field (SBERT-native NGC + Hopfield) ===
        self.field = UnifiedField(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims or [128, 32],
            fhrr_dim=fhrr_dim,
            hopfield_beta=hopfield_beta,
            ngc_settle_steps=ngc_settle_steps,
            ngc_learning_rate=ngc_learning_rate,
            sbert_dim=sbert_dim,
        )

        # === Free Energy Engine (discrete active inference) ===
        self.engine = FreeEnergyEngine(
            n_states=n_states,
            n_observations=n_observations,
            n_actions=n_actions,
            planning_horizon=planning_horizon,
            precision=precision,
            policy_depth=min(planning_horizon, 3),
        )

        # === Causal Arena ===
        self.arena = CausalArena(
            prior_concentration=1.0,
            falsification_threshold=-100.0,
            min_models=2,
        )
        self._init_default_models()

        # === Episodic Memory ===
        self.episodic = EpisodicMemory(
            context_dim=context_dim,
            capacity=10000,
            drift_rate=0.95,
            encoding_strength=0.3,
        )

        # Agent state
        self._step_count = 0
        self._prev_belief: Optional[np.ndarray] = None

    def _init_default_models(self):
        """Initialize causal arena with default competing SCMs."""
        model_a = StructuralCausalModel(name="direct_causal")
        model_a.add_variable("state", n_values=self.n_states)
        model_a.add_variable("observation", n_values=self.n_obs, parents=["state"])

        model_b = StructuralCausalModel(name=DEFAULT_MEDIATED_SCM_NAME)
        model_b.add_variable("cause", n_values=self.n_states)
        model_b.add_variable("state", n_values=self.n_states, parents=["cause"])
        model_b.add_variable("observation", n_values=self.n_obs, parents=["state"])

        self.arena.register_model(model_a)
        self.arena.register_model(model_b)

    def perceive(self, raw_observation: np.ndarray) -> Dict[str, Any]:
        """
        Perception: observation → UnifiedField → active inference → causal arena.
        """
        self._step_count += 1
        raw = np.asarray(raw_observation, dtype=np.float64).ravel()

        # Run through unified field (SBERT-native settling)
        cycle = self.field.observe(raw, input_type="numeric")
        obs_vec = cycle["observation"]
        decomp = cycle["energy"]
        surprise = float(decomp.surprise)

        # Discrete observation index for the FEE
        h = hashlib.sha256(obs_vec.astype(np.float64).tobytes()).digest()
        obs_idx = int.from_bytes(h[:8], byteorder="big", signed=False) % max(self.n_obs, 1)

        # Active inference step
        A = getattr(self.engine, '_A', None)
        if A is None:
            # Use epistemic memory-style matrices if available,
            # otherwise use engine defaults
            from tensegrity.memory.epistemic import EpistemicMemory
            em = EpistemicMemory(
                n_states=self.n_states,
                n_observations=self.n_obs,
                n_actions=self.n_actions,
            )
            self._epistemic = em
            A, B, C, D = em.A, em.B, em.C, em.D
            log_A = em.log_A
            self.engine._A = A  # cache
        else:
            em = self._epistemic
            A, B, C, D = em.A, em.B, em.C, em.D
            log_A = em.log_A

        previous_action = self.engine.prev_action
        inference_result = self.engine.step(obs_idx, A, B, C, D, log_A)
        q_states = inference_result["belief_state"]
        F = float(inference_result["free_energy"])

        # Update epistemic memory
        em.update_likelihood(obs_idx, q_states)
        if previous_action is not None and self._prev_belief is not None:
            em.update_transition(self._prev_belief, q_states, previous_action)
        self._prev_belief = q_states.copy()

        # Causal arena competition
        causal_obs = {
            "state": int(np.argmax(q_states)),
            "observation": obs_idx,
        }
        if DEFAULT_MEDIATED_SCM_NAME in self.arena.models:
            causal_obs["cause"] = int(np.argmax(q_states))
        arena_result = self.arena.compete(causal_obs)

        # Episodic memory encoding
        self.episodic.encode(
            observation=raw,
            morton_code=np.array([obs_idx], dtype=np.int64),
            belief_state=q_states,
            action=int(inference_result["action"]),
            surprise=surprise,
            free_energy=F,
            metadata={
                "obs_idx": obs_idx,
                "field_energy": float(decomp.total),
            },
        )

        return {
            "step": self._step_count,
            "observation_index": obs_idx,
            "belief_state": q_states,
            "free_energy": F,
            "surprise": surprise,
            "action": inference_result["action"],
            "action_confidence": inference_result["action_confidence"],
            "arena": arena_result,
            "epistemic_value": self.engine.epistemic_value,
            "field_cycle": cycle,
        }

    def experience_replay(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Replay past episodes to strengthen beliefs."""
        episodes = self.episodic.replay(n_episodes)
        em = getattr(self, '_epistemic', None)
        if em is not None:
            for ep in episodes:
                obs_idx = ep.metadata.get('obs_idx', 0)
                em.update_likelihood(obs_idx, ep.belief_state)
        return {
            'episodes_replayed': len(episodes),
            'mean_surprise': np.mean([ep.surprise for ep in episodes]) if episodes else 0,
        }

    def add_causal_model(self, model: StructuralCausalModel):
        """Add a competing causal model to the arena."""
        self.arena.register_model(model)
