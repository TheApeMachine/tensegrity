"""
v2 Graft: NGC prediction errors → per-step logit biases during LLM decoding.

This bridges the gap between the manifold approach (continuous constraint
propagation inside the decode loop) and Tensegrity's causal reasoning
(epistemically grounded beliefs about what's true).

At each decode step:
  1. The generated tokens so far are encoded as an FHRR sequence
  2. The NGC circuit settles on this observation (minimizing VFE)
  3. The prediction error at each NGC layer is computed
  4. These errors are projected into vocabulary space as logit biases

The projection works because:
  - Layer 0 errors (sensory) → token-level constraints (word choice)
  - Layer 1 errors (hidden) → phrase-level constraints (coherence)
  - Layer L errors (abstract) → semantic constraints (topic, logic)

Each layer's projection is a fixed random matrix (no learning needed
at the graft interface — all learning happens inside the NGC circuit).

Convergence gating:
  - Only emit bias when NGC has settled (energy delta < threshold)
  - Scale bias by inverse entropy (confident beliefs → strong bias)
  - Never worse than base: ungated fallback to native LLM logits
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Set, Tuple
import math
import logging

logger = logging.getLogger(__name__)

# Lazy torch import
torch = None
def _ensure_torch():
    global torch
    if torch is None:
        import importlib
        torch = importlib.import_module('torch')


class NGCLogitsProcessor:
    """
    HuggingFace LogitsProcessor that runs NGC settling at each decode step.
    
    This is the v2 equivalent of TensegrityLogitsProcessor, but instead of
    projecting flat hypothesis posteriors, it projects hierarchical prediction
    errors from the NGC circuit.
    
    The manifold ran ~47 internal steps per decode step until coherence > 0.96.
    We do the same: the NGC circuit settles until energy delta < threshold,
    then projects its state into logit space.
    """
    
    supports_continuous_batching = False  # Stateful
    
    def __init__(self,
                 field,  # UnifiedField instance
                 tokenizer,
                 vocab_projections: Optional[List[np.ndarray]] = None,
                 scale: float = 1.0,
                 energy_gate: float = 0.1,
                 max_settle_steps: int = 30,
                 max_bias: float = 5.0):
        """
        Args:
            field: UnifiedField instance (owns NGC + FHRR + Hopfield)
            tokenizer: HuggingFace tokenizer
            vocab_projections: Per-NGC-layer projection matrices to vocab space.
                             If None, generated randomly (fixed, not learned).
            scale: Overall bias magnitude multiplier
            energy_gate: Only emit bias when NGC energy change < this per step
            max_settle_steps: NGC settling budget per decode step
            max_bias: Clamp per-token bias magnitude
        """
        _ensure_torch()
        
        self.field = field
        self.tokenizer = tokenizer
        self.scale = scale
        self.energy_gate = energy_gate
        self.max_settle_steps = max_settle_steps
        self.max_bias = max_bias
        
        self.vocab_size = tokenizer.vocab_size
        
        # Build per-layer projection matrices: NGC layer dim → vocab_size
        # These are fixed random projections, not learned
        if vocab_projections is not None:
            self.projections = vocab_projections
        else:
            self.projections = self._build_projections()
        
        # Tracking
        self._step_count = 0
        self._emissions = 0
        self._total_settle_steps = 0
    
    def _build_projections(self) -> List[np.ndarray]:
        """
        Build random projection matrices from NGC error space to vocab space.
        
        Higher layers get stronger projection weights (semantic > surface).
        Layer weights: [1.0, 2.0, 4.0, ...] (doubling per level).
        """
        projections = []
        rng = np.random.RandomState(7777)
        
        for ell, size in enumerate(self.field.ngc.layer_sizes):
            # Random projection: (vocab_size, layer_size)
            # Scaled by 1/sqrt(layer_size) for variance normalization
            # Higher layers get more weight
            layer_weight = 2.0 ** ell
            P = rng.randn(self.vocab_size, size).astype(np.float64)
            P *= layer_weight / np.sqrt(size)
            projections.append(P)
        
        return projections
    
    def _tokens_to_observation(self, input_ids) -> np.ndarray:
        """
        Convert generated tokens so far into an FHRR observation vector,
        then project to NGC sensory space.
        
        Uses the last N tokens as a sequence encoding.
        """
        # Decode last 16 tokens to text
        ids = input_ids[0].tolist()
        recent_ids = ids[-16:]  # Last 16 tokens
        text = self.tokenizer.decode(recent_ids, skip_special_tokens=True)
        tokens = text.lower().split()
        
        if not tokens:
            return np.zeros(self.field.obs_dim, dtype=np.float64)
        
        # Encode as FHRR sequence → project to NGC observation space
        fhrr_vec = self.field.encoder.encode_sequence(tokens)
        obs_vec = self.field._fhrr_to_obs(fhrr_vec)
        
        return obs_vec
    
    def _error_to_bias(self) -> np.ndarray:
        """
        Project NGC prediction errors into vocabulary space.
        
        bias = Σ_ℓ P_ℓ · error_ℓ
        
        Where P_ℓ is the fixed random projection for layer ℓ,
        and error_ℓ is the precision-weighted prediction error.
        
        Low-level errors → token-level biases (surface form)
        High-level errors → semantic biases (topic/logic)
        """
        bias = np.zeros(self.vocab_size, dtype=np.float64)
        
        for ell in range(self.field.ngc.n_layers):
            error = self.field.ngc.layers[ell].error
            if np.linalg.norm(error) < 1e-10:
                continue
            
            # Project error into vocab space
            layer_bias = self.projections[ell] @ error
            bias += layer_bias
        
        # Normalize by number of layers
        bias /= max(self.field.ngc.n_layers, 1)
        
        return bias
    
    def __call__(self, input_ids, scores):
        """
        Called at each decode step by model.generate().
        
        1. Convert generated tokens → FHRR observation
        2. Settle NGC circuit on this observation
        3. If converged: project prediction errors into logit biases
        4. If not: pass through unmodified
        """
        self._step_count += 1
        
        # Convert tokens to observation
        obs = self._tokens_to_observation(input_ids)
        
        # Settle NGC
        settle_result = self.field.ngc.settle(obs, steps=self.max_settle_steps)
        self._total_settle_steps += self.max_settle_steps
        
        # Check convergence: did the energy stabilize?
        energy_trace = settle_result["energy_trace"]
        if len(energy_trace) >= 2:
            energy_delta = abs(energy_trace[-1] - energy_trace[-2])
            converged = energy_delta < self.energy_gate
        else:
            converged = False
        
        if not converged:
            return scores  # Graceful fallback — native LLM behavior
        
        # Query Hopfield memory with abstract state (top NGC layer)
        abstract = self.field.ngc.get_abstract_state(level=-1)
        retrieved, mem_energy = self.field.memory.retrieve(abstract, steps=3)
        
        # Compute bias from prediction errors
        bias = self._error_to_bias()
        
        # Scale by inverse energy (lower energy = more confident = stronger bias)
        current_energy = settle_result["final_energy"]
        confidence = 1.0 / (1.0 + current_energy)  # Sigmoid-like scaling
        bias *= self.scale * confidence
        
        # Clamp
        np.clip(bias, -self.max_bias, self.max_bias, out=bias)
        
        # Convert to torch and apply
        bias_tensor = torch.tensor(bias, device=scores.device, dtype=scores.dtype)
        
        self._emissions += 1
        
        return scores + bias_tensor.unsqueeze(0)
    
    @property
    def statistics(self):
        return {
            "decode_steps": self._step_count,
            "emissions": self._emissions,
            "emission_rate": self._emissions / max(self._step_count, 1),
            "total_settle_steps": self._total_settle_steps,
            "avg_settle_per_decode": self._total_settle_steps / max(self._step_count, 1),
            "ngc_energy": self.field.ngc.total_energy,
            "memory_patterns": self.field.memory.n_patterns,
        }


class V2ScoringBridge:
    """
    Bridge between v2 architecture and the benchmark harness.
    
    Converts a TaskSample's choices into FHRR observations,
    runs the NGC circuit on each, and scores choices by
    prediction error: lower error = better fit = higher score.
    
    This replaces v1's flat Bayesian posterior scoring with
    hierarchical predictive coding scoring.
    """
    
    def __init__(self, field=None, obs_dim: int = 128, 
                 hidden_dims: Optional[List[int]] = None):
        from tensegrity.v2.field import UnifiedField
        
        self.field = field or UnifiedField(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims or [64, 16],
            fhrr_dim=1024,
            hopfield_beta=0.05,
            ngc_settle_steps=20,
            ngc_learning_rate=0.005,
        )
    
    def score_choices(self, prompt: str, choices: List[str]) -> Tuple[List[float], float]:
        """
        Score each choice via v2 predictive coding.
        
        For each choice:
          1. Encode prompt as FHRR → settle NGC (establish context beliefs)
          2. Encode prompt+choice as FHRR → settle NGC (observe with choice)
          3. Score = negative prediction error (lower error = better fit)
        
        Returns:
            (scores, entropy) where scores[i] = score for choice i
        """
        # First, establish context by observing the prompt
        prompt_tokens = prompt.lower().split()[:32]  # Cap at 32 tokens
        if prompt_tokens:
            self.field.observe(prompt_tokens, input_type="tokens")
        
        # Score each choice by prediction error
        scores = []
        for choice in choices:
            choice_tokens = (prompt + " " + choice).lower().split()[-32:]
            
            # Create a fresh copy of the NGC state for counterfactual scoring
            # (we don't want scoring one choice to affect scoring another)
            saved_layers = [
                (l.z.copy(), l.z_bar.copy(), l.error.copy()) 
                for l in self.field.ngc.layers
            ]
            
            # Observe the choice
            fhrr_vec = self.field.encoder.encode_sequence(choice_tokens)
            obs = self.field._fhrr_to_obs(fhrr_vec)
            settle_result = self.field.ngc.settle(obs, steps=10)
            
            # Score = negative energy (lower energy = better explanation)
            score = -settle_result["final_energy"]
            scores.append(score)
            
            # Restore NGC state
            for i, (z, z_bar, err) in enumerate(saved_layers):
                self.field.ngc.layers[i].z = z
                self.field.ngc.layers[i].z_bar = z_bar
                self.field.ngc.layers[i].error = err
        
        # Entropy of softmax(scores) for confidence estimation
        scores_arr = np.array(scores)
        shifted = scores_arr - scores_arr.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-16)) / np.log(max(len(probs), 2)))
        
        return scores, entropy
    
    def reset(self):
        """Reset the field's NGC state between samples."""
        self.field.ngc._initialized = False
        self.field.ngc.layers = []
