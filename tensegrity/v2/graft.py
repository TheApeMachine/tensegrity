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
    """
    
    supports_continuous_batching = False  # Stateful
    
    def __init__(self, field, tokenizer,
                 vocab_projections: Optional[List[np.ndarray]] = None,
                 scale: float = 1.0, energy_gate: float = 0.1,
                 max_settle_steps: int = 30, max_bias: float = 5.0):
        _ensure_torch()
        self.field = field
        self.tokenizer = tokenizer
        self.scale = scale
        self.energy_gate = energy_gate
        self.max_settle_steps = max_settle_steps
        self.max_bias = max_bias
        self.vocab_size = tokenizer.vocab_size
        self.projections = vocab_projections or self._build_projections()
        self._step_count = 0
        self._emissions = 0
        self._total_settle_steps = 0
    
    def _build_projections(self) -> List[np.ndarray]:
        projections = []
        rng = np.random.RandomState(7777)
        for ell, size in enumerate(self.field.ngc.layer_sizes):
            layer_weight = 2.0 ** ell
            P = rng.randn(self.vocab_size, size).astype(np.float64)
            P *= layer_weight / np.sqrt(size)
            projections.append(P)
        return projections
    
    def _tokens_to_observation(self, input_ids) -> np.ndarray:
        ids = input_ids[0].tolist()
        recent_ids = ids[-16:]
        text = self.tokenizer.decode(recent_ids, skip_special_tokens=True)
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(self.field.obs_dim, dtype=np.float64)
        fhrr_vec = self.field.encoder.encode_sequence(tokens)
        return self.field._fhrr_to_obs(fhrr_vec)
    
    def _error_to_bias(self) -> np.ndarray:
        bias = np.zeros(self.vocab_size, dtype=np.float64)
        for ell in range(self.field.ngc.n_layers):
            error = self.field.ngc.layers[ell].error
            if np.linalg.norm(error) < 1e-10:
                continue
            bias += self.projections[ell] @ error
        bias /= max(self.field.ngc.n_layers, 1)
        return bias
    
    def __call__(self, input_ids, scores):
        self._step_count += 1
        obs = self._tokens_to_observation(input_ids)
        settle_result = self.field.ngc.settle(obs, steps=self.max_settle_steps)
        self._total_settle_steps += self.max_settle_steps
        energy_trace = settle_result["energy_trace"]
        if len(energy_trace) >= 2:
            converged = abs(energy_trace[-1] - energy_trace[-2]) < self.energy_gate
        else:
            converged = False
        if not converged:
            return scores
        abstract = self.field.ngc.get_abstract_state(level=-1)
        self.field.memory.retrieve(abstract, steps=3)
        bias = self._error_to_bias()
        current_energy = settle_result["final_energy"]
        confidence = 1.0 / (1.0 + current_energy)
        bias *= self.scale * confidence
        np.clip(bias, -self.max_bias, self.max_bias, out=bias)
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
    v2 scoring bridge for the benchmark harness.
    
    The fundamental change from v1: instead of parsing text into keywords
    and doing Bayesian updates on discrete hypotheses, v2 scores choices
    using multi-signal compositional analysis:
    
      1. FHRR similarity: topical/entity overlap between prompt and choice
      2. NGC energy: hierarchical prediction error after settling
      3. Choice distinctiveness: how much a choice stands out from distractors
      4. Combined coherence: semantic consistency of prompt+choice
    
    With aggressive convergence gating:
      - Adaptive thresholds per n_choices (stricter for binary tasks)
      - Only emit non-uniform scores when signal quality is high
      - Abstain otherwise (return uniform → no effect on baseline)
    """
    
    def __init__(self, field=None, obs_dim: int = 256,
                 hidden_dims: Optional[List[int]] = None,
                 fhrr_dim: int = 2048,
                 ngc_settle_steps: int = 30,
                 ngc_learning_rate: float = 0.01,
                 hopfield_beta: float = 0.05,
                 confidence_threshold: float = 0.15,
                 context_settle_steps: int = 40,
                 choice_settle_steps: int = 25,
                 context_learning_epochs: int = 3):
        from tensegrity.v2.field import UnifiedField
        
        self.field = field or UnifiedField(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims or [128, 32],
            fhrr_dim=fhrr_dim,
            hopfield_beta=hopfield_beta,
            ngc_settle_steps=ngc_settle_steps,
            ngc_learning_rate=ngc_learning_rate,
        )
        
        self.confidence_threshold = confidence_threshold
        self.context_settle_steps = context_settle_steps
        self.choice_settle_steps = choice_settle_steps
        self.context_learning_epochs = context_learning_epochs
        self._total_scored = 0
        self._total_gated = 0
    
    def _tokenize_smart(self, text: str, max_tokens: int = 48) -> List[str]:
        import re
        tokens = re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?", text.lower())
        return tokens[-max_tokens:]
    
    def _encode_and_settle(self, tokens: List[str], 
                           settle_steps: int, learn: bool = False) -> Dict:
        if not tokens:
            return {
                "energy": 0.0,
                "obs_vec": np.zeros(self.field.obs_dim),
                "abstract_state": np.zeros(self.field.ngc.layer_sizes[-1]),
                "fhrr_vec": np.ones(self.field.fhrr_dim, dtype=np.complex64),
                "settle": {"final_energy": 0.0, "energy_trace": [0.0]},
            }
        fhrr_vec = self.field.encoder.encode_sequence(tokens)
        obs_vec = self.field._fhrr_to_obs(fhrr_vec)
        settle_result = self.field.ngc.settle(obs_vec, steps=settle_steps)
        if learn:
            self.field.ngc.learn(modulation=1.0)
        abstract_state = self.field.ngc.get_abstract_state(level=-1)
        return {
            "energy": settle_result["final_energy"],
            "obs_vec": obs_vec,
            "abstract_state": abstract_state,
            "fhrr_vec": fhrr_vec,
            "settle": settle_result,
        }
    
    def score_choices(self, prompt: str, choices: List[str]) -> Tuple[List[float], float]:
        """
        Score each choice using multi-signal v2 architecture.
        
        Combines FHRR similarity, NGC energy, choice distinctiveness, and
        combined coherence with aggressive convergence gating.
        """
        self._total_scored += 1
        n = len(choices)
        
        # === 1. ENCODE PROMPT ===
        prompt_tokens = self._tokenize_smart(prompt, max_tokens=64)
        prompt_fhrr = self.field.encoder.encode_sequence(prompt_tokens) if prompt_tokens else \
            np.ones(self.field.fhrr_dim, dtype=np.complex64)
        
        # === 2. FHRR SIMILARITY SIGNALS ===
        fhrr_sims = []
        combined_sims = []
        
        for choice in choices:
            choice_tokens = self._tokenize_smart(choice, max_tokens=32)
            choice_fhrr = self.field.encoder.encode_sequence(choice_tokens) if choice_tokens else \
                np.ones(self.field.fhrr_dim, dtype=np.complex64)
            sim = self.field.encoder.similarity(prompt_fhrr, choice_fhrr)
            fhrr_sims.append(sim)
            combined_tokens = (prompt_tokens + choice_tokens)[-64:]
            combined_fhrr = self.field.encoder.encode_sequence(combined_tokens)
            combined_sim = self.field.encoder.similarity(prompt_fhrr, combined_fhrr)
            combined_sims.append(combined_sim)
        
        # === 3. NGC SETTLING SIGNAL ===
        ctx = self._encode_and_settle(
            prompt_tokens, settle_steps=self.context_settle_steps, learn=True
        )
        saved_layers = [(l.z.copy(), l.z_bar.copy(), l.error.copy()) for l in self.field.ngc.layers]
        saved_W = [W.copy() for W in self.field.ngc.W]
        saved_E = [E.copy() for E in self.field.ngc.E]
        
        ngc_energies = []
        for choice in choices:
            for i, (z, z_bar, err) in enumerate(saved_layers):
                self.field.ngc.layers[i].z = z.copy()
                self.field.ngc.layers[i].z_bar = z_bar.copy()
                self.field.ngc.layers[i].error = err.copy()
            for i, W in enumerate(saved_W):
                self.field.ngc.W[i] = W.copy()
            for i, E in enumerate(saved_E):
                self.field.ngc.E[i] = E.copy()
            choice_tokens_full = self._tokenize_smart(prompt + " " + choice, max_tokens=64)
            r = self._encode_and_settle(choice_tokens_full, settle_steps=self.choice_settle_steps, learn=False)
            ngc_energies.append(-r["energy"])
        
        # === 4. INTER-CHOICE DISTINCTIVENESS ===
        choice_fhrr_vecs = []
        for choice in choices:
            ct = self._tokenize_smart(choice, max_tokens=32)
            cv = self.field.encoder.encode_sequence(ct) if ct else \
                np.ones(self.field.fhrr_dim, dtype=np.complex64)
            choice_fhrr_vecs.append(cv)
        
        distinctiveness = []
        for i in range(n):
            if n <= 1:
                distinctiveness.append(0.0)
                continue
            dists = []
            for j in range(n):
                if i != j:
                    sim_ij = self.field.encoder.similarity(choice_fhrr_vecs[i], choice_fhrr_vecs[j])
                    dists.append(1.0 - sim_ij)
            distinctiveness.append(np.mean(dists))
        
        # === 5. COMBINE SIGNALS ===
        fhrr_arr = np.array(fhrr_sims)
        combined_arr = np.array(combined_sims)
        ngc_arr = np.array(ngc_energies)
        distinct_arr = np.array(distinctiveness)
        
        def z_normalize(arr):
            std = arr.std()
            if std < 1e-10:
                return np.zeros_like(arr)
            return (arr - arr.mean()) / std
        
        fhrr_z = z_normalize(fhrr_arr)
        combined_z = z_normalize(combined_arr)
        ngc_z = z_normalize(ngc_arr)
        distinct_z = z_normalize(distinct_arr)
        
        scores_arr = (
            fhrr_z * 0.8 +
            ngc_z * 0.3 +
            combined_z * 0.2 +
            distinct_z * 0.4
        )
        
        # === 6. CONVERGENCE GATE ===
        fhrr_spread = float(fhrr_arr.max() - fhrr_arr.min())
        fhrr_mean = float(np.abs(fhrr_arr).mean())
        fhrr_cv = fhrr_spread / fhrr_mean if fhrr_mean > 1e-8 else 0.0
        
        shifted = scores_arr - scores_arr.max()
        probs = np.exp(shifted)
        probs_sum = probs.sum()
        probs = probs / probs_sum if probs_sum > 0 else np.ones(n) / n
        entropy = float(-np.sum(probs * np.log(probs + 1e-16)) / np.log(max(n, 2)))
        
        # Adaptive threshold: stricter for fewer choices
        if n <= 2:
            effective_threshold = self.confidence_threshold * 3.0
        elif n <= 3:
            effective_threshold = self.confidence_threshold * 2.0
        else:
            effective_threshold = self.confidence_threshold * 1.5
        
        should_abstain = (fhrr_cv < effective_threshold or entropy > 0.97)
        
        if should_abstain:
            self._total_gated += 1
            return [0.0] * n, 1.0
        
        return scores_arr.tolist(), entropy
    
    def reset(self):
        """Reset NGC state between samples. FHRR codebooks persist."""
        self.field.ngc._initialized = False
        self.field.ngc.layers = []
        self.field.memory.patterns = []
        self.field.memory._dirty = True
        self.field.memory._matrix = None
        self.field.energy_history = []
        self.field._step_count = 0
        
        # Fixed seed for reproducible NGC weight initialization
        rng = np.random.RandomState(12345)
        for ell in range(self.field.ngc.n_layers - 1):
            fan_in = self.field.ngc.layer_sizes[ell + 1]
            fan_out = self.field.ngc.layer_sizes[ell]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.field.ngc.W[ell] = rng.randn(fan_out, fan_in).astype(np.float64) * scale
            self.field.ngc.E[ell] = self.field.ngc.W[ell].T.copy()
    
    @property
    def statistics(self):
        return {
            "total_scored": self._total_scored,
            "total_gated": self._total_gated,
            "gate_rate": self._total_gated / max(self._total_scored, 1),
            "field": self.field.statistics,
        }
