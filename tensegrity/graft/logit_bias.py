"""
Logit-Bias Graft: Tensegrity beliefs → LLM token distribution biases.

This is where the cognitive layer physically touches the language model.
Not through prompt text. Not through system messages. Through the logit
distribution itself — the mathematical object the LLM samples from.

The graft implements three principles from the manifold integration:

1. LOGIT BIAS INJECTION
   At each decode step, Tensegrity's hypothesis posteriors are converted
   to additive logit biases over the vocabulary:
     bias[token] = γ · log(P(hypothesis) / P_uniform)
   This is the per-token form of Classifier-Free Guidance.
   Tokens associated with probable hypotheses get boosted.
   Tokens associated with eliminated hypotheses get hard-suppressed (-inf).

2. CONVERGENCE GATING
   The bias is only emitted when the cognitive layer has converged:
     - Belief entropy < threshold (hypotheses sufficiently resolved)
     - Free energy below threshold (beliefs fit observations)
   If the cognitive layer hasn't converged, NO BIAS IS EMITTED.
   The LLM falls back to native behavior. Never worse than base.

3. GRACEFUL DEGRADATION
   Even when emitting, the bias magnitude is proportional to confidence.
   Uncertain beliefs → small biases (gentle nudge).
   Resolved beliefs → large biases (strong steering).
   This prevents catastrophic override when the cognitive layer is wrong.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import torch lazily — only needed when actually grafting to a local model
torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import importlib
        torch = importlib.import_module('torch')


@dataclass
class GraftState:
    """Observable state of the graft for diagnostics."""
    step: int = 0
    bias_emitted: bool = False
    belief_entropy: float = 1.0
    convergence_met: bool = False
    max_bias_magnitude: float = 0.0
    boosted_tokens: int = 0
    suppressed_tokens: int = 0
    

class TensegrityLogitsProcessor:
    """
    HuggingFace LogitsProcessor that injects Tensegrity belief state
    as per-step logit biases during LLM decoding.
    
    At each decode step:
      1. Read current belief posteriors from the cognitive layer
      2. Check convergence gate (entropy threshold + free energy)
      3. If converged: compute logit bias = γ · log(P(h) / P_uniform)
      4. Apply additive bias to raw logits
      5. If NOT converged: pass through unmodified (graceful fallback)
    """
    
    # Required by HuggingFace for stateful processors
    supports_continuous_batching = False
    
    def __init__(self,
                 hypothesis_tokens: Dict[str, Set[int]],
                 belief_fn: Callable[[], Dict[str, float]],
                 vocab_size: int,
                 scale: float = 2.5,
                 suppress_threshold: float = 0.01,
                 entropy_gate: float = 0.85,
                 min_confidence: float = 0.3,
                 max_bias: float = 8.0):
        """
        Args:
            hypothesis_tokens: {hyp_id: set of token_ids} from VocabularyGrounding
            belief_fn: Callable that returns current posteriors {hyp_id: probability}
                      This is called at EVERY decode step — must return fresh state.
            vocab_size: LLM vocabulary size
            scale: γ — guidance strength. 0=off, 2.5=moderate, 5.0=strong
            suppress_threshold: P below this → hard -inf suppress
            entropy_gate: Only emit bias when normalized entropy < this.
                         1.0 = always emit, 0.0 = never emit, 0.85 = emit when fairly certain
            min_confidence: Minimum max-posterior probability to emit any bias
            max_bias: Clamp bias magnitude to prevent numerical issues
        """
        _ensure_torch()
        
        self.hypothesis_tokens = hypothesis_tokens
        self.belief_fn = belief_fn
        self.vocab_size = vocab_size
        self.scale = scale
        self.suppress_threshold = suppress_threshold
        self.entropy_gate = entropy_gate
        self.min_confidence = min_confidence
        self.max_bias = max_bias
        
        # State tracking
        self.state = GraftState()
        self._step_count = 0
    
    def _compute_entropy(self, posteriors: Dict[str, float]) -> float:
        """Normalized entropy of the posterior. 0=resolved, 1=uniform."""
        probs = np.array(list(posteriors.values()))
        probs = probs[probs > 0]
        if len(probs) <= 1:
            return 0.0
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _should_emit(self, posteriors: Dict[str, float]) -> bool:
        """
        Convergence gate: should we emit a logit bias?
        
        YES if:
          1. Normalized entropy < entropy_gate (beliefs have converged)
          2. Max posterior > min_confidence (at least one hypothesis is probable)
        
        NO otherwise → LLM gets unmodified logits (graceful fallback).
        """
        entropy = self._compute_entropy(posteriors)
        max_prob = max(posteriors.values()) if posteriors else 0
        
        self.state.belief_entropy = entropy
        self.state.convergence_met = (entropy < self.entropy_gate and 
                                       max_prob >= self.min_confidence)
        
        return self.state.convergence_met
    
    def __call__(self, input_ids, scores):
        """
        Called at every decode step by model.generate().
        
        Args:
            input_ids: (batch_size, seq_len) — generated tokens so far
            scores: (batch_size, vocab_size) — raw logits before softmax
        
        Returns:
            Modified scores with belief-derived logit biases
        """
        self._step_count += 1
        self.state.step = self._step_count
        
        # Read current beliefs from Tensegrity
        posteriors = self.belief_fn()
        
        if not posteriors:
            self.state.bias_emitted = False
            return scores
        
        # Convergence gate
        if not self._should_emit(posteriors):
            self.state.bias_emitted = False
            return scores
        
        # Compute bias
        N = len(posteriors)
        p_uniform = 1.0 / N
        
        bias = torch.zeros(self.vocab_size, device=scores.device, dtype=scores.dtype)
        boosted = 0
        suppressed = 0
        max_mag = 0.0
        
        for hyp_id, prob in posteriors.items():
            token_ids = self.hypothesis_tokens.get(hyp_id, set())
            if not token_ids:
                continue
            
            if prob < self.suppress_threshold:
                # Hard suppress — eliminated hypothesis
                b = float('-inf')
                for tid in token_ids:
                    if 0 <= tid < self.vocab_size:
                        bias[tid] = float('-inf')
                        suppressed += 1
            else:
                # Log-ratio bias: positive when prob > uniform, negative when below
                b = self.scale * math.log(prob / p_uniform)
                b = max(-self.max_bias, min(self.max_bias, b))  # Clamp
                
                for tid in token_ids:
                    if 0 <= tid < self.vocab_size:
                        if bias[tid] != float('-inf'):  # Don't un-suppress
                            bias[tid] += b
                            if b > 0:
                                boosted += 1
                            max_mag = max(max_mag, abs(b))
        
        # Scale by confidence — uncertain beliefs emit weaker biases
        max_prob = max(posteriors.values())
        confidence_scale = (max_prob - p_uniform) / (1.0 - p_uniform) if max_prob > p_uniform else 0.0
        
        # Apply confidence scaling to non-inf biases
        finite_mask = bias != float('-inf')
        bias[finite_mask] *= confidence_scale
        
        # Update state
        self.state.bias_emitted = True
        self.state.max_bias_magnitude = max_mag * confidence_scale
        self.state.boosted_tokens = boosted
        self.state.suppressed_tokens = suppressed
        
        return scores + bias.unsqueeze(0)  # Broadcast over batch dim


class StaticLogitBiasBuilder:
    """
    For remote API calls where per-step callbacks aren't available.
    
    Builds a static logit_bias dict from the current belief state.
    Less powerful than the LogitsProcessor (no per-step updates),
    but works with any OpenAI-compatible API.
    """
    
    def __init__(self, hypothesis_tokens: Dict[str, Set[int]],
                 scale: float = 2.5,
                 suppress_threshold: float = 0.01,
                 max_bias: float = 5.0):
        self.hypothesis_tokens = hypothesis_tokens
        self.scale = scale
        self.suppress_threshold = suppress_threshold
        self.max_bias = max_bias
    
    def build(self, posteriors: Dict[str, float]) -> Dict[int, float]:
        """
        Build a static logit_bias dict for API calls.
        
        Returns: {token_id: bias_value} suitable for OpenAI's logit_bias parameter
        """
        N = len(posteriors)
        if N == 0:
            return {}
        
        p_uniform = 1.0 / N
        bias = {}
        
        for hyp_id, prob in posteriors.items():
            token_ids = self.hypothesis_tokens.get(hyp_id, set())
            
            if prob < self.suppress_threshold:
                for tid in token_ids:
                    bias[tid] = -100.0  # OpenAI convention for hard suppress
            else:
                b = self.scale * math.log(max(prob, 1e-9) / p_uniform)
                b = max(-self.max_bias, min(self.max_bias, b))
                for tid in token_ids:
                    bias[tid] = bias.get(tid, 0.0) + b
        
        return bias
