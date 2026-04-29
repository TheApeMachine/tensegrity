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
import threading
import numpy as np
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def _validate_hypothesis_token_scores_weights(
    hypothesis_token_scores: Optional[Dict[str, Dict[int, float]]],
    *,
    context: str,
) -> None:
    """``hypothesis_token_scores`` weights must lie in **[0.0, 1.0]** (see graft docstrings)."""
    if not hypothesis_token_scores:
        return
    bad = []
    for hyp_id, m in hypothesis_token_scores.items():
        for tid, w in m.items():
            wf = float(w)
            if not (0.0 <= wf <= 1.0):
                bad.append(f"{hyp_id}[{tid}]={wf!r}")
    if bad:
        raise ValueError(
            f"{context}: each hypothesis_token_scores value must be in [0.0, 1.0]; "
            f"misconfigured entries (showing up to 12): {bad[:12]}"
        )


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
                 hypothesis_token_scores: Optional[Dict[str, Dict[int, float]]] = None,
                 scale: float = 2.5,
                 suppress_threshold: float = 0.01,
                 entropy_gate: float = 0.85,
                 min_confidence: float = 0.3,
                 max_bias: float = 8.0,
                 async_beliefs: bool = False,
                 belief_poll_s: float = 0.005):
        """
        Args:
            hypothesis_tokens: {hyp_id: set of token_ids} from VocabularyGrounding
            hypothesis_token_scores: optional per-token **weights** in **[0.0, 1.0]**
                from ``VocabularyGrounding.from_semantic_projection`` (stored on
                ``VocabularyGrounding.hypothesis_token_scores``).
                **0.0** applies no incremental bias mass to that token; **1.0** applies the
                full clamped hypothesis bias ``b`` before per-token stacking. Values outside
                ``[0.0, 1.0]`` raise ``ValueError`` at processor construction time.
            belief_fn: Callable that returns current posteriors {hyp_id: probability}
                      Sync mode: called each decode step. Async mode: polled in a worker thread.
            vocab_size: LLM vocabulary size
            scale: γ — guidance strength. 0=off, 2.5=moderate, 5.0=strong
            suppress_threshold: P below this → hard -inf suppress
            entropy_gate: Only emit bias when normalized entropy < this.
                         1.0 = always emit, 0.0 = never emit, 0.85 = emit when fairly certain
            min_confidence: Minimum max-posterior probability to emit any bias
            max_bias: Clamp bias magnitude to prevent numerical issues
            async_beliefs: If True, belief_fn runs in a daemon thread; __call__ is O(1) bias add
            belief_poll_s: Sleep between async polls (seconds)
        """
        self.hypothesis_tokens = hypothesis_tokens
        self.hypothesis_token_scores = hypothesis_token_scores or {}
        self.belief_fn = belief_fn
        self.vocab_size = vocab_size
        self.scale = scale
        self.suppress_threshold = suppress_threshold
        self.entropy_gate = entropy_gate
        self.min_confidence = min_confidence
        self.max_bias = max_bias
        self.async_beliefs = async_beliefs
        self.belief_poll_s = belief_poll_s

        _validate_hypothesis_token_scores_weights(
            self.hypothesis_token_scores,
            context="TensegrityLogitsProcessor",
        )
        
        # State tracking
        self.state = GraftState()
        self._step_count = 0
        
        self._bias_lock = threading.RLock()
        self._latest_bias_np: Optional[np.ndarray] = None
        self._stop_worker = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if self.async_beliefs:
            self._worker = threading.Thread(target=self._async_belief_worker, daemon=True)
            self._worker.start()
    
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
    
    def close(self):
        """Stop the background belief worker (async mode)."""
        self._stop_worker.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                logger.warning(
                    "TensegrityLogitsProcessor.close: belief worker still alive after 2.0s timeout "
                    "(belief_fn may be blocking)"
                )
            self._worker = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _async_belief_worker(self):
        while not self._stop_worker.is_set():
            boxed: List[Any] = []

            def run():
                try:
                    boxed.append(self.belief_fn())
                except Exception as e:
                    boxed.append(e)

            tr = threading.Thread(target=run, daemon=True)
            tr.start()
            tr.join(timeout=0.35)
            bias_np = None
            if tr.is_alive():
                logger.warning(
                    "Async belief_fn still running after 0.35s — skipping bias update this cycle"
                )
            elif boxed and isinstance(boxed[0], Exception):
                logger.debug("Async belief worker error: %s", boxed[0])
            elif boxed:
                bias_np = self._compute_bias_numpy(boxed[0])
            else:
                bias_np = None
            with self._bias_lock:
                self._latest_bias_np = bias_np
            self._stop_worker.wait(self.belief_poll_s)
    
    def _compute_bias_numpy(self, posteriors: Dict[str, float]) -> Optional[np.ndarray]:
        """Build vocab-sized bias vector on CPU, or None if gated off."""
        with self._bias_lock:
            if not posteriors:
                self.state.bias_emitted = False
                return None
            if not self._should_emit(posteriors):
                self.state.bias_emitted = False
                return None
            
            N = len(posteriors)
            p_uniform = 1.0 / N
            bias = np.zeros(self.vocab_size, dtype=np.float64)
            boosted = 0
            suppressed = 0
            max_mag = 0.0
            
            for hyp_id, prob in posteriors.items():
                token_ids = self.hypothesis_tokens.get(hyp_id, set())
                if not token_ids:
                    continue
                token_scores = self.hypothesis_token_scores.get(hyp_id, {})
                
                if prob <= self.suppress_threshold:
                    # Dynamic temperature scaling instead of hard -inf.
                    # The review correctly identified that hard suppression
                    # to -inf collides with the LLM's syntactic expectations,
                    # causing broken grammar when suppressed tokens are
                    # structurally necessary (pronouns, conjunctions, etc.).
                    #
                    # Instead: apply a strong but finite negative bias that
                    # makes the token very unlikely but not impossible. The
                    # LLM can still use it if syntactic context demands it.
                    suppress_bias = -self.max_bias  # e.g., -8.0 instead of -inf
                    for tid in token_ids:
                        if 0 <= tid < self.vocab_size:
                            bias[tid] = suppress_bias
                            suppressed += 1
                else:
                    b = self.scale * math.log(max(float(prob), 1e-12) / p_uniform)
                    b = max(-self.max_bias, min(self.max_bias, b))
                    for tid in token_ids:
                        if 0 <= tid < self.vocab_size:
                            if not np.isneginf(bias[tid]):
                                weighted_b = b * float(token_scores.get(tid, 1.0))
                                weighted_b = max(-self.max_bias, min(self.max_bias, weighted_b))
                                bias[tid] += weighted_b
                                if weighted_b > 0:
                                    boosted += 1
                                max_mag = max(max_mag, abs(weighted_b))
            
            max_prob = max(posteriors.values())
            confidence_scale = (
                (max_prob - p_uniform) / (1.0 - p_uniform) if max_prob > p_uniform else 0.0
            )
            finite = np.isfinite(bias)
            bias[finite] *= confidence_scale
            
            self.state.bias_emitted = True
            self.state.max_bias_magnitude = max_mag * confidence_scale
            self.state.boosted_tokens = boosted
            self.state.suppressed_tokens = suppressed
            return bias
    
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
        
        if self.async_beliefs:
            with self._bias_lock:
                bias_np = None if self._latest_bias_np is None else self._latest_bias_np.copy()
        else:
            posteriors = self.belief_fn()
            bias_np = self._compute_bias_numpy(posteriors)
        
        if bias_np is None:
            self.state.bias_emitted = False
            return scores
        
        _ensure_torch()
        assert scores.shape[0] == 1, (
            f"TensegrityLogitsProcessor expects batch size 1, got {scores.shape[0]}"
        )
        score_vocab = int(scores.shape[-1])
        if bias_np.shape[0] < score_vocab:
            bias_np = np.pad(bias_np, (0, score_vocab - bias_np.shape[0]), constant_values=0.0)
        elif bias_np.shape[0] > score_vocab:
            bias_np = bias_np[:score_vocab]
        bias = torch.tensor(bias_np, device=scores.device, dtype=scores.dtype)
        return scores + bias.unsqueeze(0)


class StaticLogitBiasBuilder:
    """
    For remote API calls where per-step callbacks aren't available.
    
    Builds a static logit_bias dict from the current belief state.
    Less powerful than the LogitsProcessor (no per-step updates),
    but works with any OpenAI-compatible API.

    ``hypothesis_token_scores`` matches ``TensegrityLogitsProcessor``: optional
    ``{hyp_id: {token_id: weight}}`` with each **weight** in **[0.0, 1.0]** —
    cosine-style scores must be scaled before injection (see
    ``VocabularyGrounding.from_semantic_projection``, which emits [0.0, 1.0] weights).
    """
    
    def __init__(self, hypothesis_tokens: Dict[str, Set[int]],
                 hypothesis_token_scores: Optional[Dict[str, Dict[int, float]]] = None,
                 scale: float = 2.5,
                 suppress_threshold: float = 0.01,
                 max_bias: float = 5.0):
        self.hypothesis_tokens = hypothesis_tokens
        self.hypothesis_token_scores = hypothesis_token_scores or {}
        self.scale = scale
        self.suppress_threshold = suppress_threshold
        self.max_bias = max_bias
        _validate_hypothesis_token_scores_weights(
            self.hypothesis_token_scores,
            context="StaticLogitBiasBuilder",
        )

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
            token_scores = self.hypothesis_token_scores.get(hyp_id, {})
            
            if prob <= self.suppress_threshold:
                for tid in token_ids:
                    bias[tid] = -self.max_bias  # Finite suppress, not -100
            else:
                b = self.scale * math.log(max(prob, 1e-9) / p_uniform)
                b = max(-self.max_bias, min(self.max_bias, b))
                for tid in token_ids:
                    weighted_b = b * float(token_scores.get(tid, 1.0))
                    weighted_b = max(-self.max_bias, min(self.max_bias, weighted_b))
                    bias[tid] = bias.get(tid, 0.0) + weighted_b
        
        return bias
