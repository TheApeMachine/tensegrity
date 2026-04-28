"""
Semantic scoring bridge + NGC logit bias injection (part of the unified engine).
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Set, Tuple, Any
import math
import logging
import re
import threading

logger = logging.getLogger(__name__)

torch = None
def _ensure_torch():
    global torch
    if torch is None:
        import importlib
        torch = importlib.import_module('torch')


class NGCLogitsProcessor:
    """NGC prediction errors → per-step logit biases during LLM decoding."""
    
    supports_continuous_batching = False
    
    def __init__(self, field, tokenizer, vocab_projections=None,
                 scale=1.0, energy_gate=0.1, max_settle_steps=30, max_bias=5.0,
                 async_cognitive: bool = True):
        _ensure_torch()
        self.field = field
        self.tokenizer = tokenizer
        self.scale = scale
        self.energy_gate = energy_gate
        self.max_settle_steps = max_settle_steps
        self.max_bias = max_bias
        self.async_cognitive = async_cognitive
        self.vocab_size = tokenizer.vocab_size
        self.projections = vocab_projections or self._build_projections()
        self._step_count = 0
        self._emissions = 0
        self._total_settle_steps = 0
        
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._wake = threading.Event()
        self._pending_ids: Optional[List[int]] = None
        self._latest_bias_np: Optional[np.ndarray] = None
        self._worker: Optional[threading.Thread] = None
        if self.async_cognitive:
            self._worker = threading.Thread(target=self._cognitive_loop, daemon=True)
            self._worker.start()
    
    def close(self):
        self._halt.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                logger.warning(
                    "NGCLogitsProcessor worker did not stop within 2.0s (belief_fn may block)"
                )
            self._worker = None
    
    def _build_projections(self):
        projections = []
        rng = np.random.RandomState(7777)
        for ell, size in enumerate(self.field.ngc.layer_sizes):
            P = rng.randn(self.vocab_size, size).astype(np.float64)
            P *= (2.0 ** ell) / np.sqrt(size)
            projections.append(P)
        return projections
    
    def _compute_bias_from_ids(self, ids: List[int]) -> Optional[np.ndarray]:
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        tokens = text.lower().split()
        if not tokens:
            return None
        obs = self.field._fhrr_to_obs(self.field.encoder.encode_sequence(tokens))
        settle = self.field.ngc.settle(obs)
        self._total_settle_steps += int(settle.get("settle_steps", self.max_settle_steps))
        et = settle["energy_trace"]
        if len(et) < 2 or abs(et[-1] - et[-2]) >= self.energy_gate:
            return None
        bias = np.zeros(self.vocab_size, dtype=np.float64)
        for ell in range(self.field.ngc.n_layers):
            err = self.field.ngc.layers[ell].error
            if np.linalg.norm(err) > 1e-10:
                bias += self.projections[ell] @ err
        bias /= max(self.field.ngc.n_layers, 1)
        confidence = 1.0 / (1.0 + settle["final_energy"])
        bias *= self.scale * confidence
        np.clip(bias, -self.max_bias, self.max_bias, out=bias)
        self._emissions += 1
        return bias
    
    def _cognitive_loop(self):
        while not self._halt.is_set():
            if not self._wake.wait(timeout=0.05):
                continue
            self._wake.clear()
            if self._halt.is_set():
                break
            with self._lock:
                ids = self._pending_ids
            if ids is None:
                continue
            try:
                bias_np = self._compute_bias_from_ids(ids)
            except Exception as e:
                logger.debug("NGC cognitive worker: %s", e)
                bias_np = None
            with self._lock:
                self._latest_bias_np = bias_np
    
    def __call__(self, input_ids, scores):
        self._step_count += 1
        _ensure_torch()
        if not isinstance(input_ids, torch.Tensor):
            arr = np.asarray(input_ids)
            if arr.ndim == 1:
                flat = arr.tolist()
            elif arr.ndim == 2:
                flat = arr[-1].tolist()
            else:
                raise ValueError(f"input_ids must be 1D or 2D, got shape {arr.shape}")
        else:
            if input_ids.dim() == 1:
                flat = input_ids.detach().cpu().tolist()
            elif input_ids.dim() == 2:
                flat = input_ids[-1].detach().cpu().tolist()
            else:
                raise ValueError(f"input_ids must be 1D or 2D, got shape {tuple(input_ids.shape)}")
        ids = flat[-16:]
        if self.async_cognitive:
            with self._lock:
                self._pending_ids = list(ids)
            self._wake.set()
            with self._lock:
                bias_np = None if self._latest_bias_np is None else self._latest_bias_np.copy()
            if bias_np is None:
                return scores
            _ensure_torch()
            assert scores.shape[0] == 1, (
                f"NGCLogitsProcessor expects batch size 1, got {scores.shape[0]}"
            )
            return scores + torch.tensor(bias_np, device=scores.device, dtype=scores.dtype).unsqueeze(0)
        
        try:
            bias_np = self._compute_bias_from_ids(ids)
        except Exception as e:
            logger.debug("NGCLogitsProcessor: %s", e)
            return scores
        if bias_np is None:
            return scores
        assert scores.shape[0] == 1, (
            f"NGCLogitsProcessor expects batch size 1, got {scores.shape[0]}"
        )
        return scores + torch.tensor(bias_np, device=scores.device, dtype=scores.dtype).unsqueeze(0)
    
    @property
    def statistics(self):
        return {
            "decode_steps": self._step_count, "emissions": self._emissions,
            "emission_rate": self._emissions / max(self._step_count, 1),
            "ngc_energy": self.field.ngc.total_energy,
        }


class ScoringBridge:
    """
    Semantic scoring bridge for benchmark evaluation.
    
    Combines sentence-level sbert similarity (primary signal) with
    token-level semantic FHRR and NGC energy (complementary signals).
    """
    
    def __init__(self, field=None, obs_dim=256, hidden_dims=None,
                 fhrr_dim=2048, ngc_settle_steps=30, ngc_learning_rate=0.01,
                 hopfield_beta=0.05, confidence_threshold=0.15,
                 context_settle_steps=40, choice_settle_steps=25,
                 context_learning_epochs=3):
        from tensegrity.engine.unified_field import UnifiedField
        self.field = field or UnifiedField(
            obs_dim=obs_dim, hidden_dims=hidden_dims or [128, 32],
            fhrr_dim=fhrr_dim, hopfield_beta=hopfield_beta,
            ngc_settle_steps=ngc_settle_steps, ngc_learning_rate=ngc_learning_rate,
        )
        self.confidence_threshold = confidence_threshold
        self.context_settle_steps = context_settle_steps
        self.choice_settle_steps = choice_settle_steps
        self.context_learning_epochs = context_learning_epochs
        self._total_scored = 0
        self._total_gated = 0
    
    def _tokenize_smart(self, text: str, max_tokens: int = 48) -> List[str]:
        return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?", text.lower())[-max_tokens:]
    
    def _encode_and_settle(self, tokens, settle_steps, learn=False):
        if not tokens:
            return {"energy": 0.0, "obs_vec": np.zeros(self.field.obs_dim),
                    "abstract_state": np.zeros(self.field.ngc.layer_sizes[-1]),
                    "fhrr_vec": np.ones(self.field.fhrr_dim, dtype=np.complex64),
                    "settle": {"final_energy": 0.0, "energy_trace": [0.0]}}
        fhrr_vec = self.field.encoder.encode_sequence(tokens)
        obs_vec = self.field._fhrr_to_obs(fhrr_vec)
        settle_result = self.field.ngc.settle(obs_vec, steps=settle_steps)
        if learn:
            self.field.ngc.learn(modulation=1.0)
        return {"energy": settle_result["final_energy"], "obs_vec": obs_vec,
                "abstract_state": self.field.ngc.get_abstract_state(level=-1),
                "fhrr_vec": fhrr_vec, "settle": settle_result}
    
    def score_choices(self, prompt: str, choices: List[str]) -> Tuple[List[float], float]:
        """Score choices via sentence similarity + semantic FHRR + NGC energy."""
        self._total_scored += 1
        n = len(choices)
        
        # 1. Sentence-level similarity (primary)
        sentence_sims = self._sentence_similarities(prompt, choices)
        
        # 2. Token-level FHRR similarity
        pt = self._tokenize_smart(prompt, max_tokens=64)
        pf = self.field.encoder.encode_sequence(pt) if pt else np.ones(self.field.fhrr_dim, dtype=np.complex64)
        fhrr_sims = []
        for choice in choices:
            ct = self._tokenize_smart(choice, max_tokens=32)
            enc_c = (
                self.field.encoder.encode_sequence(ct)
                if ct
                else np.ones(self.field.fhrr_dim, dtype=np.complex64)
            )
            fhrr_sims.append(self.field.encoder.similarity(pf, enc_c))
        
        # 3. NGC energy
        self._encode_and_settle(pt, settle_steps=self.context_settle_steps, learn=True)
        base_state = self.field.ngc.save_state()
        ngc_energies = []
        for choice in choices:
            self.field.ngc.restore_state(base_state)
            r = self._encode_and_settle(
                self._tokenize_smart(prompt + " " + choice, 64),
                self.choice_settle_steps,
                False,
            )
            ngc_energies.append(-r["energy"])
        
        # 4. Combine
        def znorm(a):
            s = a.std()
            return (a - a.mean()) / s if s > 1e-10 else np.zeros_like(a)
        
        scores_arr = znorm(np.array(sentence_sims)) * 1.0 + znorm(np.array(fhrr_sims)) * 0.3 + znorm(np.array(ngc_energies)) * 0.2
        
        # 5. Gate
        sa = np.array(sentence_sims)
        spread = float(sa.max() - sa.min())
        mean = float(np.abs(sa).mean())
        cv = spread / mean if mean > 1e-8 else 0.0
        
        shifted = scores_arr - scores_arr.max()
        probs = np.exp(shifted)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n
        entropy = float(-np.sum(probs * np.log(probs + 1e-16)) / np.log(max(n, 2)))
        
        thresh = self.confidence_threshold * (3.0 if n <= 2 else 2.0 if n <= 3 else 1.5)
        if cv < thresh or entropy > 0.97:
            self._total_gated += 1
            return [0.0] * n, 1.0
        return scores_arr.tolist(), entropy
    
    def _sentence_similarities(self, prompt, choices):
        features = self.field.encoder.features
        if hasattr(features, '_sbert') and features._sbert is not None and features._sbert != "FALLBACK":
            features._ensure_sbert()
            embs = features._sbert.encode([prompt] + choices, show_progress_bar=False)
            pe, pn = embs[0], np.linalg.norm(embs[0])
            return [float(np.dot(pe, embs[i+1]) / (pn * np.linalg.norm(embs[i+1])))
                    if pn > 1e-8 and np.linalg.norm(embs[i+1]) > 1e-8 else 0.0
                    for i in range(len(choices))]
        pt = self._tokenize_smart(prompt, 64)
        pf = self.field.encoder.encode_sequence(pt) if pt else np.ones(self.field.fhrr_dim, dtype=np.complex64)
        out = []
        for c in choices:
            ct = self._tokenize_smart(c, 32)
            enc = self.field.encoder.encode_sequence(ct) if ct else np.ones(self.field.fhrr_dim, dtype=np.complex64)
            out.append(self.field.encoder.similarity(pf, enc))
        return out
    
    def reset(self):
        self.field.ngc.reinitialize(12345)
        self.field.memory.patterns.clear()
        self.field.memory._matrix = None
        self.field.memory._dirty = True
        self.field.energy_history.clear()
        self.field._step_count = 0
    
    @property
    def statistics(self):
        return {"total_scored": self._total_scored, "total_gated": self._total_gated,
                "gate_rate": self._total_gated / max(self._total_scored, 1)}
