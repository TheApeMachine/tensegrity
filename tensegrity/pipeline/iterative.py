"""
Iterative cognitive scorer — LLM-free multi-pass settling over choices.

Single-shot ScoringBridge encodes prompt once, settles NGC once per choice,
fuses sentence + FHRR + NGC scores in one shot. The graft results show this
behaves like an undifferentiated bias field.

This iterative scorer instead runs an active-inference loop:
  1. Encode prompt context, settle NGC, learn (ground the field).
  2. Initialize uniform belief over choices.
  3. For each iteration up to a budget:
     a. Score each choice via NGC free-energy under the *current* field state.
     b. Update beliefs by accumulating evidence (Bayesian-style log-odds).
     c. Take the leading choice's encoding, learn a small Hebbian step under
        it (modulation = belief mass), shaping the field toward that
        interpretation.
     d. Optionally retrieve from Hopfield with the leading encoding to inject
        memory pressure.
     e. Check convergence: top-1 belief mass > τ, or marginal change < ε.
  4. Commit argmax.

The LLM is absent. The cognitive layer alone resolves the choice.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IterationTrace:
    iteration: int
    energies: List[float]
    sentence_sims: List[float]
    fhrr_sims: List[float]
    log_belief: List[float]
    belief: List[float]
    top_idx: int
    top_p: float


@dataclass
class IterativeResult:
    scores: List[float]                # final fused scores per choice
    belief: List[float]                # final belief vector per choice
    committed_idx: int
    iterations_used: int
    converged: bool
    trace: List[IterationTrace] = field(default_factory=list)


class IterativeCognitiveScorer:
    """
    Multi-pass cognitive scorer over a UnifiedField.

    No LLM in the loop. Operates on prompt+choices via:
      - sbert sentence similarity (one-shot, doesn't change across iterations)
      - FHRR similarity (one-shot)
      - NGC free energy (recomputed each iteration as the field is shaped)
      - Hopfield retrieval (cumulative memory pressure across iterations)
    """

    def __init__(
        self,
        field=None,
        *,
        obs_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        fhrr_dim: int = 2048,
        ngc_settle_steps: int = 30,
        ngc_learning_rate: float = 0.01,
        hopfield_beta: float = 0.05,
        # iteration controls
        max_iterations: int = 6,
        convergence_top_p: float = 0.75,
        convergence_delta: float = 1e-3,
        # context settling
        context_settle_steps: int = 40,
        choice_settle_steps: int = 25,
        context_learning_epochs: int = 3,
        # fusion weights (z-scored)
        w_sbert: float = 0.5,
        w_fhrr: float = 0.3,
        w_ngc: float = 1.0,
        w_falsify: float = 0.7,
        # belief update step
        belief_step: float = 0.6,
        # Hebbian shaping is now under the prompt context (not the leading choice),
        # so iteration deepens the prompt model rather than reinforcing the leader.
        shaping_lr_scale: float = 0.5,
        # Hopfield: store leading encoding each iteration; query each iteration
        use_hopfield: bool = True,
        hopfield_steps: int = 2,
        # Episodic memory persists across items in a session. At the start of
        # each item we retrieve past episodes whose context matches the current
        # prompt and use their stored chosen-answer FHRR vectors to bias the
        # current choices — the cross-item learning channel.
        use_episodic: bool = True,
        episodic_context_dim: int = 64,
        episodic_capacity: int = 4096,
        episodic_top_k: int = 8,
        # Default off: the simple "past-answer FHRR similarity" signal is too
        # noisy to help. The wiring (encode/retrieve) stays so smarter signals
        # can be plugged in here without re-plumbing.
        w_episodic: float = 0.0,
    ):
        from tensegrity.engine.unified_field import UnifiedField
        from tensegrity.memory.episodic import EpisodicMemory
        self.field = field or UnifiedField(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims or [128, 32],
            fhrr_dim=fhrr_dim,
            hopfield_beta=hopfield_beta,
            ngc_settle_steps=ngc_settle_steps,
            ngc_learning_rate=ngc_learning_rate,
        )
        self.max_iterations = max_iterations
        self.convergence_top_p = convergence_top_p
        self.convergence_delta = convergence_delta
        self.context_settle_steps = context_settle_steps
        self.choice_settle_steps = choice_settle_steps
        self.context_learning_epochs = context_learning_epochs
        self.w_sbert = w_sbert
        self.w_fhrr = w_fhrr
        self.w_ngc = w_ngc
        self.w_falsify = w_falsify
        self.belief_step = belief_step
        self.shaping_lr_scale = shaping_lr_scale
        self.use_hopfield = use_hopfield
        self.hopfield_steps = hopfield_steps
        self.use_episodic = use_episodic
        self.episodic_top_k = episodic_top_k
        self.w_episodic = w_episodic
        # Dirichlet-style per-channel reliability. Each channel accumulates a
        # pseudocount that grows when the channel's top-ranked choice matches
        # the committed belief on an item. Fusion weights = normalized counts.
        # Uniform prior of 1.0 means we start with equal trust; the system
        # discovers which channels are reliable for the current task on its own.
        self._channels = ["sbert", "fhrr", "ngc", "falsify", "hop", "episodic"]
        self._channel_counts: Dict[str, float] = {c: 1.0 for c in self._channels}
        self.episodic = EpisodicMemory(
            context_dim=episodic_context_dim,
            capacity=episodic_capacity,
            drift_rate=0.95,
            encoding_strength=0.3,
        ) if use_episodic else None

    # ---------- text helpers ----------

    def _tokenize(self, text: str, max_tokens: int = 48) -> List[str]:
        return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?", text.lower())[-max_tokens:]

    def _encode(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.ones(self.field.fhrr_dim, dtype=np.complex64)
        return self.field.encoder.encode_sequence(tokens)

    # ---------- one-shot signals (computed once per item) ----------

    def _sbert_similarities(self, prompt: str, choices: List[str]) -> List[float]:
        features = self.field.encoder.features
        if hasattr(features, "_ensure_sbert") and getattr(features, "_sbert", None) is None:
            features._ensure_sbert()
        sbert = getattr(features, "_sbert", None)
        if sbert is not None and sbert != "FALLBACK":
            embs = sbert.encode([prompt] + choices, show_progress_bar=False)
            pe = embs[0]
            pn = float(np.linalg.norm(pe))
            out = []
            for i in range(len(choices)):
                ce = embs[i + 1]
                cn = float(np.linalg.norm(ce))
                out.append(float(np.dot(pe, ce) / (pn * cn)) if pn > 1e-8 and cn > 1e-8 else 0.0)
            return out
        # fallback to FHRR similarity
        pf = self._encode(self._tokenize(prompt, 64))
        return [
            self.field.encoder.similarity(pf, self._encode(self._tokenize(c, 32)))
            for c in choices
        ]

    def _fhrr_similarities(self, prompt: str, choices: List[str]) -> List[float]:
        pf = self._encode(self._tokenize(prompt, 64))
        return [
            self.field.encoder.similarity(pf, self._encode(self._tokenize(c, 32)))
            for c in choices
        ]

    # ---------- iterative loop ----------

    def score(self, prompt: str, choices: List[str]) -> IterativeResult:
        n = len(choices)
        if n == 0:
            return IterativeResult(scores=[], belief=[], committed_idx=-1,
                                   iterations_used=0, converged=False)

        # 1. One-shot signals
        sbert_sims = np.asarray(self._sbert_similarities(prompt, choices), dtype=np.float64)
        fhrr_sims = np.asarray(self._fhrr_similarities(prompt, choices), dtype=np.float64)

        # 2. Encode + settle prompt context, learn it
        prompt_tokens = self._tokenize(prompt, max_tokens=64)
        for _ in range(max(1, self.context_learning_epochs)):
            ctx_obs = self.field._fhrr_to_obs(self._encode(prompt_tokens))
            self.field.ngc.settle(ctx_obs, steps=self.context_settle_steps)
            self.field.ngc.learn(modulation=1.0)
        base_state = self.field.ngc.save_state()

        # Pre-tokenize choice contexts (prompt+choice for joint settling)
        choice_token_lists = [self._tokenize(prompt + " " + c, 64) for c in choices]
        choice_obs = [self.field._fhrr_to_obs(self._encode(t)) for t in choice_token_lists]
        # Choice-only obs (for falsification: settle under choice alone, then predict prompt)
        choice_only_obs = [
            self.field._fhrr_to_obs(self._encode(self._tokenize(c, 32))) for c in choices
        ]
        choice_fhrr = [self._encode(self._tokenize(c, 32)) for c in choices]
        # Cache prompt observation vector for falsification target
        prompt_obs_vec = self.field._fhrr_to_obs(self._encode(prompt_tokens))

        # Episodic retrieval: project current prompt into context space and ask
        # the episodic store for similar past episodes. Each retrieved episode
        # carries the FHRR of the answer that won there. We bias current
        # choices by their similarity to those past winners, weighted by the
        # context match. This is the cross-item memory channel.
        episodic_bias = np.zeros(n, dtype=np.float64)
        if self.use_episodic and self.episodic is not None and len(self.episodic.episodes) > 0:
            uniform_belief = np.full(n, 1.0 / n, dtype=np.float64)
            try:
                query_ctx = self.episodic._compute_item_representation(
                    prompt_obs_vec, uniform_belief
                )
                retrieved = self.episodic.retrieve_by_context(
                    query_context=query_ctx, k=self.episodic_top_k
                )
            except Exception as e:
                logger.debug("episodic retrieval skipped: %s", e)
                retrieved = []
            if retrieved:
                # Real-valued unit-norm choice vectors (cached for reuse)
                ch_real = []
                for f in choice_fhrr:
                    v = np.real(f).astype(np.float64)
                    nrm = np.linalg.norm(v)
                    ch_real.append(v / nrm if nrm > 1e-10 else v)
                # Only trust episodes whose prompt context strongly matches.
                # Below this threshold, "similar past answer" is noise, not signal.
                CTX_SIM_THRESHOLD = 0.5
                for ep in retrieved:
                    ans_vec = ep.metadata.get("chosen_fhrr_real") if ep.metadata else None
                    if ans_vec is None:
                        continue
                    ctx_sim = float(np.dot(query_ctx, ep.context_vector))
                    if ctx_sim < CTX_SIM_THRESHOLD:
                        continue
                    # Also discount by past surprise: episodes the agent struggled
                    # with (low committed confidence) carry less authority.
                    confidence = max(0.0, 1.0 - float(ep.surprise))
                    weight = ctx_sim * confidence
                    if weight <= 0:
                        continue
                    for i in range(n):
                        episodic_bias[i] += weight * float(np.dot(ch_real[i], ans_vec))

        # 3. Initialize belief uniformly in log space
        log_belief = np.zeros(n, dtype=np.float64)

        trace: List[IterationTrace] = []
        prev_belief = np.ones(n) / n
        converged = False
        iterations_used = 0
        last_channel_scores: Dict[str, np.ndarray] = {}

        for it in range(self.max_iterations):
            iterations_used = it + 1

            # 3a. Score each choice under current field state.
            # Two NGC signals:
            #   energies: free energy of settling on (prompt+choice) jointly.
            #   falsify:  -prediction_error of (prompt | settled-on-choice-alone).
            #             This asks "does this choice's state predict the prompt?"
            #             — a real falsification operation, not a fit score.
            energies = np.zeros(n, dtype=np.float64)
            falsify = np.zeros(n, dtype=np.float64)
            for i in range(n):
                self.field.ngc.restore_state(base_state)
                r = self.field.ngc.settle(choice_obs[i], steps=self.choice_settle_steps)
                energies[i] = float(r["final_energy"])

                # Falsification: settle under choice-only, then ask the field
                # to predict the prompt observation. Higher prediction error =
                # this choice does a worse job of explaining the prompt.
                self.field.ngc.restore_state(base_state)
                self.field.ngc.settle(choice_only_obs[i], steps=self.choice_settle_steps)
                pe = self.field.ngc.prediction_error(prompt_obs_vec)
                falsify[i] = -float(pe)
            ngc_score = -energies

            # Hopfield bonus: similarity of choice FHRR to retrieved memory
            hop_bonus = np.zeros(n, dtype=np.float64)
            if self.use_hopfield and self.field.memory.n_patterns > 0:
                for i in range(n):
                    q = np.real(choice_fhrr[i]).astype(np.float64)
                    qn = np.linalg.norm(q)
                    if qn < 1e-8:
                        continue
                    q = q / qn
                    retrieved, _e = self.field.memory.retrieve(q, steps=self.hopfield_steps)
                    rn = np.linalg.norm(retrieved)
                    if rn > 1e-8:
                        hop_bonus[i] = float(np.dot(q, retrieved / rn))

            # 3b. Fuse z-normalized
            def znorm(a: np.ndarray) -> np.ndarray:
                s = a.std()
                return (a - a.mean()) / s if s > 1e-10 else np.zeros_like(a)

            # Normalized channel weights from accumulated reliability counts.
            total = sum(self._channel_counts.values())
            w = {c: self._channel_counts[c] / total for c in self._channels}

            channel_scores = {
                "sbert": znorm(sbert_sims),
                "fhrr": znorm(fhrr_sims),
                "ngc": znorm(ngc_score),
                "falsify": znorm(falsify),
                "hop": znorm(hop_bonus) if self.use_hopfield else np.zeros(n),
                "episodic": znorm(episodic_bias) if self.use_episodic else np.zeros(n),
            }
            fused = sum(w[c] * channel_scores[c] for c in self._channels)
            last_channel_scores = channel_scores

            # 3c. Accumulate evidence into log-belief
            log_belief = log_belief + self.belief_step * fused
            shifted = log_belief - log_belief.max()
            belief = np.exp(shifted)
            belief = belief / belief.sum() if belief.sum() > 0 else np.ones(n) / n

            top_idx = int(np.argmax(belief))
            top_p = float(belief[top_idx])

            trace.append(IterationTrace(
                iteration=it,
                energies=energies.tolist(),
                sentence_sims=sbert_sims.tolist(),
                fhrr_sims=fhrr_sims.tolist(),
                log_belief=log_belief.tolist(),
                belief=belief.tolist(),
                top_idx=top_idx,
                top_p=top_p,
            ))

            # 3d. Hebbian shaping under the PROMPT (not the leading choice).
            # This deepens the field's model of the question over iterations
            # without injecting a positive-feedback loop on the leader.
            self.field.ngc.restore_state(base_state)
            self.field.ngc.settle(prompt_obs_vec, steps=self.context_settle_steps)
            self.field.ngc.learn(modulation=self.shaping_lr_scale)

            # 3e. Hopfield: store the *prompt* encoding so cross-iteration memory
            # accumulates evidence about the question, not the current guess.
            if self.use_hopfield:
                self.field.memory.store(self._encode(prompt_tokens))

            # Re-base on the prompt-grounded state for next iteration's scoring
            base_state = self.field.ngc.save_state()

            # 3f. Convergence checks
            db = float(np.max(np.abs(belief - prev_belief)))
            prev_belief = belief
            if top_p >= self.convergence_top_p or db < self.convergence_delta:
                converged = True
                break

        committed_idx = int(np.argmax(prev_belief))

        # Reliability update via *cross-channel agreement* (not agreement with
        # the committed belief — that would be self-fulfilling). Each channel
        # earns one pseudocount per OTHER active channel that picked the same
        # top choice. The consensus structure is the anchor; no single
        # channel is privileged. Channels tracking signal grow together;
        # noisy outliers don't.
        if last_channel_scores and n > 1:
            active = []
            for c in self._channels:
                cs = last_channel_scores.get(c)
                if cs is None:
                    continue
                if not np.any(np.abs(cs) > 1e-12):
                    continue
                active.append((c, int(np.argmax(cs))))
            for i, (c_i, top_i) in enumerate(active):
                agreements = sum(
                    1 for j, (_, top_j) in enumerate(active) if j != i and top_j == top_i
                )
                if agreements > 0:
                    self._channel_counts[c_i] += float(agreements) / max(len(active) - 1, 1)

        # Episodic encoding: store the prompt context together with the FHRR
        # of the chosen answer, so future items can retrieve "what worked
        # last time on a similar prompt."
        if self.use_episodic and self.episodic is not None:
            top_p_final = float(prev_belief[committed_idx]) if n > 0 else 0.0
            chosen_real = np.real(choice_fhrr[committed_idx]).astype(np.float64)
            chosen_norm = np.linalg.norm(chosen_real)
            if chosen_norm > 1e-10:
                chosen_real = chosen_real / chosen_norm
            try:
                self.episodic.encode(
                    observation=prompt_obs_vec,
                    morton_code=np.zeros(1, dtype=np.int64),
                    belief_state=np.asarray(prev_belief, dtype=np.float64),
                    action=committed_idx,
                    surprise=float(1.0 - top_p_final),
                    free_energy=float(np.mean(energies) if n > 0 else 0.0),
                    metadata={"chosen_fhrr_real": chosen_real},
                )
            except Exception as e:
                logger.debug("episodic encode skipped: %s", e)

        return IterativeResult(
            scores=log_belief.tolist(),
            belief=prev_belief.tolist(),
            committed_idx=committed_idx,
            iterations_used=iterations_used,
            converged=converged,
            trace=trace,
        )

    def reset(self):
        """Per-item reset. Clears NGC working state but PRESERVES Hopfield
        patterns and episodic memory — those carry across items in a session
        and provide cross-item learning."""
        self.field.ngc.reinitialize(12345)
        self.field.energy_history.clear()
        self.field._step_count = 0

    def reset_session(self):
        """Full reset. Use at task / session boundaries to clear all memory
        and per-channel reliability priors (which are task-specific)."""
        self.reset()
        self.field.memory.patterns.clear()
        self.field.memory._matrix = None
        self.field.memory._dirty = True
        if self.episodic is not None:
            self.episodic.clear()
        for c in self._channels:
            self._channel_counts[c] = 1.0
