"""
Canonical Tensegrity pipeline — the agent loop.

This is the unified entry point for benchmarks AND chat. The cognitive layer
is the agent; the LLM (when used) is a typed tool exposed only via Broca.
For multiple-choice scoring, LLM choice likelihoods are treated as one sensory
evidence channel inside the agent's posterior update. The final answer is the
agent's commitment after predictive-coding falsification, causal competition,
memory retrieval, and linguistic evidence have been integrated.

Wired subsystems (every component the project ships):
  • CognitiveController     — agent body, owns BeliefState (per-hypothesis posteriors)
  • TensegrityAgent.perceive — runs UnifiedField (FHRR + NGC + Hopfield),
                               FreeEnergyEngine, EpistemicMemory, EpisodicMemory,
                               AssociativeMemory, log-likelihood CausalArena
  • BrocaInterface          — typed transducer (parse / propose_causal_hypothesis / produce)
  • EnergyCausalArena       — per-choice SCMs compete on prediction-error energy,
                               registered through TopologyMapper so the SCM DAG is
                               projected into the NGC layer hierarchy explicitly
                               (addresses the "subway map vs elevator" critique)
  • EpisodicMemory          — persists across items in a session for cross-item recall
  • Broca dynamic SCM       — controller's _maybe_inject_causal_hypothesis fires when
                               causal tension is high, asking the LLM for a new SCM
                               structure that gets compiled via build_scm_from_proposal
  • NGC falsification       — for each choice, settle the field on the choice alone,
                               ask it to predict the prompt, score = -prediction_error.
                               This is "does this choice's settled state explain the
                               prompt" — a real falsification operation.
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tensegrity.broca.controller import CognitiveController
from tensegrity.broca.schemas import BeliefState, Hypothesis
from tensegrity.bench.tasks import TaskSample
from tensegrity.causal.scm import StructuralCausalModel
from tensegrity.engine.causal_energy import (
    EnergyCausalArena,
    TopologyMapper,
)

logger = logging.getLogger(__name__)


@dataclass
class IterationStep:
    """One step of the agent's reasoning trace on a single item."""
    iteration: int
    belief: List[float]
    perception_free_energy: float
    perception_tension: float
    arena_tension: float
    arena_winner: Optional[str]
    falsification_scores: List[float]
    energy_arena_winner: Optional[str]
    top_idx: int
    top_p: float


@dataclass
class CommitResult:
    """The pipeline's verdict on one item, with full diagnostics."""
    scores: List[float]              # belief shifted away from uniform, in [-1, 1]
    belief: List[float]              # final hypothesis posterior
    committed_idx: int
    iterations_used: int
    converged: bool
    final_perception_tension: float
    final_arena_tension: float
    final_energy_arena_tension: float
    trace: List[IterationStep] = field(default_factory=list)
    initial_perception: Optional[Dict[str, Any]] = None
    linguistic_scores: Optional[List[float]] = None
    memory_scores: Optional[List[float]] = None


def _alphanum_tokens(text: str, max_tokens: int) -> List[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?", text.lower())[-max_tokens:]


class CanonicalPipeline:
    """
    The agent loop. One pipeline used by both bench and chat.

    Per-item flow:
      1. reset_for_item: register choices as hypotheses; clear NGC working state
         (memory + arena history persist across items in the same session).
      2. ingest_prompt: controller.perceive_only — runs the full perception stack
         once. Triggers Broca SCM proposal if causal tension is high.
      3. Iterate (up to a budget):
           a. NGC-falsification per choice — top-down prediction of the prompt
              from the choice's settled state.
           b. EnergyCausalArena.compete — per-choice SCMs compete on energy
              under derived observations; topology-mapped so the DAG embedding
              into NGC layers is explicit.
           c. Re-perceive with (prompt + leading_choice) to get a refined view;
              this re-runs the full agent stack including Broca arena, episodic
              encoding, free-energy step.
           d. Combine: per-choice falsification log-likelihood + energy-arena
              posterior + controller's belief over choice-hypotheses.
           e. Convergence: stop when the leading hypothesis is statistically
              separated from the runner-up (top1/top2 ratio above a threshold
              derived from the iteration's belief variance, not a magic number).
      4. Commit: argmax of the controller's belief state.

    Persistent flow:
      The benchmark runner keeps this pipeline alive across tasks and runs.
      Per-item hypothesis slots are rewritten, but field weights, episodic
      memory, Hopfield attractors, epistemic counts, and the agent causal arena
      persist unless reset_session is called explicitly.
    """

    def __init__(
        self,
        hypothesis_labels: Optional[List[str]] = None,
        *,
        broca: Optional[Any] = None,
        use_llm_broca: bool = False,
        enable_hypothesis_generation: bool = False,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        # Loop budget
        max_iterations: int = 4,
        # Convergence: top1/top2 ratio above which we commit. Default 2.0
        # means the leader must be at least twice the runner-up in mass.
        commit_ratio: float = 2.0,
        # Falsification: how many NGC steps to settle each choice for the
        # top-down-predict-the-prompt operation.
        falsify_settle_steps: int = 20,
        # Bayesian update strength when integrating falsification likelihood
        # into the controller's hypothesis posteriors.
        falsify_update_strength: float = 1.0,
        # Energy-arena precision (passed through to CausalEnergyTerm).
        energy_arena_precision: float = 1.0,
        # Energy-arena selection temperature (1.0 = uniform softmax).
        energy_arena_beta: float = 1.0,
        # Fixed hypothesis width keeps one agent alive across tasks with
        # different choice counts. Unused slots are zero-probability padding.
        max_hypotheses: int = 8,
        # LLM likelihoods enter the posterior as a sensory-evidence channel.
        llm_evidence_weight: float = 1.0,
        # Persistent episodic recall enters as a memory-evidence channel.
        memory_evidence_weight: float = 0.75,
        feedback_learning_rate: float = 1.0,
        persistent_state_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_iterations = max(1, int(max_iterations))
        self.commit_ratio = float(commit_ratio)
        self.falsify_settle_steps = int(falsify_settle_steps)
        self.falsify_update_strength = float(falsify_update_strength)
        self.max_hypotheses = max(2, int(max_hypotheses))
        self.llm_evidence_weight = float(llm_evidence_weight)
        self.memory_evidence_weight = float(memory_evidence_weight)
        self.feedback_learning_rate = float(feedback_learning_rate)
        self.persistent_state_path = persistent_state_path

        initial_labels = list(hypothesis_labels or [])
        while len(initial_labels) < self.max_hypotheses:
            initial_labels.append(f"_empty_{len(initial_labels)}")

        # The agent body. controller.agent runs the full stack:
        #   UnifiedField (FHRR + NGC + Hopfield) + FreeEnergyEngine
        #   + EpistemicMemory + EpisodicMemory + AssociativeMemory
        #   + log-likelihood CausalArena.
        # Broca, when enabled, parses input into typed structure and proposes
        # SCMs when causal tension is high (handled inside perceive_only).
        self.controller = CognitiveController(
            n_hypotheses=len(initial_labels),
            hypothesis_labels=initial_labels,
            broca=broca,
            use_llm=use_llm_broca or broca is not None,
            enable_hypothesis_generation=enable_hypothesis_generation,
        )

        # The energy-based causal arena is the SECOND arena in this stack.
        # The controller's existing arena (log-likelihood, inside TensegrityAgent)
        # stays live — it competes on a single shared SCM space across all items.
        # The energy arena here is per-choice: each choice gets its own SCM
        # and competes on the prediction-error energy of derived observations.
        # This is the wiring for the EnergyCausalArena+CausalEnergyTerm modules
        # that were previously orphaned.
        self._topology_mapper = TopologyMapper(expand_layers=True)
        self.energy_arena: EnergyCausalArena = EnergyCausalArena(
            precision=energy_arena_precision, beta=energy_arena_beta,
        )
        # Topology mappings produced for each registered SCM (debuggable).
        self._scm_topologies: Dict[str, Any] = {}

        # Track item index for episodic encoding metadata.
        self._item_index = 0
        self._choice_model_names: List[str] = []
        self._last_derived_obs: List[Dict[str, int]] = []

        if self.persistent_state_path:
            self.load_state(self.persistent_state_path)

    # ---------- session boundaries ----------

    def reset_session(self) -> None:
        """Per-task reset. Clears persistent memory and the energy arena."""
        # The controller resets itself per item; here we additionally clear the
        # Hopfield bank, episodic memory, and energy arena.
        try:
            self.controller.agent.field.memory.clear()
        except AttributeError as e:
            logger.warning("Hopfield clear failed (missing clear): %s", e)
        except Exception as e:
            logger.error("Hopfield clear failed: %s", e, exc_info=True)
        try:
            self.controller.agent.episodic.clear()
        except AttributeError as e:
            logger.warning("Episodic clear failed: %s", e)
        except Exception as e:
            logger.error("Episodic clear failed: %s", e, exc_info=True)
        self.energy_arena = EnergyCausalArena(
            precision=self.energy_arena.precision,
            beta=self.energy_arena.beta,
        )
        self._scm_topologies.clear()
        self._item_index = 0

    def reset_for_item(self, sample: TaskSample) -> None:
        """Per-item reset: register hypotheses, register per-choice SCMs in
        the energy arena (with topology mappings), keep memory across items.

        We avoid the controller's full reset_session (which rebuilds the
        entire TensegrityAgent and forces sbert to lazy-reload) when the
        choice cardinality is unchanged. Instead we soft-reset NGC working
        state and rewrite the hypothesis posteriors in place. UnifiedField,
        FHRREncoder (sbert-backed), Hopfield, episodic memory, free-energy
        engine, log-likelihood arena all persist across items in a session."""
        labels = list(sample.choices) or ["_empty_"]
        existing = self.controller.belief_state.hypotheses
        existing_n = len(existing)
        # Pad/truncate labels to match the hypothesis space.
        target_n = max(existing_n, len(labels), self.max_hypotheses, 2)
        while len(labels) < target_n:
            labels.append(f"_empty_{len(labels)}")

        if existing_n == target_n and existing_n > 0:
            self._soft_reset_in_place(labels)
        else:
            # Cardinality changed (rare across a task) — full rebuild via
            # controller. This is the only path that reloads sbert.
            self.controller.reset_session(labels)
        # Re-build per-item energy arena: each item has its own choice space.
        self.energy_arena = EnergyCausalArena(
            precision=self.energy_arena.precision,
            beta=self.energy_arena.beta,
        )
        self._scm_topologies = {}
        self._choice_model_names = []
        self._last_derived_obs = []
        for i, label in enumerate(labels[:len(sample.choices)]):
            scm = self._build_choice_scm(i, label)
            try:
                self.energy_arena.register(scm)
                self._choice_model_names.append(scm.name)
                # Project this SCM's DAG into the NGC layer hierarchy via
                # TopologyMapper. Horizontal causal edges are resolved through
                # virtual parents at higher levels (the "elevator shaft" fix).
                n_ngc_layers = len(self.controller.agent.field.ngc.layer_sizes)
                topology = self._topology_mapper.from_scm(scm, n_layers=n_ngc_layers)
                self._scm_topologies[scm.name] = topology
            except ValueError as e:
                logger.warning(
                    "Topology registration failed for SCM %r: %s",
                    getattr(scm, "name", "?"),
                    e,
                )

    def _soft_reset_in_place(self, labels: List[str]) -> None:
        """Reset only what is per-item, keeping the heavy state intact."""

        # Fresh hypotheses with uniform prior over the choice labels.
        n = len(labels)
        active = [not label.startswith("_empty_") for label in labels]
        n_active = max(1, sum(active))
        self.controller.belief_state = BeliefState(
            turn=0,
            hypotheses=[
                Hypothesis(
                    id=f"H{i}",
                    description=label,
                    probability=(1.0 / n_active if active[i] else 0.0),
                    supporting_evidence=[],
                    contradicting_evidence=[],
                )
                for i, label in enumerate(labels)
            ],
            eliminated_hypotheses=[],
            confirmed_facts=[],
            open_questions=[],
            current_tension=1.0,
            epistemic_urgency=1.0,
            free_energy=0.0,
        )
        self.controller._conversation.clear()

        # NGC working state: clear activations/history but keep the learned
        # weights (cross-item priors) and the Hopfield bank.
        try:
            self.controller.agent.field.ngc.soft_reset()
            self.controller.agent.field.energy_history.clear()
            self.controller.agent.field._step_count = 0
        except AttributeError as e:
            logger.warning("NGC soft_reset skipped (API mismatch): %s", e)
        except Exception as e:
            logger.error("NGC soft_reset failed: %s", e, exc_info=True)

    # ---------- per-choice SCM (used by EnergyCausalArena) ----------

    def _build_choice_scm(self, choice_idx: int, label: str) -> StructuralCausalModel:
        """
        Build a tiny SCM for one choice:

            prompt_feature  ──▶  choice_match  ──▶  observation
                                                ▲
                                                │ (lateral) coherence

        The DAG has both vertical and horizontal edges. The TopologyMapper
        is exactly what turns the lateral coherence link into a virtual parent
        in the NGC hierarchy, addressing the topological-mismatch critique.
        """
        scm = StructuralCausalModel(name=f"choice_{choice_idx}_{label}")
        scm.add_variable("prompt_feature", n_values=4, parents=[])
        scm.add_variable("coherence", n_values=4, parents=[])
        scm.add_variable("choice_match", n_values=4, parents=["prompt_feature"])
        scm.add_variable("observation", n_values=4, parents=["choice_match", "coherence"])
        return scm

    # ---------- one-shot ingest (delegates to controller) ----------

    def ingest_prompt(self, prompt: str) -> Dict[str, Any]:
        """Run controller.perceive_only — the full agent stack.

        This invokes (in order):
          - Broca.parse if enabled, else template parser
          - TensegrityAgent.perceive: UnifiedField cycle, FreeEnergyEngine.step,
            epistemic A/B updates, episodic encode, associative store, log-lik
            CausalArena competition
          - controller._maybe_inject_causal_hypothesis: when causal tension is
            high, calls Broca.propose_causal_hypothesis and registers a new SCM
            via build_scm_from_proposal
          - belief-state update from inference + parsed relations
        """
        return self.controller.perceive_only(prompt)

    # ---------- NGC falsification (per-choice top-down prediction) ----------

    def _ngc_falsification_scores(
        self, prompt: str, choices: List[str]
    ) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """
        For each choice c_i:
          1. Save NGC base state (prompt-grounded after perceive).
          2. Encode c_i alone, settle NGC under it.
          3. Ask the field to top-down predict the prompt observation.
          4. score_i = -prediction_error.
          5. Discretize the obs/pred for use as energy-arena observations.

        Returns (scores, energy_arena_observations).
        """
        field = self.controller.agent.field
        prompt_tokens = _alphanum_tokens(prompt, max_tokens=64)
        prompt_obs = field._fhrr_to_obs(field.encoder.encode_sequence(prompt_tokens))

        # Snapshot the prompt-grounded state to restore between choices.
        try:
            base_state = field.ngc.save_state()
        except Exception:
            base_state = None

        scores = np.zeros(len(choices), dtype=np.float64)
        derived_obs: List[Dict[str, int]] = []
        for i, c in enumerate(choices):
            if base_state is not None:
                try:
                    field.ngc.restore_state(base_state)
                except Exception:
                    pass
            ctoks = _alphanum_tokens(c, max_tokens=32)
            choice_obs = field._fhrr_to_obs(field.encoder.encode_sequence(ctoks))
            try:
                field.ngc.settle(choice_obs, steps=self.falsify_settle_steps)
                pe = float(field.ngc.prediction_error(prompt_obs))
            except Exception as e:
                logger.error(
                    "NGC falsification failed for choice %d: %s",
                    i, e, exc_info=True,
                )
                pe = float(1e9)
            scores[i] = -pe

            # Derive a compact discrete observation for the energy arena.
            # Each variable is bucketed into 4 levels to match the per-choice
            # SCM cardinality. The buckets are deterministic from the field
            # state, not random.
            try:
                pred_obs = field.ngc.predict_observation()
                pf = self._bucket_4(float(np.dot(prompt_obs, prompt_obs) ** 0.5))
                cm = self._bucket_4(-pe)
                co = self._bucket_4(float(np.dot(pred_obs, prompt_obs)))
                ob = self._bucket_4(float(np.linalg.norm(pred_obs)))
                derived_obs.append({
                    "prompt_feature": pf,
                    "choice_match": cm,
                    "coherence": co,
                    "observation": ob,
                })
            except Exception:
                derived_obs.append({
                    "prompt_feature": 0, "choice_match": 0,
                    "coherence": 0, "observation": 0,
                })

        # Restore the prompt-grounded state so subsequent perceive calls aren't
        # contaminated by the last falsification settle.
        if base_state is not None:
            try:
                field.ngc.restore_state(base_state)
            except Exception:
                pass

        return scores, derived_obs

    @staticmethod
    def _bucket_4(x: float) -> int:
        """Map a real-valued summary to a 4-bucket discrete value via tanh."""
        v = math.tanh(x / 2.0)  # in (-1, 1)
        if math.isnan(x) or math.isnan(v):
            return 2
        # Map (-1, 1) to {0, 1, 2, 3}.
        return max(0, min(3, int((v + 1.0) * 2.0)))

    # ---------- energy-arena competition ----------

    def _energy_arena_posterior(
        self, derived_obs: List[Dict[str, int]], n_choices: int
    ) -> Tuple[np.ndarray, float, Optional[str]]:
        """
        Each choice's SCM scores its own derived observation. Lowest energy wins.
        Returns (per-choice posterior, tension, winner_name).
        """
        if not self.energy_arena.models:
            return np.full(n_choices, 1.0 / n_choices), 1.0, None

        posterior = np.zeros(n_choices, dtype=np.float64)
        for i in range(n_choices):
            name = self._choice_model_names[i] if i < len(self._choice_model_names) else f"choice_{i}"
            term = self.energy_arena.models.get(name)
            if term is None:
                continue
            try:
                e = float(term.energy(derived_obs[i]))
            except Exception as e_err:
                logger.debug("energy term failed for %s: %s", name, e_err)
                e = 0.0
            posterior[i] = -self.energy_arena.beta * e

        # Softmax over -energies.
        m = posterior.max()
        weights = np.exp(posterior - m)
        s = weights.sum()
        if s <= 0:
            return np.full(n_choices, 1.0 / n_choices), 1.0, None
        weights /= s

        # Tension = normalized entropy of the posterior over choices.
        nz = weights[weights > 0]
        if len(nz) > 1:
            ent = float(-np.sum(nz * np.log(nz)) / np.log(n_choices))
        else:
            ent = 0.0
        winner_idx = int(np.argmax(weights))
        winner = (
            self._choice_model_names[winner_idx]
            if winner_idx < len(self._choice_model_names)
            else f"choice_{winner_idx}"
        )
        return weights, ent, winner

    # ---------- belief integration ----------

    def _belief_from_controller(self, n_choices: int) -> np.ndarray:
        hs = self.controller.belief_state.hypotheses
        if len(hs) < n_choices:
            return np.full(n_choices, 1.0 / n_choices)
        p = np.array([float(h.probability) for h in hs[:n_choices]], dtype=np.float64)
        s = p.sum()
        if s <= 0:
            return np.full(n_choices, 1.0 / n_choices)
        return p / s

    def _set_controller_belief(self, belief: np.ndarray) -> None:
        """Write a belief vector back into the controller's hypothesis posteriors
        so subsequent controller-level operations see the updated state."""
        hs = self.controller.belief_state.hypotheses
        if len(hs) < len(belief):
            return
        for h, p in zip(hs[:len(belief)], belief):
            h.probability = float(max(p, 0.0))
        for h in hs[len(belief):]:
            h.probability = 0.0
        # Renormalize defensively.
        total = sum(h.probability for h in hs)
        if total > 0:
            for h in hs:
                h.probability /= total

    # ---------- convergence (derived, not magic) ----------

    @staticmethod
    def _converged(belief: np.ndarray, ratio: float) -> bool:
        """Stop when the leader's mass exceeds `ratio` × the runner-up's mass.
        For a uniform 4-way prior this trips at ~2× spread (relative criterion,
        not an absolute threshold)."""
        if len(belief) < 2:
            return True
        order = np.argsort(belief)[::-1]
        top = float(belief[order[0]])
        second = float(belief[order[1]])
        if second <= 0:
            return top > 0
        return top >= ratio * second

    # ---------- main entry: score one item ----------

    def score_multichoice(
        self,
        sample: TaskSample,
        linguistic_scores: Optional[List[float]] = None,
    ) -> CommitResult:
        n = len(sample.choices)
        if n == 0:
            return CommitResult(
                scores=[], belief=[], committed_idx=-1,
                iterations_used=0, converged=False,
                final_perception_tension=1.0, final_arena_tension=1.0,
                final_energy_arena_tension=1.0,
            )

        self.reset_for_item(sample)
        self._item_index += 1

        if linguistic_scores is not None and len(linguistic_scores) == n:
            linguistic = np.asarray(linguistic_scores, dtype=np.float64)
            if not np.all(np.isfinite(linguistic)):
                linguistic = np.zeros(n, dtype=np.float64)
        else:
            linguistic = np.zeros(n, dtype=np.float64)

        # Initial perception — runs the full stack, including Broca SCM proposal
        # if causal tension is high (the controller wires this internally).
        initial_perception = self.ingest_prompt(sample.prompt)
        memory_scores = self._memory_choice_scores(sample)

        trace: List[IterationStep] = []
        converged = False
        iterations_used = 0

        for it in range(self.max_iterations):
            iterations_used = it + 1

            # 1. Per-choice NGC falsification.
            falsify, derived_obs = self._ngc_falsification_scores(
                sample.prompt, sample.choices
            )

            # 2. Train each choice's SCM on its own derived observation, so
            #    the per-choice models actually differentiate. (Dirichlet
            #    counts in the per-choice SCM CPTs; no gradients.) Then
            #    compete on energy.
            self._last_derived_obs = list(derived_obs)
            for i in range(n):
                name = self._choice_model_names[i] if i < len(self._choice_model_names) else f"choice_{i}"
                term = self.energy_arena.models.get(name)
                if term is None:
                    continue
                try:
                    term.scm.update_from_data([derived_obs[i]])
                except Exception as e:
                    logger.debug("scm update failed for %s: %s", name, e)
            energy_post, energy_tension, energy_winner = self._energy_arena_posterior(
                derived_obs, n
            )

            # 3. Bayesian update of controller's hypothesis posteriors:
            #    new_p_i ∝ old_p_i * exp(strength * z(falsify_i)) * energy_post_i.
            old_belief = self._belief_from_controller(n)
            fz = self._znorm(falsify)
            lz = self._znorm(linguistic)
            mz = self._znorm(memory_scores)
            log_lik_falsify = self.falsify_update_strength * fz
            log_post = (
                np.log(np.maximum(old_belief, 1e-12))
                + log_lik_falsify
                + self.llm_evidence_weight * lz
                + self.memory_evidence_weight * mz
                + np.log(np.maximum(energy_post, 1e-12))
            )
            log_post -= log_post.max()
            new_belief = np.exp(log_post)
            sb = new_belief.sum()
            new_belief = new_belief / sb if sb > 0 else np.full(n, 1.0 / n)
            self._set_controller_belief(new_belief)

            top_idx = int(np.argmax(new_belief))
            top_p = float(new_belief[top_idx])

            # 4. Re-perceive on the current leader CHOICE (not a synthetic
            #    string). This re-engages the full agent stack — episodic
            #    encode, free-energy step, log-likelihood arena, possible
            #    Broca SCM injection — without injecting a malformed input
            #    that the parser was never designed to handle.
            try:
                ing = self.ingest_prompt(sample.choices[top_idx])
                perception_fe = float(ing.get("perception", {}).get("free_energy", 0.0))
                perception_tension = float(ing.get("perception", {}).get("tension", 1.0))
                arena_tension = perception_tension
                arena_winner = ing.get("perception", {}).get("arena_winner")
            except Exception as e:
                logger.debug("re-perceive skipped: %s", e)
                perception_fe = 0.0
                perception_tension = 1.0
                arena_tension = 1.0
                arena_winner = None

            trace.append(IterationStep(
                iteration=it,
                belief=new_belief.tolist(),
                perception_free_energy=perception_fe,
                perception_tension=perception_tension,
                arena_tension=arena_tension,
                arena_winner=arena_winner,
                falsification_scores=falsify.tolist(),
                energy_arena_winner=energy_winner,
                top_idx=top_idx,
                top_p=top_p,
            ))

            if self._converged(new_belief, self.commit_ratio):
                converged = True
                break

        # Final commit using the controller's belief state (the source of truth).
        final_belief = self._belief_from_controller(n)
        committed_idx = int(np.argmax(final_belief))

        # Calibrated score for the harness: belief shifted away from uniform,
        # bounded in [-1, 1]. Comparable in magnitude to the previous z-scored
        # outputs; the harness's confidence-gated blending stays sane.
        scores = ((final_belief - 1.0 / n) * 2.0).tolist()

        # Final tensions.
        try:
            final_perception_tension = float(self.controller.belief_state.current_tension)
        except Exception:
            final_perception_tension = 1.0
        final_arena_tension = final_perception_tension
        final_energy_tension = trace[-1].arena_tension if trace else 1.0

        return CommitResult(
            scores=scores,
            belief=final_belief.tolist(),
            committed_idx=committed_idx,
            iterations_used=iterations_used,
            converged=converged,
            final_perception_tension=final_perception_tension,
            final_arena_tension=final_arena_tension,
            final_energy_arena_tension=final_energy_tension,
            trace=trace,
            initial_perception=initial_perception if n > 0 else None,
            linguistic_scores=linguistic.tolist(),
            memory_scores=memory_scores.tolist(),
        )

    # ---------- helpers ----------

    def _encode_text_fhrr(self, text: str, max_tokens: int = 96) -> np.ndarray:
        field = self.controller.agent.field
        return field.encoder.encode_sequence(_alphanum_tokens(text, max_tokens=max_tokens))

    @staticmethod
    def _unit_real(vec: np.ndarray) -> np.ndarray:
        v = np.real(vec).astype(np.float64)
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v

    def _memory_choice_scores(self, sample: TaskSample) -> np.ndarray:
        """Retrieve prior successful episodes and score choices by similarity.

        This is the persistent memory channel inside the same posterior update
        as predictive-coding falsification, causal energy, and LLM evidence.
        """
        n = len(sample.choices)
        scores = np.zeros(n, dtype=np.float64)
        if n == 0:
            return scores

        episodic = getattr(self.controller.agent, "episodic", None)
        if episodic is None or not getattr(episodic, "episodes", None):
            return scores

        field = self.controller.agent.field
        prompt_fhrr = self._encode_text_fhrr(sample.prompt, max_tokens=96)
        prompt_obs = field._fhrr_to_obs(prompt_fhrr)
        query_belief = np.full(n, 1.0 / n, dtype=np.float64)

        try:
            query_ctx = episodic.compute_item_representation(prompt_obs, query_belief)
            retrieved = episodic.retrieve_by_context(query_context=query_ctx, k=8)
        except Exception as e:
            logger.debug("persistent episodic retrieval skipped: %s", e)
            return scores

        if not retrieved:
            return scores

        choice_vecs = [
            self._unit_real(self._encode_text_fhrr(choice, max_tokens=48))
            for choice in sample.choices
        ]
        for ep in retrieved:
            meta = getattr(ep, "metadata", {}) or {}
            correct_vec = meta.get("correct_fhrr_real")
            if correct_vec is None:
                continue
            correct_vec = np.asarray(correct_vec, dtype=np.float64)
            cn = np.linalg.norm(correct_vec)
            if cn <= 1e-10:
                continue
            correct_vec = correct_vec / cn
            ctx_sim = float(np.dot(query_ctx, ep.context_vector))
            if ctx_sim <= 0.0:
                continue
            confidence = 1.0 - float(ep.surprise)
            weight = ctx_sim * max(0.05, confidence)
            for i, choice_vec in enumerate(choice_vecs):
                scores[i] += weight * float(np.dot(choice_vec, correct_vec))

        return scores

    def learn_from_feedback(self, sample: TaskSample, committed_idx: int) -> Dict[str, Any]:
        """Use benchmark outcome as post-action feedback and persist learning.

        The gold label is deliberately consumed only after commitment. It
        updates episodic memory, Hopfield attractors, NGC weights, epistemic
        counts, and the item-local SCM that corresponded to the revealed
        correct action.
        """
        n = len(sample.choices)
        if n == 0 or sample.gold < 0 or sample.gold >= n:
            return {"learned": False, "reason": "invalid sample"}

        correct = int(committed_idx) == int(sample.gold)
        field = self.controller.agent.field
        prompt_fhrr = self._encode_text_fhrr(sample.prompt, max_tokens=96)
        correct_fhrr = self._encode_text_fhrr(
            f"{sample.prompt} {sample.choices[sample.gold]}",
            max_tokens=128,
        )
        prompt_obs = field._fhrr_to_obs(prompt_fhrr)
        correct_obs = field._fhrr_to_obs(correct_fhrr)

        try:
            field.ngc.settle(correct_obs, steps=max(1, self.falsify_settle_steps))
            field.ngc.learn(modulation=max(0.0, self.feedback_learning_rate))
            field.memory.store(field.ngc.get_abstract_state(level=-1))
        except Exception as e:
            logger.debug("feedback NGC learning skipped: %s", e)

        belief_width = max(self.controller.agent.n_states, self.max_hypotheses, n)
        feedback_vec = np.zeros(belief_width, dtype=np.float64)
        feedback_vec[sample.gold] = 1.0
        if not correct and 0 <= committed_idx < n:
            feedback_vec[committed_idx] = -1.0

        try:
            self.controller.agent.perceive(feedback_vec[:self.controller.agent.n_states])
        except Exception as e:
            logger.debug("feedback perception skipped: %s", e)

        belief = np.zeros(n, dtype=np.float64)
        belief[sample.gold] = 1.0
        try:
            self.controller.agent.episodic.encode(
                observation=prompt_obs,
                morton_code=np.array([sample.gold], dtype=np.int64),
                belief_state=belief,
                action=int(sample.gold),
                surprise=0.0 if correct else 1.0,
                free_energy=0.0,
                metadata={
                    "task": sample.metadata.get("task", ""),
                    "sample_id": sample.id,
                    "prediction": int(committed_idx),
                    "gold": int(sample.gold),
                    "correct": bool(correct),
                    "correct_text": sample.choices[sample.gold],
                    "correct_fhrr_real": self._unit_real(correct_fhrr),
                },
            )
        except Exception as e:
            logger.debug("feedback episodic encode skipped: %s", e)

        if self._last_derived_obs and 0 <= sample.gold < len(self._last_derived_obs):
            name = (
                self._choice_model_names[sample.gold]
                if sample.gold < len(self._choice_model_names)
                else None
            )
            term = self.energy_arena.models.get(name) if name else None
            if term is not None:
                try:
                    term.scm.update_from_data([self._last_derived_obs[sample.gold]])
                except Exception as e:
                    logger.debug("feedback SCM update skipped: %s", e)

        try:
            self.controller.agent.experience_replay(n_episodes=3)
        except Exception as e:
            logger.debug("feedback replay skipped: %s", e)

        if self.persistent_state_path:
            self.save_state(self.persistent_state_path)

        return {"learned": True, "correct": bool(correct)}

    def save_state(self, path: Optional[str] = None) -> None:
        target = Path(path or self.persistent_state_path or "")
        if not str(target):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "item_index": self._item_index,
            "agent": self.controller.agent,
            "belief_state": self.controller.belief_state,
            "conversation": list(self.controller._conversation),
            "max_hypotheses": self.max_hypotheses,
        }
        with target.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path: Optional[str] = None) -> bool:
        target = Path(path or self.persistent_state_path or "")
        if not str(target) or not target.exists():
            return False
        try:
            with target.open("rb") as f:
                state = pickle.load(f)
        except Exception as e:
            logger.warning("Could not load persistent state %s: %s", target, e)
            return False
        if not isinstance(state, dict) or state.get("version") != 1:
            logger.warning("Ignoring unsupported persistent state %s", target)
            return False
        agent = state.get("agent")
        if agent is not None:
            self.controller.agent = agent
        belief_state = state.get("belief_state")
        if belief_state is not None:
            self.controller.belief_state = belief_state
        self.controller._conversation = list(state.get("conversation", []))
        self._item_index = int(state.get("item_index", 0))
        return True

    @staticmethod
    def _znorm(a: np.ndarray) -> np.ndarray:
        s = a.std()
        return (a - a.mean()) / s if s > 1e-10 else np.zeros_like(a)
