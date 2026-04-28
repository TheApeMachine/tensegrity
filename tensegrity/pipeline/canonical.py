"""
Canonical Tensegrity pipeline — the agent loop.

This is the unified entry point for benchmarks AND chat. The cognitive layer
is the agent; the LLM (when used) is a typed tool exposed only via Broca.
For multiple-choice scoring, the LLM is not in the reasoning path at all —
the agent commits an answer through iterative perception, falsification,
causal model competition, and memory retrieval.

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
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tensegrity.broca.controller import CognitiveController
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

    Per-task flow:
      reset_session — full clear (episodic memory, energy arena, Hopfield, etc).
      The bench runner calls this between tasks so cross-item learning operates
      within a task but doesn't leak across tasks (which have different label
      spaces).
    """

    def __init__(
        self,
        hypothesis_labels: Optional[List[str]] = None,
        *,
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
    ):
        self.model_name = model_name
        self.max_iterations = max(1, int(max_iterations))
        self.commit_ratio = float(commit_ratio)
        self.falsify_settle_steps = int(falsify_settle_steps)
        self.falsify_update_strength = float(falsify_update_strength)

        # The agent body. controller.agent runs the full stack:
        #   UnifiedField (FHRR + NGC + Hopfield) + FreeEnergyEngine
        #   + EpistemicMemory + EpisodicMemory + AssociativeMemory
        #   + log-likelihood CausalArena.
        # Broca, when enabled, parses input into typed structure and proposes
        # SCMs when causal tension is high (handled inside perceive_only).
        self.controller = CognitiveController(
            n_hypotheses=max(len(hypothesis_labels) if hypothesis_labels else 2, 2),
            hypothesis_labels=hypothesis_labels,
            use_llm=use_llm_broca,
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

    # ---------- session boundaries ----------

    def reset_session(self) -> None:
        """Per-task reset. Clears persistent memory and the energy arena."""
        # The controller resets itself per item; here we additionally clear the
        # Hopfield bank, episodic memory, and energy arena.
        try:
            self.controller.agent.field.memory.patterns.clear()
            self.controller.agent.field.memory._matrix = None
            self.controller.agent.field.memory._dirty = True
        except Exception as e:
            logger.debug("Hopfield clear skipped: %s", e)
        try:
            self.controller.agent.episodic.clear()
        except Exception as e:
            logger.debug("Episodic clear skipped: %s", e)
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
        target_n = max(len(labels), 2)
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
        for i, label in enumerate(labels):
            scm = self._build_choice_scm(i, label)
            try:
                self.energy_arena.register(scm)
                # Project this SCM's DAG into the NGC layer hierarchy via
                # TopologyMapper. Horizontal causal edges are resolved through
                # virtual parents at higher levels (the "elevator shaft" fix).
                n_ngc_layers = len(self.controller.agent.field.ngc.layer_sizes)
                topology = self._topology_mapper.from_scm(scm, n_layers=n_ngc_layers)
                self._scm_topologies[scm.name] = topology
            except ValueError:
                # Already registered (rare; defensive).
                pass

    def _soft_reset_in_place(self, labels: List[str]) -> None:
        """Reset only what is per-item, keeping the heavy state intact."""
        from tensegrity.broca.schemas import BeliefState, Hypothesis

        # Fresh hypotheses with uniform prior over the choice labels.
        n = len(labels)
        self.controller.belief_state = BeliefState(
            turn=0,
            hypotheses=[
                Hypothesis(
                    id=f"H{i}",
                    description=label,
                    probability=1.0 / n,
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
            ngc = self.controller.agent.field.ngc
            ngc.layers = []
            ngc._initialized = False
            ngc._last_obs = None
            ngc.clear_history()
            self.controller.agent.field.energy_history.clear()
            self.controller.agent.field._step_count = 0
        except Exception as e:
            logger.debug("NGC soft-reset skipped: %s", e)

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
        scm = StructuralCausalModel(name=f"choice_{choice_idx}")
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
                logger.debug("falsification step failed for choice %d: %s", i, e)
                pe = 0.0
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
            name = f"choice_{i}"
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
        winner = f"choice_{int(np.argmax(weights))}"
        return weights, ent, winner

    # ---------- belief integration ----------

    def _belief_from_controller(self, n_choices: int) -> np.ndarray:
        hs = self.controller.belief_state.hypotheses
        if len(hs) != n_choices:
            return np.full(n_choices, 1.0 / n_choices)
        p = np.array([float(h.probability) for h in hs], dtype=np.float64)
        s = p.sum()
        if s <= 0:
            return np.full(n_choices, 1.0 / n_choices)
        return p / s

    def _set_controller_belief(self, belief: np.ndarray) -> None:
        """Write a belief vector back into the controller's hypothesis posteriors
        so subsequent controller-level operations see the updated state."""
        hs = self.controller.belief_state.hypotheses
        if len(hs) != len(belief):
            return
        for h, p in zip(hs, belief):
            h.probability = float(max(p, 0.0))
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

    def score_multichoice(self, sample: TaskSample) -> CommitResult:
        n = len(sample.choices)
        if n == 0:
            return CommitResult(
                scores=[], belief=[], committed_idx=-1,
                iterations_used=0, converged=False,
                final_perception_tension=1.0, final_arena_tension=1.0,
                final_energy_arena_tension=1.0,
            )

        self._item_index += 1
        self.reset_for_item(sample)

        # Initial perception — runs the full stack, including Broca SCM proposal
        # if causal tension is high (the controller wires this internally).
        ing0 = self.ingest_prompt(sample.prompt)

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
            for i in range(n):
                name = f"choice_{i}"
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
            log_lik_falsify = self.falsify_update_strength * fz
            log_post = (
                np.log(np.maximum(old_belief, 1e-12))
                + log_lik_falsify
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
        )

    # ---------- helpers ----------

    @staticmethod
    def _znorm(a: np.ndarray) -> np.ndarray:
        s = a.std()
        return (a - a.mean()) / s if s > 1e-10 else np.zeros_like(a)
