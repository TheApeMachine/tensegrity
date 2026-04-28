"""
Canonical Tensegrity pipeline — one code path for benchmarks and generation.

Composes:
  • CognitiveController + TensegrityAgent (Broca parse, free-energy inference,
    causal arena, dynamic SCM injection when enabled)
  • ScoringBridge (FHRR + NGC + Hopfield scoring for multiple-choice items)

Benchmark mode scores each option by fusing LLM log-probabilities with this
stack. Hybrid generation can reuse the same controller for logit grafting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tensegrity.broca.controller import CognitiveController
from tensegrity.bench.tasks import TaskSample
from tensegrity.engine.scoring import ScoringBridge

logger = logging.getLogger(__name__)


class CanonicalPipeline:
    """
    Full cognitive stack shared by benchmarks and the hybrid graft path.

    For each MC item: reset controller + field scorer state, ingest the prompt,
    score choices with ScoringBridge, fuse hypothesis posterior with field scores.
    """

    def __init__(
        self,
        hypothesis_labels: List[str],
        *,
        use_llm_broca: bool = False,
        enable_hypothesis_generation: bool = False,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        belief_blend: float = 0.35,
        obs_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        fhrr_dim: int = 2048,
        ngc_settle_steps: int = 30,
        ngc_learning_rate: float = 0.01,
        hopfield_beta: float = 0.05,
        confidence_threshold: float = 0.15,
        context_settle_steps: int = 40,
        choice_settle_steps: int = 25,
        context_learning_epochs: int = 3,
    ):
        # model_name: kept for API stability and future wiring to remote models / logging
        self.model_name = model_name
        self.belief_blend = belief_blend
        self.controller = CognitiveController(
            n_hypotheses=max(len(hypothesis_labels), 2),
            hypothesis_labels=hypothesis_labels,
            use_llm=use_llm_broca,
            enable_hypothesis_generation=enable_hypothesis_generation,
        )
        self.scoring = ScoringBridge(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims or [128, 32],
            fhrr_dim=fhrr_dim,
            ngc_settle_steps=ngc_settle_steps,
            ngc_learning_rate=ngc_learning_rate,
            hopfield_beta=hopfield_beta,
            confidence_threshold=confidence_threshold,
            context_settle_steps=context_settle_steps,
            choice_settle_steps=choice_settle_steps,
            context_learning_epochs=context_learning_epochs,
        )

    def reset_for_multichoice(self, sample: TaskSample) -> None:
        """I.I.D. benchmark item: fresh agent memories / arena and field scorer."""
        labels = list(sample.choices)
        if not labels:
            labels = ["_empty_"]
        self.controller.reset_session(labels)
        self.scoring.reset()

    def ingest_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse + perceive (+ optional causal hypothesis); no verbalization."""
        return self.controller.perceive_only(prompt)

    def score_multichoice(self, sample: TaskSample) -> Tuple[List[float], float, Dict[str, Any]]:
        """
        Run the full stack on one TaskSample.

        Returns:
            combined_scores: fused list for argmax over choices
            gate_entropy: scorer gate entropy
            diagnostics: raw components for debugging / logging
        """
        self.reset_for_multichoice(sample)
        ing = self.ingest_prompt(sample.prompt)
        field_scores, entropy = self.scoring.score_choices(sample.prompt, sample.choices)
        field_arr = np.asarray(field_scores, dtype=np.float64)
        agent_probs = self._agent_choice_posterior(len(sample.choices))
        combined = self._fuse_field_and_hypotheses(field_arr, agent_probs)
        return (
            combined.tolist(),
            float(entropy),
            {
                "field_scores": field_scores,
                "agent_probs": agent_probs.tolist(),
                "perception_tension": ing.get("perception", {}).get("tension"),
                "free_energy": ing.get("perception", {}).get("free_energy"),
            },
        )

    def _agent_choice_posterior(self, n_choices: int) -> np.ndarray:
        hs = self.controller.belief_state.hypotheses
        if len(hs) != n_choices:
            return np.ones(n_choices, dtype=np.float64) / max(n_choices, 1)
        p = np.array([float(h.probability) for h in hs], dtype=np.float64)
        s = p.sum()
        if s <= 0:
            return np.ones(n_choices, dtype=np.float64) / n_choices
        return p / s

    def _fuse_field_and_hypotheses(self, field: np.ndarray, agent_probs: np.ndarray) -> np.ndarray:
        if field.shape != agent_probs.shape:
            logger.warning(
                "_fuse_field_and_hypotheses: shape mismatch field=%s agent_probs=%s; returning field unchanged",
                field.shape,
                agent_probs.shape,
            )
            return field
        zf = (field - field.mean()) / (field.std() + 1e-8)
        a = np.log(agent_probs + 1e-12)
        za = (a - a.mean()) / (a.std() + 1e-8)
        return zf + self.belief_blend * za

