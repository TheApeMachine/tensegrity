"""
Cognitive Controller: Tensegrity as the cognitive body, LLM as the linguistic interface.

This is the main loop. Tensegrity owns:
  - Belief state (hypotheses, probabilities, evidence)
  - Memory (epistemic, episodic, associative)
  - Action selection (via Expected Free Energy minimization)
  - Causal reasoning (competing SCMs in the arena)

The LLM owns:
  - Parsing natural language → typed structs
  - Producing natural language from typed actions

The controller NEVER asks the LLM "what should I do?"
The controller NEVER lets the LLM see raw belief probabilities as editable.
The controller ALWAYS makes the decision, then asks the LLM to verbalize it.
"""

import numpy as np
from typing import Optional, Dict, List, Any
import logging
import re
from difflib import SequenceMatcher

from tensegrity.legacy.v1.agent import TensegrityAgent, DEFAULT_MEDIATED_SCM_NAME
from tensegrity.broca.schemas import (
    ParsedObservation,
    ParsedFeedback,
    BeliefState,
    CognitiveAction,
    Hypothesis,
    RelationMention,
)
from tensegrity.broca.interface import BrocaInterface
from tensegrity.causal.from_proposal import build_scm_from_proposal

logger = logging.getLogger(__name__)

IMPLICIT_RELATION_WEIGHT = 0.3
_REL_SEQUENCE_RATIO_THRESHOLD = 0.8


def _alnum_tokens(s: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", s) if t}


def _relation_term_matches_hypothesis(term: str, label: str) -> bool:
    """Stricter than substring overlap: tokens, regex word boundaries, or high string similarity."""
    term = term.strip().lower()
    label = label.strip().lower()

    if not term or not label:
        return False
    
    if term == label:
        return True
    
    if re.search(r"\b" + re.escape(term) + r"\b", label):
        return True
    
    if re.search(r"\b" + re.escape(label) + r"\b", term):
        return True
    
    tt, lt = _alnum_tokens(term), _alnum_tokens(label)
    
    if tt and lt and tt & lt:
        return True
    
    if SequenceMatcher(None, term, label).ratio() > _REL_SEQUENCE_RATIO_THRESHOLD:
        return True
    
    return False


class CognitiveController:
    """
    The integration layer: Tensegrity cognition + LLM language.
    
    This implements the full loop:
      1. Receive input (text from user/environment)
      2. LLM parses it → ParsedObservation
      3. Tensegrity processes the observation → updates beliefs, memory, causal arena
      4. Tensegrity selects an action via EFE minimization
      5. LLM verbalizes the action → natural language output
      6. Return output + full cognitive state
    
    The controller maintains a typed BeliefState that persists across turns.
    This is what makes it different from a stateless LLM: the beliefs are
    explicit, inspectable, and revised via Bayesian logic — not via
    attention over a rolling context window.
    """
    
    def __init__(
        self,
        agent: Optional[TensegrityAgent] = None,
        broca: Optional[BrocaInterface] = None,
        n_hypotheses: int = 8,
        hypothesis_labels: Optional[List[str]] = None,
        use_llm: bool = True,
        enable_hypothesis_generation: bool = False,
        confirmed_facts_max: int = 5,
    ):
        """
        Args:
            agent: TensegrityAgent instance. Created with defaults if None.
            broca: BrocaInterface instance. Created with defaults if None.
            n_hypotheses: Number of competing hypotheses to maintain
            hypothesis_labels: Labels for the hypothesis space
            use_llm: If False, uses template-based parse/produce (for testing without API)
            enable_hypothesis_generation: If True and use_llm, may add LLM-proposed SCMs when tension is high
            confirmed_facts_max: Rolling window size for belief_state.confirmed_facts (matches parse-context tail)
        """
        self.use_llm = use_llm
        self.enable_hypothesis_generation = enable_hypothesis_generation
        self.confirmed_facts_max = max(1, int(confirmed_facts_max))
        
        labels_for_hyp: Optional[List[str]] = None
        
        if hypothesis_labels is not None:
            labels_for_hyp = list(hypothesis_labels)
            n_states = max(len(labels_for_hyp), n_hypotheses, 2)
            while len(labels_for_hyp) < n_states:
                labels_for_hyp.append(f"_empty_{len(labels_for_hyp)}")
        else:
            n_states = max(n_hypotheses, 2)
        
        n_obs = n_states * 4  # Observation space > hypothesis space
        n_actions = 4  # ask, state, eliminate, conclude
        
        self.agent = agent or TensegrityAgent(
            n_states=n_states,
            n_observations=n_obs,
            n_actions=n_actions,
            sensory_dims=n_states,  # Morton dims = hypothesis dims
            sensory_bits=4,
            context_dim=32,
            associative_dim=64,
            planning_horizon=2,
            precision=4.0,
        )
        
        # Linguistic interface
        if use_llm and broca is None:
            self.broca = BrocaInterface()
        else:
            self.broca = broca
        
        # Belief state — persists across turns
        self.belief_state = BeliefState(
            turn=0,
            hypotheses=[],
            eliminated_hypotheses=[],
            confirmed_facts=[],
            open_questions=[],
            current_tension=1.0,
            epistemic_urgency=1.0,
            free_energy=0.0,
        )
        
        # Hypothesis labels
        if hypothesis_labels is not None and labels_for_hyp is not None:
            self._init_hypotheses(labels_for_hyp)
        
        # Conversation history (for LLM context — but NOT the belief state)
        self._conversation: List[Dict[str, str]] = []
        
        # Action type mapping: Tensegrity action index → CognitiveAction type
        self._action_map = {
            0: "ask_question",
            1: "state_belief",
            2: "eliminate_hypothesis",
            3: "state_conclusion",
        }
    
    def reset_session(self, hypothesis_labels: List[str]) -> None:
        """
        Start a fresh session for an independent item (e.g. one benchmark sample).

        Rebuilds the substrate agent with dimensions matched to the hypothesis
        count and clears conversational artifacts.
        """
        labels = list(hypothesis_labels)

        if not labels:
            labels = ["_empty_"]
        
        n_hyp = max(len(labels), 2)
        
        while len(labels) < n_hyp:
            labels.append(f"_empty_{len(labels)}")
        
        self.agent = TensegrityAgent(
            n_states=n_hyp,
            n_observations=n_hyp * 4,
            n_actions=4,
            sensory_dims=n_hyp,
            sensory_bits=4,
            context_dim=32,
            associative_dim=64,
            planning_horizon=2,
            precision=4.0,
        )
        
        self.belief_state = BeliefState(
            turn=0,
            hypotheses=[],
            eliminated_hypotheses=[],
            confirmed_facts=[],
            open_questions=[],
            current_tension=1.0,
            epistemic_urgency=1.0,
            free_energy=0.0,
        )
        
        self._init_hypotheses(labels)
        self._conversation.clear()

    def _trim_confirmed_facts(self) -> None:
        """Keep only the last ``confirmed_facts_max`` entries (rolling window)."""
        n = self.confirmed_facts_max
        facts = self.belief_state.confirmed_facts
        
        if len(facts) > n:
            self.belief_state.confirmed_facts = facts[-n:]

    def _record_parsed_facts(self, parsed: ParsedObservation) -> None:
        """Append structured parse results to confirmed facts with consistent formatting."""
        turn = self.belief_state.turn
        
        for entity in parsed.entities:
            fact = (
                f"[T{turn}] Observed: {entity.normalized} ({entity.entity_type})"
            )
        
            self.belief_state.confirmed_facts.append(fact)
        
        for relation in parsed.relations:
            fact = f"[T{turn}] {relation.subject} {relation.predicate} {relation.object}"
        
            if relation.negated:
                fact = f"NOT({fact})"
        
            self.belief_state.confirmed_facts.append(fact)
        
        for relation in parsed.implicit_relations:
            fact = (
                f"[T{turn}] (implicit) {relation.subject} "
                f"{relation.predicate} {relation.object}"
            )
        
            if relation.negated:
                fact = f"NOT({fact})"
        
            self.belief_state.confirmed_facts.append(fact)
        
        self._trim_confirmed_facts()
    
    def perceive_only(self, input_text: str) -> Dict[str, Any]:
        """
        Parse and run perception + belief update only (no action / no verbalization).
        """
        self.belief_state.turn += 1

        if self.use_llm and self.broca:
            parsed = self.broca.parse(input_text, context=self._get_parse_context())
        else:
            parsed = self._template_parse(input_text)
        
        obs_vector = self._observation_to_vector(parsed)
        perception = self.agent.perceive(obs_vector)
        
        self._maybe_inject_causal_hypothesis(perception, input_text)
        self._update_hypotheses_from_inference(perception, obs_vector)
        self._record_parsed_facts(parsed)
        
        return {
            "perception": {
                "free_energy": perception["free_energy"],
                "surprise": perception["surprise"],
                "tension": perception["arena"]["tension"],
                "epistemic_value": perception["epistemic_value"],
            },
            "belief_state": self.belief_state.model_dump(),
            "parsed_input": parsed.model_dump(),
            "turn": self.belief_state.turn,
        }
    
    def _init_hypotheses(self, labels: List[str]):
        """Initialize the hypothesis space with uniform priors."""
        n = len(labels)
        
        self.belief_state.hypotheses = [Hypothesis(
            id=f"H{i}",
            description=label,
            probability=1.0 / n,
            supporting_evidence=[],
            contradicting_evidence=[],
        ) for i, label in enumerate(labels)]
    
    @staticmethod
    def _apply_relation_evidence(
        features: np.ndarray,
        hyp_labels: Dict[str, int],
        relations: List[RelationMention],
        weight: float,
    ) -> None:
        """Add hypothesis-indexed evidence from typed relations (scaled by weight)."""
        for relation in relations:
            subj = relation.subject.lower()
            obj = relation.object.lower()
            
            subj_matches = [
                i for label, i in hyp_labels.items()
                if _relation_term_matches_hypothesis(subj, label)
            ]
            
            obj_matches = [
                i for label, i in hyp_labels.items()
                if _relation_term_matches_hypothesis(obj, label)
            ]
            
            sign = -1.0 if relation.negated else 1.0
            w = weight
            
            if relation.predicate in ("causes", "enables", "confirms", "is_a", "has_property"):
                for idx in obj_matches:
                    features[idx] += 0.8 * sign * w
                for idx in subj_matches:
                    features[idx] += 0.4 * sign * w
            
            elif relation.predicate in ("prevents", "contradicts"):
                for idx in obj_matches:
                    features[idx] -= 0.8 * sign * w
                for idx in subj_matches:
                    features[idx] -= 0.3 * sign * w
    
    def _maybe_inject_causal_hypothesis(self, perception: Dict[str, Any], input_text: str) -> None:
        """If causal fit is poor, ask Broca for a new SCM and register it (LLM only)."""
        if not self.enable_hypothesis_generation or not self.use_llm or not self.broca:
            return
        
        if not hasattr(self.broca, "propose_causal_hypothesis"):
            return
        
        ar = perception.get("arena") or {}
        
        if ar.get("tension", 0) < 0.72:
            return
        
        lls = ar.get("log_likelihoods") or {}
        
        if lls and max(lls.values()) > -2.0:
            return
        
        try:
            names = list(self.agent.arena.models.keys())
            prop = self.broca.propose_causal_hypothesis(input_text[:2000], names)
            scm = build_scm_from_proposal(prop)
        
            if scm.name in self.agent.arena.models:
                return
        
            self.agent.add_causal_model(scm)
            q = perception["belief_state"]
            obs_idx = perception["observation_index"]
        
            causal_obs: Dict[str, int] = {
                "state": int(np.argmax(q)),
                "observation": int(obs_idx),
            }
        
            if DEFAULT_MEDIATED_SCM_NAME in self.agent.arena.models:
                causal_obs["cause"] = int(np.argmax(q))
        
            perception["arena"] = self.agent.arena.compete(causal_obs)
        except (KeyError, ValueError, IndexError, TypeError) as e:
            logger.warning(
                "Dynamic causal hypothesis skipped [%s]: %s",
                type(e).__name__,
                e,
                exc_info=False,
            )
    
    def _observation_to_vector(self, parsed: ParsedObservation) -> np.ndarray:
        """
        Convert a ParsedObservation into a numeric vector for Tensegrity.
        
        This is the typed interface: the LLM's structured output becomes
        a numeric input to the cognitive engine. No freeform text crosses
        this boundary.
        
        The vector has one dimension per hypothesis. Positive values mean
        evidence FOR that hypothesis; negative values mean evidence AGAINST.
        """
        n = len(self.belief_state.hypotheses) or self.agent.n_states
        features = np.zeros(n)
        
        # Map entities and relations to hypothesis dimensions using the
        # known hypothesis labels. The LLM parser (or template fallback)
        # extracts entities that may match hypothesis names.
        hyp_labels = {h.description.lower(): i for i, h in enumerate(self.belief_state.hypotheses)}
        
        for entity in parsed.entities:
            ename = entity.normalized.lower()
        
            # Direct match: entity IS a hypothesis
            if ename in hyp_labels:
                features[hyp_labels[ename]] += 1.0
            else:
                # Partial match: entity mentions a hypothesis keyword
                for label, idx in hyp_labels.items():
                    if ename in label or label in ename:
                        features[idx] += 0.5
        
        self._apply_relation_evidence(features, hyp_labels, parsed.relations, weight=1.0)
        self._apply_relation_evidence(
            features, hyp_labels, parsed.implicit_relations, weight=IMPLICIT_RELATION_WEIGHT,
        )
        
        # Linguistic confidence modulates the whole vector
        features *= parsed.confidence_linguistic
        
        # Negation flips the sign
        if parsed.negation_present:
            features *= -0.5
        
        return features
    
    def _update_hypotheses_from_inference(
        self, perception_result: Dict[str, Any], obs_vector: np.ndarray,
    ) -> None:
        """
        Update hypothesis probabilities using BOTH:
          1. Tensegrity's generic state inference (q_states)
          2. Direct Bayesian update from the observation vector
        
        The observation vector encodes semantic evidence:
          obs_vector[i] > 0 → evidence FOR hypothesis i
          obs_vector[i] < 0 → evidence AGAINST hypothesis i
          obs_vector[i] = 0 → no evidence about hypothesis i
        
        We treat the observation vector as a log-likelihood ratio
        and multiply it into the prior to get the posterior.
        This is pure Bayes: P(H|E) ∝ P(E|H) P(H)
        """
        n = len(self.belief_state.hypotheses)
        
        # Get current priors
        priors = np.array([h.probability for h in self.belief_state.hypotheses])
        eliminated = set(self.belief_state.eliminated_hypotheses)
        active_mask = np.array(
            [h.id not in eliminated for h in self.belief_state.hypotheses],
            dtype=bool,
        )
        if not np.any(active_mask):
            active_mask[:] = True
        priors = np.where(active_mask, priors, 0.0)
        prior_total = priors.sum()
        if prior_total > 0:
            priors = priors / prior_total
        else:
            priors = active_mask.astype(np.float64)
            priors = priors / priors.sum()
        
        # Direct Bayesian update from observation vector
        # obs_vector acts as log-likelihood: positive = confirms, negative = contradicts
        if len(obs_vector) >= n:
            log_likelihood = obs_vector[:n]
        else:
            log_likelihood = np.zeros(n)
            log_likelihood[:len(obs_vector)] = obs_vector
        
        # Convert to likelihood ratios: exp(obs_vector)
        # Clamp to prevent overflow
        log_likelihood = np.clip(log_likelihood, -5.0, 5.0)
        likelihood = np.exp(log_likelihood)
        
        # Posterior ∝ likelihood × prior
        posterior = likelihood * priors
        
        # Also incorporate Tensegrity's state inference (blend)
        q_engine = perception_result['belief_state']

        if len(q_engine) >= n:
            engine_probs = q_engine[:n]
        else:
            engine_probs = np.ones(n) / n
        engine_probs = np.asarray(engine_probs, dtype=np.float64)
        engine_probs = np.where(active_mask, engine_probs, 0.0)
        engine_total = engine_probs.sum()
        if engine_total > 0:
            engine_probs = engine_probs / engine_total
        else:
            engine_probs = priors.copy()
        
        # Blend: 70% direct Bayesian, 30% engine inference
        # The direct update is more reliable when the parser extracts good signal
        blended = 0.7 * posterior + 0.3 * engine_probs
        blended = np.where(active_mask, blended, 0.0)
        
        # Normalize
        total = blended.sum()
        
        if total > 0:
            blended /= total
        else:
            blended = np.ones(n) / n
        
        # Apply to hypotheses
        for i, hyp in enumerate(self.belief_state.hypotheses):
            if i < n:
                hyp.probability = float(blended[i])
        
        # Eliminate hypotheses below threshold
        threshold = 0.005
        
        for hyp in self.belief_state.hypotheses:
            if hyp.id in eliminated:
                hyp.probability = 0.0
            elif hyp.probability < threshold:
                self.belief_state.eliminated_hypotheses.append(hyp.id)
                eliminated.add(hyp.id)
                hyp.probability = 0.0
        
        # Re-normalize after elimination
        total = sum(h.probability for h in self.belief_state.hypotheses)
        
        if total > 0:
            for h in self.belief_state.hypotheses:
                h.probability /= total
        
        # Update tension and free energy
        self.belief_state.current_tension = float(perception_result['arena']['tension'])
        self.belief_state.free_energy = float(perception_result['free_energy'])
        
        # Epistemic urgency: normalized entropy of posterior
        probs = np.array([h.probability for h in self.belief_state.hypotheses])
        probs = probs[probs > 0]
        
        if len(probs) > 1:
            entropy = float(-np.sum(probs * np.log(probs + 1e-16)))
            max_entropy = float(np.log(len(probs)))
            self.belief_state.epistemic_urgency = entropy / max_entropy if max_entropy > 0 else 0
        else:
            self.belief_state.epistemic_urgency = 0.0
    
    def _select_cognitive_action(self, perception_result: Dict[str, Any]) -> CognitiveAction:
        """
        Tensegrity selects the action. NOT the LLM.
        
        Decision logic based on EFE and belief state:
          - High epistemic urgency → ask_question (explore)
          - One hypothesis dominant → state_conclusion (exploit)
          - Evidence against a hypothesis → eliminate_hypothesis
          - Otherwise → state_belief (share current understanding)
        """
        action_idx = perception_result['action']
        action_type = self._action_map.get(action_idx, "state_belief")
        
        # Override based on epistemic urgency
        if self.belief_state.epistemic_urgency > 0.7:
            action_type = "ask_question"
        
        # Check if any hypothesis is dominant enough to conclude
        if self.belief_state.hypotheses:
            probs = [h.probability for h in self.belief_state.hypotheses]
            max_prob = max(probs) if probs else 0
            
            if max_prob > 0.85:
                action_type = "state_conclusion"
            elif max_prob < 0.15 and any(h.probability > 0.3 for h in self.belief_state.hypotheses):
                logger.info(
                    "Competing hypotheses remain (max_prob=%.3f) — keeping EFE-selected "
                    "action; no hypothesis elimination performed.",
                    max_prob,
                )
                pass  # Let the EFE-selected action stand
        
        # Build the action content
        target = None
        content = None
        confidence = 0.5
        
        if action_type == "ask_question":
            # Find the most uncertain variable
            if self.belief_state.hypotheses:
                # Ask about the hypothesis with highest entropy contribution
                active = [h for h in self.belief_state.hypotheses 
                         if h.id not in self.belief_state.eliminated_hypotheses]

                if active:
                    # Ask about the hypothesis closest to 0.5 (most uncertain)
                    most_uncertain = min(active, key=lambda h: abs(h.probability - 0.5))
                    target = most_uncertain.description
                    content = f"Need to distinguish: {most_uncertain.description}"

            confidence = self.belief_state.epistemic_urgency
            
        elif action_type == "state_conclusion":
            if self.belief_state.hypotheses:
                winner = max(self.belief_state.hypotheses, key=lambda h: h.probability)
                target = winner.description
                content = winner.description
                confidence = winner.probability
                
        elif action_type == "eliminate_hypothesis":
            if self.belief_state.hypotheses:
                weakest = min(
                    [h for h in self.belief_state.hypotheses 
                     if h.id not in self.belief_state.eliminated_hypotheses],
                    key=lambda h: h.probability,
                    default=None
                )
                
                if weakest:
                    target = weakest.description
                    content = f"Ruling out {weakest.description}"
                    confidence = 1.0 - weakest.probability
                    
        elif action_type == "state_belief":
            if self.belief_state.hypotheses:
                top = max(self.belief_state.hypotheses, key=lambda h: h.probability)
                target = top.description
                content = f"Current best guess: {top.description} (p={top.probability:.2f})"
                confidence = top.probability
        
        return CognitiveAction(
            action_type=action_type,
            target=target,
            content=content,
            confidence=confidence,
        )
    
    def step(self, input_text: str) -> Dict[str, Any]:
        """
        One full cognitive cycle:
          1. Parse input (LLM)
          2. Process observation (Tensegrity)
          3. Select action (Tensegrity)
          4. Verbalize action (LLM)
          5. Return everything
        
        Args:
            input_text: Natural language input from user/environment
        
        Returns:
            Full cycle results including output text, belief state, and diagnostics
        """
        self.belief_state.turn += 1
        
        # === 1. PARSE (LLM as input transducer) ===
        if self.use_llm and self.broca:
            parsed = self.broca.parse(input_text, context=self._get_parse_context())
        else:
            parsed = self._template_parse(input_text)
        
        # === 2. PROCESS (Tensegrity cognition) ===
        obs_vector = self._observation_to_vector(parsed)
        perception = self.agent.perceive(obs_vector)
        self._maybe_inject_causal_hypothesis(perception, input_text)
        
        # Update hypothesis probabilities from Tensegrity beliefs
        self._update_hypotheses_from_inference(perception, obs_vector)
        
        # Update confirmed facts and evidence
        self._record_parsed_facts(parsed)

        # === 3. SELECT ACTION (Tensegrity decides) ===
        action = self._select_cognitive_action(perception)
        
        # === 4. VERBALIZE (LLM as output transducer) ===
        if self.use_llm and self.broca:
            utterance = self.broca.produce(action, self.belief_state)
            output_text = utterance.text
        else:
            output_text = self.broca.produce_simple(action) if self.broca else self._template_produce(action)
        
        # Track conversation
        self._conversation.append({"role": "user", "content": input_text})
        self._conversation.append({"role": "assistant", "content": output_text})
        
        return {
            "output": output_text,
            "action": action.model_dump(),
            "belief_state": self.belief_state.model_dump(),
            "perception": {
                "free_energy": perception["free_energy"],
                "surprise": perception["surprise"],
                "tension": perception["arena"]["tension"],
                "epistemic_value": perception["epistemic_value"],
            },
            "parsed_input": parsed.model_dump(),
            "turn": self.belief_state.turn,
            "hypotheses": [
                {"id": h.id, "desc": h.description, "prob": round(h.probability, 4)}
                for h in self.belief_state.hypotheses
            ],
        }
    
    def step_with_feedback(self, feedback_text: str, 
                           action_taken: str) -> Dict[str, Any]:
        """
        Process feedback from the environment after an action.
        
        This updates beliefs based on whether the action succeeded or not,
        using the feedback to confirm/contradict hypotheses.
        """
        if self.use_llm and self.broca:
            hyp_descriptions = [h.description for h in self.belief_state.hypotheses
                               if h.id not in self.belief_state.eliminated_hypotheses]
            feedback = self.broca.parse_feedback(feedback_text, action_taken, hyp_descriptions)
        else:
            feedback = self._template_parse_feedback(feedback_text)
        
        # Update evidence on hypotheses
        if feedback.confirms_hypothesis:
            for h in self.belief_state.hypotheses:
                if feedback.confirms_hypothesis.lower() in h.description.lower():
                    h.supporting_evidence.append(f"[T{self.belief_state.turn}] {feedback_text}")
        
        if feedback.contradicts_hypothesis:
            for h in self.belief_state.hypotheses:
                if feedback.contradicts_hypothesis.lower() in h.description.lower():
                    h.contradicting_evidence.append(f"[T{self.belief_state.turn}] {feedback_text}")
        
        # Add new information
        for info in feedback.new_information:
            self.belief_state.confirmed_facts.append(f"[T{self.belief_state.turn}] {info}")
        
        self._trim_confirmed_facts()
        
        return self.step(feedback_text)
    
    def _get_parse_context(self) -> str:
        """Build context string for parse calls."""
        ctx_parts = []
        
        if self.belief_state.confirmed_facts:
            slice_n = self.confirmed_facts_max
            facts_str = '; '.join(self.belief_state.confirmed_facts[-slice_n:])
            ctx_parts.append(f"Known facts: {facts_str}")
        
        if self.belief_state.hypotheses:
            active = [h for h in self.belief_state.hypotheses 
                     if h.id not in self.belief_state.eliminated_hypotheses]
        
            if active:
                ctx_parts.append(f"Active hypotheses: {', '.join(h.description for h in active[:5])}")
        
        return " | ".join(ctx_parts) if ctx_parts else "New conversation"
    
    def _template_parse(self, text: str) -> ParsedObservation:
        """
        Hypothesis-aware fallback parser when LLM is unavailable.
        
        Scans the text for keywords that match hypothesis labels and builds
        entities/relations based on co-occurrence with semantic markers.
        This is the semantic bridge in template mode.
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        from tensegrity.broca.schemas import EntityMention, RelationMention
        
        entities = []
        relations = []
        
        # Get hypothesis labels for matching
        hyp_labels = [h.description.lower() for h in self.belief_state.hypotheses]
        
        # 1. Find hypothesis mentions in text
        mentioned_hyps = []
        
        for label in hyp_labels:
            # Check for the label or its parts in the text
            label_parts = label.replace("_", " ").split()
        
            for part in label_parts:
                if len(part) > 2 and part in text_lower:
                    mentioned_hyps.append(label)
                    entities.append(EntityMention(
                        text=part, entity_type="object", normalized=label
                    ))
        
                    break
        
        # 2. Detect semantic polarity markers
        positive_markers = {"has", "can", "is", "does", "with", "found", "seen", 
                          "shows", "resembl", "over", "common", "confirm", "support",
                          "toward", "point", "evidence", "known", "happen"}
        
        negative_markers = {"not", "no", "never", "without", "lacks", "doesn't", 
                          "isn't", "can't", "won't", "neither", "nor", "rule",
                          "doesn't", "cannot", "absent", "miss"}
        
        has_positive = any(m in text_lower for m in positive_markers)
        has_negative = any(m in text_lower for m in negative_markers)
        
        # 3. Build property entities for non-hypothesis keywords that appear as clue content
        property_keywords = {
            # Animal game
            "feather": ["parrot"], "fur": ["cat", "dog", "hamster", "rabbit"],
            "speech": ["parrot"], "speak": ["parrot"], "talk": ["parrot"],
            "leg": ["cat", "dog", "hamster", "rabbit", "turtle"], 
            "four legs": ["cat", "dog", "hamster", "rabbit"],
            "swim": ["goldfish", "turtle"], "fly": ["parrot"],
            "scale": ["snake", "goldfish"], "shell": ["turtle"],
            "50 years": ["parrot", "turtle"], "long life": ["parrot", "turtle"],
            # RCA game
            "cpu": ["cpu_overload"], "memory": ["memory_leak", "deadlock"],
            "disk": ["disk_full"], "network": ["network_timeout"],
            "dns": ["dns_failure"], "config": ["config_error"],
            "connection": ["dependency_crash", "network_timeout"],
            "refused": ["dependency_crash"], "upstream": ["dependency_crash"],
            "503": ["dependency_crash"], "timeout": ["network_timeout"],
            "deploy": ["config_error", "dependency_crash"],
            "health": ["dependency_crash"],
            # Mystery game
            "study": ["librarian", "butler"], "kitchen": ["chef", "butler"],
            "book": ["librarian"], "garden": ["gardener"], "mud": ["gardener"],
            "safe": ["librarian", "butler"], "combination": ["librarian", "butler"],
            "arguing": ["librarian"], "driv": ["driver"],
        }
        
        for keyword, associated_hyps in property_keywords.items():
            if keyword in text_lower:
                entities.append(EntityMention(
                    text=keyword, entity_type="property", normalized=keyword
                ))
        
                # Create relations between property and associated hypotheses
                for ahyp in associated_hyps:
                    if ahyp.lower() in [h.lower() for h in hyp_labels]:
                        predicate = "confirms" if not has_negative else "contradicts"
        
                        relations.append(RelationMention(
                            subject=keyword,
                            predicate=predicate,
                            object=ahyp,
                            negated=has_negative
                        ))
        
        # 4. If text directly mentions a hypothesis with sentiment
        for label in hyp_labels:
            parts = label.replace("_", " ").split()
        
            for part in parts:
                if len(part) > 2 and part in text_lower:
                    if has_negative and not has_positive:
                        relations.append(RelationMention(
                            subject="clue", predicate="contradicts", object=label, negated=False
                        ))
                    elif has_positive:
                        relations.append(RelationMention(
                            subject="clue", predicate="confirms", object=label, negated=False
                        ))
        
        negation_words = {"not", "no", "never", "neither", "nor", "isn't", "doesn't", "can't", "won't", "does not"}
        negation = any(nw in text_lower for nw in negation_words)
        
        is_question = text.strip().endswith("?")
        is_command = any(w in words[:2] for w in ["tell", "show", "give", "find", "do"])
        
        return ParsedObservation(
            entities=entities,
            relations=relations,
            implicit_relations=[],
            is_question=is_question,
            is_assertion=not is_question and not is_command,
            is_command=is_command,
            negation_present=negation,
            temporal_marker=None,
            confidence_linguistic=0.7 if relations else 0.3,
        )
    
    def _template_produce(self, action: CognitiveAction) -> str:
        """Fallback producer when LLM is unavailable."""
        templates = {
            "ask_question": f"I need to know more about {action.target}. Can you tell me?",
            "state_belief": f"Based on what I've seen, I think: {action.content}",
            "propose_hypothesis": f"Here's a possibility: {action.content}",
            "eliminate_hypothesis": f"I can rule out {action.target}.",
            "state_conclusion": f"I'm confident the answer is: {action.content}",
            "defer": "I need more information before I can say.",
            "request_intervention": f"What would change if we modified {action.target}?",
        }
        
        return templates.get(action.action_type, str(action.content))
    
    def _template_parse_feedback(self, text: str) -> ParsedFeedback:
        """Fallback feedback parser."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["yes", "correct", "right", "exactly"]):
            outcome = "success"
        elif any(w in text_lower for w in ["no", "wrong", "incorrect", "nope"]):
            outcome = "failure"
        else:
            outcome = "ambiguous"
        
        return ParsedFeedback(
            outcome=outcome,
            confirms_hypothesis=None,
            contradicts_hypothesis=None,
            new_information=[],
            surprise_linguistic=0.5,
        )
    
    def get_state_summary(self) -> str:
        """Human-readable summary of the cognitive state."""
        lines = [f"=== Turn {self.belief_state.turn} ==="]
        lines.append(f"Tension: {self.belief_state.current_tension:.3f}")
        lines.append(f"Epistemic urgency: {self.belief_state.epistemic_urgency:.3f}")
        lines.append(f"Free energy: {self.belief_state.free_energy:.3f}")
        
        if self.belief_state.hypotheses:
            lines.append("\nHypotheses:")
        
            for h in sorted(self.belief_state.hypotheses, key=lambda x: x.probability, reverse=True):
                status = "✗" if h.id in self.belief_state.eliminated_hypotheses else " "
                lines.append(f"  [{status}] {h.id} {h.description}: {h.probability:.4f}")
        
        if self.belief_state.confirmed_facts:
            lines.append(f"\nConfirmed facts ({len(self.belief_state.confirmed_facts)}):")
        
            for fact in self.belief_state.confirmed_facts[-5:]:
                lines.append(f"  {fact}")
        
        return "\n".join(lines)
    
    @property
    def statistics(self) -> Dict[str, Any]:
        return {
            "turns": self.belief_state.turn,
            "active_hypotheses": sum(
                1 for h in self.belief_state.hypotheses 
                if h.id not in self.belief_state.eliminated_hypotheses
            ),
            "eliminated": len(self.belief_state.eliminated_hypotheses),
            "confirmed_facts": len(self.belief_state.confirmed_facts),
            "tension": self.belief_state.current_tension,
            "epistemic_urgency": self.belief_state.epistemic_urgency,
            "free_energy": self.belief_state.free_energy,
            "broca": self.broca.statistics if self.broca and hasattr(self.broca, 'statistics') else {},
        }
