"""
Broca's Interface: The LLM as a linguistic transducer.

Two operations only:
  1. PARSE:   natural language → ParsedObservation (structured extraction)
  2. PRODUCE: CognitiveAction + BeliefState → Utterance (verbal realization)

The LLM never sees raw belief state without context framing.
The LLM never proposes actions — only Tensegrity does.
The LLM never evaluates hypotheses — only Tensegrity does.

This module uses the OpenAI SDK pointed at HF's inference router
with Pydantic schema enforcement. The LLM physically cannot return
data that doesn't match the schema.
"""

import os
import json
import logging
from typing import Optional, Tuple, Type, TypeVar, Union, List

from pydantic import BaseModel

from tensegrity.broca.schemas import (
    ParsedObservation,
    ParsedFeedback,
    Utterance,
    QuestionUtterance,
    BeliefState,
    CognitiveAction,
    ProposedSCM,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

_CAUSAL_SUMMARY_MAX_CHARS = 2400


class DuplicateModelNameError(ValueError):
    """Raised when ProposedSCM.name collides with ``existing_model_names`` from the causal arena."""

def truncate_to_sentence(text: str, max_len: int = _CAUSAL_SUMMARY_MAX_CHARS) -> Tuple[str, bool]:
    """
    Trim ``text`` to at most ``max_len`` characters, preferring truncation after the last
    complete sentence (. ? ! optionally followed by space) within that window.
    """
    if len(text) <= max_len:
        return text, False

    chunk = text[:max_len]
    last_break_end = -1
    i = 0

    while i < len(chunk):
        if chunk[i] in ".?!" and (i + 1 == len(chunk) or chunk[i + 1].isspace()):
            last_break_end = i + 1

        i += 1

    if last_break_end > 0:
        return chunk[:last_break_end].rstrip(), True

    cut = chunk.rfind(" ")

    if cut > max_len // 2:
        return chunk[:cut].rstrip(), True

    return chunk, True


class BrocaInterface:
    """
    LLM as Broca's area: parse language in, produce language out.
    
    The interface enforces typed schemas on all LLM calls.
    No freeform generation. No reasoning chains. No action proposals.
    
    Uses OpenAI SDK pointed at HuggingFace inference router.
    """
    
    def __init__(self, 
                 model: str = "Qwen/Qwen2.5-72B-Instruct",
                 api_key: Optional[str] = None,
                 base_url: str = "https://router.huggingface.co/v1",
                 temperature: float = 0.0,
                 max_parse_tokens: int = 512,
                 max_produce_tokens: int = 256):
        """
        Args:
            model: HF model ID for the LLM
            api_key: HF token. If None, reads from HF_TOKEN env var.
            base_url: Inference endpoint URL
            temperature: 0.0 for deterministic parsing
            max_parse_tokens: Budget for parse calls (keep small)
            max_produce_tokens: Budget for produce calls (keep small)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai  — required for Broca interface")
        
        self.model = model
        self.temperature = temperature
        self.max_parse_tokens = max_parse_tokens
        self.max_produce_tokens = max_produce_tokens
        
        api_key = api_key or os.environ.get("HF_TOKEN")

        if not api_key:
            raise ValueError("HF_TOKEN environment variable or api_key required")
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
        # Call counter for diagnostics
        self._parse_calls = 0
        self._hypothesis_calls = 0
        self._produce_calls = 0
        self._total_tokens = 0
    
    def _call_llm(self, messages: list, schema: Type[T], max_tokens: int) -> T:
        """
        Core LLM call with schema enforcement.
        
        Returns a validated Pydantic instance. The LLM physically
        cannot return anything that doesn't match the schema.
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=schema,
                max_tokens=max_tokens,
                temperature=self.temperature,
                seed=42,
            )
            
            if completion.usage:
                self._total_tokens += completion.usage.total_tokens
            
            result = completion.choices[0].message.parsed
            if result is None:
                # Fallback: try manual parse from content
                content = completion.choices[0].message.content

                if content:
                    result = schema.model_validate_json(content)
                else:
                    raise ValueError("LLM returned empty response")
            
            return result
            
        except Exception as e:
            logger.error(f"Broca LLM call failed: {e}")
            raise
    
    def parse(self, text: str, context: Optional[str] = None) -> ParsedObservation:
        """
        PARSE: Natural language → ParsedObservation.
        
        The LLM extracts entities, relations, and linguistic features.
        It does NOT interpret, evaluate, or reason about the content.
        
        Args:
            text: Raw natural language input
            context: Optional framing context (e.g., "This is a game clue")
        
        Returns:
            ParsedObservation with typed fields
        """
        system_prompt = (
            "You are a linguistic parser. Extract structured information from the input.\n"
            "relations: predicates that are DIRECTLY stated in the text.\n"
            "implicit_relations: the SAME RelationMention shape for links that are NOT quoted "
            "but are logically required for the scenario to hold (commonsense bridges only). "
            "Keep implicit_relations sparse; do not invent unrelated facts.\n"
            "Do NOT output prose reasoning — only typed fields. "
            "If something is unclear, set confidence_linguistic lower."
        )

        if context:
            system_prompt += f"\n\nContext: {context}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        
        self._parse_calls += 1

        return self._call_llm(messages, ParsedObservation, self.max_parse_tokens)
    
    def propose_causal_hypothesis(
        self,
        situation_summary: str,
        existing_model_names: List[str],
    ) -> ProposedSCM:
        """
        Propose a new structural causal model when existing SCMs fit poorly.

        Returns a bounded DAG schema only (no free-form reasoning).
        """
        system_prompt = (
            "You are a causal model designer. Propose ONE small directed acyclic graph "
            "as variable names and typed edges (causes / prevents / enables). "
            "Use short snake_case identifiers. At most 12 edges. "
            "Name must differ from existing model names. Output only the schema fields."
        )
        existing = ", ".join(existing_model_names[:24]) if existing_model_names else "(none)"
        summary, did_truncate = truncate_to_sentence(situation_summary, _CAUSAL_SUMMARY_MAX_CHARS)

        if did_truncate:
            logger.warning(
                "situation_summary truncated for causal hypothesis prompt: "
                "original_length=%d max_chars=%d",
                len(situation_summary),
                _CAUSAL_SUMMARY_MAX_CHARS,
            )

        user_content = (
            f"Existing models: {existing}\n\n"
            f"Observations / situation:\n{summary}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        self._hypothesis_calls += 1
        proposed = self._call_llm(messages, ProposedSCM, self.max_parse_tokens)
        existing_lower = {n.casefold(): n for n in existing_model_names}

        if proposed.name.casefold() in existing_lower:
            raise DuplicateModelNameError(
                f"LLM proposed duplicate SCM name {proposed.name!r}; existing names included "
                f"{existing_lower[proposed.name.casefold()]!r}. Update prompts or regenerate."
            )

        return proposed
    
    def parse_feedback(self, feedback: str, 
                       action_taken: str,
                       hypotheses: list) -> ParsedFeedback:
        """
        Parse feedback/response after an action was taken.
        
        The LLM classifies the outcome relative to known hypotheses.
        It does NOT generate new hypotheses or suggest actions.
        """
        hyp_list = "\n".join(f"  - {h}" for h in hypotheses) if hypotheses else "  (none)"
        
        system_prompt = (
            "You are a feedback classifier. Given an action that was taken and "
            "the resulting feedback, classify the outcome. "
            "Do NOT suggest next actions. Do NOT reason about strategy. "
            "Only classify what the feedback tells us.\n\n"
            f"Action taken: {action_taken}\n"
            f"Current hypotheses:\n{hyp_list}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": feedback},
        ]
        
        self._parse_calls += 1

        return self._call_llm(messages, ParsedFeedback, self.max_parse_tokens)
    
    def produce(self, action: CognitiveAction, 
                belief_state: BeliefState,
                audience: str = "user") -> Utterance:
        """
        PRODUCE: CognitiveAction + BeliefState → natural language Utterance.
        
        Tensegrity has already decided WHAT to do.
        The LLM decides HOW to say it.
        
        Args:
            action: The action selected by Tensegrity's inference engine
            belief_state: Current belief state (READ-ONLY context for LLM)
            audience: Who the output is for
        
        Returns:
            Utterance with natural language text
        """
        # Serialize belief state as read-only context
        # Strip internal fields the LLM doesn't need
        belief_summary = {
            "confirmed_facts": belief_state.confirmed_facts[-5:],  # Last 5 only
            "open_questions": belief_state.open_questions[-3:],
            "current_confidence": 1.0 - belief_state.current_tension,
        }
        
        system_prompt = (
            "You are a language production module. Given a cognitive action and context, "
            "produce a natural language utterance that realizes the action. "
            "Do NOT add information beyond what the action specifies. "
            "Do NOT offer opinions, suggestions, or commentary. "
            "Match the register to the audience.\n\n"
            f"Audience: {audience}\n"
            f"Context: {json.dumps(belief_summary)}"
        )
        
        action_description = (
            f"Action type: {action.action_type}\n"
            f"Target: {action.target or 'N/A'}\n"
            f"Content: {action.content or 'N/A'}\n"
            f"Confidence: {action.confidence:.2f}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Produce an utterance for this action:\n{action_description}"},
        ]
        
        self._produce_calls += 1
        
        if action.action_type == "ask_question":
            # Use question schema for questions
            result = self._call_llm(messages, QuestionUtterance, self.max_produce_tokens)
            return Utterance(text=result.question_text, register="casual")
        else:
            return self._call_llm(messages, Utterance, self.max_produce_tokens)
    
    def produce_simple(self, action: CognitiveAction) -> str:
        """
        Lightweight production: skip full LLM call for simple actions.
        
        Uses templates for deterministic, fast responses.
        Falls back to LLM only when the action needs nuanced language.
        """
        templates = {
            "ask_question": f"Can you tell me about {action.target}?",
            "state_belief": f"Based on what I know, I believe {action.content}.",
            "propose_hypothesis": f"Here's a possibility: {action.content}",
            "eliminate_hypothesis": f"I can rule out {action.target} — {action.content}.",
            "state_conclusion": f"My conclusion: {action.content}",
            "defer": "I don't have enough information yet to be sure.",
            "request_intervention": f"What happens if we change {action.target}?",
        }
        
        return templates.get(action.action_type, f"[{action.action_type}]: {action.content}")
    
    @property
    def statistics(self):
        return {
            "parse_calls": self._parse_calls,
            "hypothesis_calls": self._hypothesis_calls,
            "produce_calls": self._produce_calls,
            "total_tokens": self._total_tokens,
            "model": self.model,
        }
