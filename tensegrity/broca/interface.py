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
import re
from typing import Any, Optional, Tuple, Type, TypeVar, Union, List

from pydantic import BaseModel

from tensegrity.broca.schemas import (
    ParsedObservation,
    ParsedFeedback,
    Utterance,
    QuestionUtterance,
    BeliefState,
    CognitiveAction,
    ProposedSCM,
    CausalEdge,
    EntityMention,
    RelationMention,
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


def _json_object_from_text(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from model output."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _hypotheses_from_context(context: Optional[str]) -> List[str]:
    if not context:
        return []
    marker = "Active hypotheses:"
    idx = context.find(marker)
    if idx < 0:
        return []
    tail = context[idx + len(marker):]
    tail = tail.split("|", 1)[0]
    return [h.strip() for h in tail.split(",") if h.strip()]


def _snake_identifier(text: str, fallback: str, max_len: int = 48) -> str:
    """Normalize arbitrary text into a short SCM-safe identifier."""
    parts = re.findall(r"[A-Za-z0-9]+", text.lower())
    ident = "_".join(parts[:6]).strip("_")
    if not ident:
        ident = fallback
    if ident[0].isdigit():
        ident = f"v_{ident}"
    return ident[:max_len].strip("_") or fallback


def _unique_scm_name(base: str, existing_model_names: List[str]) -> str:
    existing = {n.casefold() for n in existing_model_names}
    root = _snake_identifier(base, "broca_model", max_len=48)
    name = root
    i = 1
    while name.casefold() in existing:
        suffix = f"_{i}"
        name = f"{root[:64 - len(suffix)]}{suffix}"
        i += 1
    return name


class DeterministicBrocaInterface:
    """
    Schema-valid Broca transducer used when no remote/local structured LLM
    parse is available.

    This keeps the controller path honest: parsing still crosses the Broca
    boundary and returns ``ParsedObservation``. It is not a scorer and it does
    not choose actions.
    """

    def __init__(self, model: str = "deterministic-broca"):
        self.model = model
        self._parse_calls = 0
        self._hypothesis_calls = 0
        self._produce_calls = 0
        self._total_tokens = 0

    def parse(self, text: str, context: Optional[str] = None) -> ParsedObservation:
        self._parse_calls += 1
        text_lower = text.lower()
        words = re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?", text_lower)
        hypotheses = _hypotheses_from_context(context)
        entities: List[EntityMention] = []
        relations: List[RelationMention] = []

        seen_entities = set()

        def add_entity(surface: str, entity_type: str = "object", normalized: Optional[str] = None):
            norm = (normalized or surface).strip().lower()
            key = (surface.strip().lower(), norm, entity_type)
            if not surface.strip() or key in seen_entities:
                return
            seen_entities.add(key)
            entities.append(EntityMention(
                text=surface.strip(),
                entity_type=entity_type,  # type: ignore[arg-type]
                normalized=norm,
            ))

        for token in words[:32]:
            if len(token) > 2:
                add_entity(token, "other", token)

        positive_markers = {
            "because", "cause", "causes", "caused", "therefore", "result", "supports",
            "support", "confirms", "shows", "has", "is", "are", "leads", "enables",
        }
        negative_markers = {
            "not", "no", "never", "without", "lacks", "contradicts", "false",
            "incorrect", "cannot", "can't", "isn't", "doesn't", "prevents",
        }
        has_positive = any(w in text_lower for w in positive_markers)
        has_negative = any(w in text_lower for w in negative_markers)

        for hyp in hypotheses:
            hyp_lower = hyp.lower()
            parts = [p for p in re.findall(r"[a-zA-Z0-9]+", hyp_lower) if len(p) > 2]
            mentioned = hyp_lower in text_lower or any(p in text_lower for p in parts)
            if not mentioned:
                continue
            add_entity(hyp, "object", hyp_lower)
            if has_negative and not has_positive:
                relations.append(RelationMention(
                    subject="input",
                    predicate="contradicts",
                    object=hyp,
                    negated=False,
                ))
            else:
                relations.append(RelationMention(
                    subject="input",
                    predicate="confirms",
                    object=hyp,
                    negated=False,
                ))

        causal_patterns = [
            (r"\b(.{1,64}?)\s+(?:causes|caused|leads to|produces|results in)\s+(.{1,64}?)(?:[.?!]|$)", "causes"),
            (r"\b(.{1,64}?)\s+(?:prevents|blocks|stops)\s+(.{1,64}?)(?:[.?!]|$)", "prevents"),
            (r"\b(.{1,64}?)\s+(?:enables|allows)\s+(.{1,64}?)(?:[.?!]|$)", "enables"),
        ]
        for pattern, predicate in causal_patterns:
            for m in re.finditer(pattern, text_lower):
                subj = m.group(1).strip(" ,;:")
                obj = m.group(2).strip(" ,;:")
                if subj and obj:
                    add_entity(subj, "event", subj)
                    add_entity(obj, "event", obj)
                    relations.append(RelationMention(
                        subject=subj,
                        predicate=predicate,  # type: ignore[arg-type]
                        object=obj,
                        negated=False,
                    ))

        negation_words = {"not", "no", "never", "neither", "nor", "isn't", "doesn't", "can't", "won't", "cannot"}
        negation = any(nw in text_lower for nw in negation_words)
        is_question = text.strip().endswith("?")
        is_command = bool(words[:1] and words[0] in {"tell", "show", "give", "find", "choose", "pick", "answer"})

        return ParsedObservation(
            entities=entities[:64],
            relations=relations[:48],
            implicit_relations=[],
            is_question=is_question,
            is_assertion=not is_question and not is_command,
            is_command=is_command,
            negation_present=negation,
            temporal_marker=None,
            confidence_linguistic=0.75 if relations else 0.45,
        )

    def parse_feedback(self, feedback: str, action_taken: str, hypotheses: list) -> ParsedFeedback:
        self._parse_calls += 1
        text_lower = feedback.lower()
        if any(w in text_lower for w in ["correct", "right", "success", "yes"]):
            outcome = "success"
        elif any(w in text_lower for w in ["wrong", "incorrect", "failure", "no"]):
            outcome = "failure"
        else:
            outcome = "ambiguous"
        return ParsedFeedback(
            outcome=outcome,  # type: ignore[arg-type]
            confirms_hypothesis=None,
            contradicts_hypothesis=None,
            new_information=[],
            surprise_linguistic=0.5,
        )

    def propose_causal_hypothesis(
        self,
        situation_summary: str,
        existing_model_names: List[str],
    ) -> ProposedSCM:
        """
        Deterministic Broca-side SCM proposal.

        The proposal is deliberately small and compatible with the agent arena's
        observed variables (``state`` and ``observation``), while allowing one
        latent contextual cause extracted from the current situation.
        """
        self._hypothesis_calls += 1
        summary, _ = truncate_to_sentence(situation_summary, max_len=512)
        text_lower = summary.lower()
        name = _unique_scm_name("broca_contextual_causal", existing_model_names)

        causal_patterns = [
            r"\b(.{1,48}?)\s+(?:causes|caused|leads to|produces|results in)\s+(.{1,48}?)(?:[.?!]|$)",
            r"\b(.{1,48}?)\s+(?:enables|allows)\s+(.{1,48}?)(?:[.?!]|$)",
            r"\b(.{1,48}?)\s+(?:prevents|blocks|stops)\s+(.{1,48}?)(?:[.?!]|$)",
        ]
        latent_source = "context_signal"
        latent_effect = "state"
        mechanism: str = "causes"
        for pattern in causal_patterns:
            m = re.search(pattern, text_lower)
            if not m:
                continue
            latent_source = _snake_identifier(m.group(1), "context_signal")
            latent_effect = _snake_identifier(m.group(2), "state")
            matched = m.group(0)
            if "prevent" in matched or "block" in matched or "stop" in matched:
                mechanism = "prevents"
            elif "enable" in matched or "allow" in matched:
                mechanism = "enables"
            else:
                mechanism = "causes"
            break

        if latent_source in {"cause", "state", "observation"}:
            latent_source = "context_signal"
        if latent_effect in {"cause", "state", "observation"}:
            latent_effect = "context_signal"

        edges = [
            CausalEdge(source="cause", target=latent_source, mechanism="enables"),
            CausalEdge(source=latent_source, target="state", mechanism=mechanism),  # type: ignore[arg-type]
            CausalEdge(source="state", target="observation", mechanism="causes"),
        ]
        if latent_effect != latent_source:
            edges.insert(2, CausalEdge(source=latent_source, target=latent_effect, mechanism=mechanism))  # type: ignore[arg-type]

        return ProposedSCM(
            name=name,
            description=summary[:512] or "Contextual causal bridge between latent cause, state, and observation.",
            edges=edges,
        )

    def produce(self, action: CognitiveAction, belief_state: BeliefState, audience: str = "user") -> Utterance:
        self._produce_calls += 1
        return Utterance(text=self.produce_simple(action), register="casual")

    def produce_simple(self, action: CognitiveAction) -> str:
        templates = {
            "ask_question": f"Can you tell me about {action.target}?",
            "state_belief": f"Based on what I know, I believe {action.content}.",
            "propose_hypothesis": f"Here's a possibility: {action.content}",
            "eliminate_hypothesis": f"I can rule out {action.target}.",
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


class LocalBrocaInterface(DeterministicBrocaInterface):
    """
    Broca parser backed by the local benchmark LLM.

    The local model is asked for a bounded JSON object and the result is
    schema-validated. Invalid generations fall back to the deterministic
    transducer so cognition still receives a valid typed observation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str = "local-broca",
        max_parse_tokens: int = 256,
    ):
        super().__init__(model=model_name)
        self._model_obj = model
        self._tokenizer = tokenizer
        self.max_parse_tokens = int(max_parse_tokens)

    def parse(self, text: str, context: Optional[str] = None) -> ParsedObservation:
        self._parse_calls += 1
        try:
            generated = self._generate_parse_json(text, context)
            json_text = _json_object_from_text(generated)
            if json_text is None:
                raise ValueError("local Broca emitted no JSON object")
            return ParsedObservation.model_validate_json(json_text)
        except Exception as e:
            logger.debug("Local Broca parse fell back to deterministic transducer: %s", e)
            # Avoid double-counting fallback as a separate Broca call.
            self._parse_calls -= 1
            parsed = super().parse(text, context)
            return parsed

    def propose_causal_hypothesis(
        self,
        situation_summary: str,
        existing_model_names: List[str],
    ) -> ProposedSCM:
        self._hypothesis_calls += 1
        try:
            generated = self._generate_proposal_json(situation_summary, existing_model_names)
            json_text = _json_object_from_text(generated)
            if json_text is None:
                raise ValueError("local Broca emitted no ProposedSCM JSON object")
            proposed = ProposedSCM.model_validate_json(json_text)
            nodes = {e.source for e in proposed.edges} | {e.target for e in proposed.edges}
            if "state" not in nodes or "observation" not in nodes:
                raise ValueError("local Broca ProposedSCM did not include state and observation")
            existing_lower = {n.casefold(): n for n in existing_model_names}
            if proposed.name.casefold() in existing_lower:
                proposed.name = _unique_scm_name(proposed.name, existing_model_names)
            return proposed
        except Exception as e:
            logger.debug("Local Broca SCM proposal fell back to deterministic transducer: %s", e)
            self._hypothesis_calls -= 1
            return super().propose_causal_hypothesis(situation_summary, existing_model_names)

    def _generate_parse_json(self, text: str, context: Optional[str]) -> str:
        import torch

        schema_hint = {
            "entities": [{"text": "surface text", "entity_type": "object", "normalized": "canonical"}],
            "relations": [{"subject": "x", "predicate": "causes", "object": "y", "negated": False}],
            "implicit_relations": [],
            "is_question": False,
            "is_assertion": True,
            "is_command": False,
            "negation_present": False,
            "temporal_marker": None,
            "confidence_linguistic": 0.8,
        }
        system = (
            "You are Broca: a linguistic transducer. Extract only typed structure. "
            "Do not answer, explain, or evaluate. Return only valid JSON matching this shape: "
            f"{json.dumps(schema_hint)}"
        )
        user = f"Context: {context or 'New conversation'}\n\nInput:\n{text}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            encoded = self._tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
            attention_mask = (
                getattr(encoded, "attention_mask", None)
                if hasattr(encoded, "input_ids")
                else None
            )
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        else:
            prompt = f"{system}\n\n{user}\n\nJSON:"
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

        if hasattr(self._model_obj, "device"):
            input_ids = input_ids.to(self._model_obj.device)
            attention_mask = attention_mask.to(self._model_obj.device)

        pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
        with torch.no_grad():
            out = self._model_obj.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_parse_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_proposal_json(
        self,
        situation_summary: str,
        existing_model_names: List[str],
    ) -> str:
        import torch

        schema_hint = {
            "name": "broca_contextual_causal_1",
            "description": "One sentence describing the causal model.",
            "edges": [
                {"source": "cause", "target": "state", "mechanism": "enables"},
                {"source": "state", "target": "observation", "mechanism": "causes"},
            ],
        }
        existing = ", ".join(existing_model_names[:24]) if existing_model_names else "(none)"
        summary, _ = truncate_to_sentence(situation_summary, _CAUSAL_SUMMARY_MAX_CHARS)
        system = (
            "You are Broca proposing one small SCM schema for the causal arena. "
            "Return only valid JSON. Use short snake_case identifiers. "
            "The graph must be acyclic and must include state and observation so "
            "the arena can score the model from agent observations. Shape: "
            f"{json.dumps(schema_hint)}"
        )
        user = f"Existing model names: {existing}\n\nSituation:\n{summary}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            encoded = self._tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
            attention_mask = (
                getattr(encoded, "attention_mask", None)
                if hasattr(encoded, "input_ids")
                else None
            )
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        else:
            prompt = f"{system}\n\n{user}\n\nJSON:"
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

        if hasattr(self._model_obj, "device"):
            input_ids = input_ids.to(self._model_obj.device)
            attention_mask = attention_mask.to(self._model_obj.device)

        pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
        with torch.no_grad():
            out = self._model_obj.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_parse_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
