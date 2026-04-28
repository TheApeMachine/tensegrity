"""
Typed protocol schemas for the Broca interface.

These Pydantic models define the ONLY data shapes that cross the boundary
between Tensegrity (cognitive layer) and the LLM (linguistic layer).

The LLM can ONLY return instances of these schemas. It cannot return
freeform text, reasoning chains, action proposals, or evaluations.
The LLM is a transducer: structured-in → structured-out.

CRITICAL DESIGN RULE:
  - No `reasoning: str` field. No `explanation: str` field.
  - No `suggested_action: str` field. No `plan: str` field.
  - Every field is a typed primitive, a bounded numeric, or a closed enum.
  - If the LLM can express an opinion in a field, that field shouldn't exist.
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal
from enum import Enum


# ============================================================
# PARSE SCHEMAS (LLM reads natural language → structured data)
# ============================================================

class EntityMention(BaseModel):
    """A single entity extracted from text."""
    text: str = Field(description="The surface form as it appears in the input")
    entity_type: Literal["object", "agent", "location", "quantity", "property", "event", "other"]
    normalized: str = Field(description="Canonical/normalized form")


class RelationMention(BaseModel):
    """A relationship between two entities."""
    subject: str
    predicate: Literal[
        "causes", "prevents", "enables", "contains", "is_a", 
        "has_property", "located_at", "before", "after", "contradicts",
        "confirms", "unknown"
    ]
    object: str
    negated: bool = False


_SCM_IDENTIFIER_MAX = 64


class CausalEdge(BaseModel):
    """One edge in a proposed structural causal model (SCM)."""
    source: str = Field(
        min_length=1,
        max_length=_SCM_IDENTIFIER_MAX,
        description="Cause or enabling variable name",
    )
    target: str = Field(
        min_length=1,
        max_length=_SCM_IDENTIFIER_MAX,
        description="Effect variable name",
    )
    mechanism: Literal["causes", "prevents", "enables"]


class ProposedSCM(BaseModel):
    """LLM-proposed SCM as a named DAG plus short description."""
    name: str = Field(max_length=_SCM_IDENTIFIER_MAX, description="Short identifier, suitable for SCM.name")
    description: str = Field(max_length=512, description="One sentence: what this model claims")
    edges: List[CausalEdge] = Field(max_length=48, description="Directed edges; must be acyclic")

    @model_validator(mode="after")
    def _edges_must_be_acyclic(self) -> "ProposedSCM":
        from collections import defaultdict, deque

        edges = [(e.source.strip(), e.target.strip()) for e in self.edges]
        adj: defaultdict[str, list[str]] = defaultdict(list)
        indegree: defaultdict[str, int] = defaultdict(int)
        nodes: set[str] = set()
        for s, t in edges:
            if not s or not t:
                continue
            nodes.add(s)
            nodes.add(t)
            adj[s].append(t)
            indegree[t] += 1

        q: deque[str] = deque([n for n in nodes if indegree.get(n, 0) == 0])
        visited_count = 0
        while q:
            u = q.popleft()
            visited_count += 1
            for v in adj[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)
        if edges and visited_count != len(nodes):
            raise ValueError("ProposedSCM.edges must form a DAG; a cycle was detected.")
        return self


class ParsedObservation(BaseModel):
    """
    Schema for LLM-as-parser: convert natural language into structured observation.
    
    The LLM extracts WHAT was said. Tensegrity decides what it MEANS.
    """
    entities: List[EntityMention] = Field(default_factory=list)
    relations: List[RelationMention] = Field(default_factory=list)
    implicit_relations: List[RelationMention] = Field(
        default_factory=list,
        max_length=48,
        description=(
            "Typed implications required for consistency with the text but not literally stated; "
            "use closed predicates only (same vocabulary as relations)."
        ),
    )
    is_question: bool = Field(description="Is the input asking for information?")
    is_assertion: bool = Field(description="Is the input stating a fact/claim?")
    is_command: bool = Field(description="Is the input requesting an action?")
    negation_present: bool = Field(description="Does the input contain negation?")
    temporal_marker: Optional[Literal["past", "present", "future", "hypothetical"]] = None
    confidence_linguistic: float = Field(
        ge=0.0, le=1.0, 
        description="How clear/unambiguous is the input? 1.0 = perfectly clear"
    )


class ParsedFeedback(BaseModel):
    """
    Schema for parsing environment/user feedback after an action.
    
    The LLM classifies the feedback. Tensegrity updates beliefs.
    """
    outcome: Literal["success", "failure", "partial", "ambiguous", "no_feedback"]
    confirms_hypothesis: Optional[str] = Field(
        default=None, description="Which hypothesis (if any) this feedback supports"
    )
    contradicts_hypothesis: Optional[str] = Field(
        default=None, description="Which hypothesis (if any) this feedback contradicts"
    )
    new_information: List[str] = Field(
        default_factory=list, 
        description="Facts revealed by this feedback that weren't known before"
    )
    surprise_linguistic: float = Field(
        ge=0.0, le=1.0,
        description="How unexpected is this feedback given the context? 1.0 = very surprising"
    )


# ============================================================
# PRODUCE SCHEMAS (Tensegrity state → LLM → natural language)
# ============================================================

class Utterance(BaseModel):
    """
    Schema for LLM-as-verbalizer: convert Tensegrity's chosen action into language.
    
    Tensegrity decides WHAT to communicate. The LLM decides HOW to say it.
    """
    text: str = Field(description="The natural language output")
    register: Literal["formal", "casual", "technical", "empathetic"] = "casual"


class QuestionUtterance(BaseModel):
    """Schema for when Tensegrity decides to ask a question (epistemic action)."""
    question_text: str = Field(description="The question to ask")
    target_variable: str = Field(description="What variable/hypothesis this question targets")
    expected_information_gain: float = Field(ge=0.0, le=1.0)


# ============================================================
# INTERNAL COGNITIVE SCHEMAS (Tensegrity-owned, NOT for LLM)
# ============================================================

class Hypothesis(BaseModel):
    """A hypothesis maintained by Tensegrity's causal arena."""
    id: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    causal_model_name: Optional[str] = None


class BeliefState(BaseModel):
    """
    The full cognitive state of the agent. 
    
    This is what Tensegrity maintains across turns.
    The LLM receives a READ-ONLY serialization of this.
    The LLM CANNOT modify it — only Tensegrity can.
    """
    turn: int = 0
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    eliminated_hypotheses: List[str] = Field(
        default_factory=list, 
        description="Hypotheses that have been falsified — LLM must not re-introduce"
    )
    confirmed_facts: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    current_tension: float = Field(
        ge=0.0, le=1.0, 
        description="Entropy of posterior over competing models"
    )
    epistemic_urgency: float = Field(
        ge=0.0, le=1.0,
        description="How much the agent needs to gather information vs act"
    )
    free_energy: float = 0.0


class CognitiveAction(BaseModel):
    """
    An action selected by Tensegrity's inference engine.
    
    This is NOT generated by the LLM. The LLM only verbalizes it.
    """
    action_type: Literal[
        "ask_question",      # Epistemic: reduce uncertainty
        "state_belief",      # Declare what the agent believes
        "propose_hypothesis", # Offer a causal explanation
        "eliminate_hypothesis", # Falsify a hypothesis
        "request_intervention", # Ask to do(X=x) — causal experiment
        "state_conclusion",  # Declare final answer
        "defer",             # Not enough information to act
    ]
    target: Optional[str] = None  # What variable/hypothesis this targets
    content: Optional[str] = None  # Structured content for the action
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
