"""
Smoke test: can Llama 3.2 1B reliably TRANSDUCE prompts into structure the
cognitive layer can consume?

The LLM never judges, scores, or opines. It only converts text into typed
fields (entities, relations, role-filler bindings). All reasoning happens
downstream in NGC / arena / FHRR.

We measure: does transduction produce sensible structure for the prompt
at hand — entities present in the text, relations grounded in the text,
roles bound to actual phrases.
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import List, Literal, TypedDict

from pydantic import BaseModel, Field

from outlines import models, generate

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

_LOGGER = logging.getLogger(__name__)

try:
    from outlines.errors import OutlinesStructuredGenerationError as _OutlineGenError
except ImportError:
    _OutlineGenError = None  # type: ignore[misc, assignment]


class Entity(BaseModel):
    text: str = Field(description="The exact phrase as it appears in the prompt")
    kind: Literal["object", "person", "concept", "property", "action", "quantity", "place"]


class Relation(BaseModel):
    subject: str = Field(description="Entity phrase from the prompt")
    predicate: str = Field(description="A 1-3 word relation label, e.g. 'fits in', 'causes', 'is part of'")
    object: str = Field(description="Entity phrase from the prompt")


class RoleBinding(BaseModel):
    role: str = Field(description="A short role name, e.g. 'cause', 'effect', 'referent', 'premise', 'question_focus'")
    filler: str = Field(description="The phrase from the prompt that fills this role")


class PromptStructure(BaseModel):
    entities: List[Entity]
    relations: List[Relation]
    roles: List[RoleBinding]


class SmokeItem(TypedDict):
    task: str
    prompt: str


# Hand-picked items spanning the benchmark behavior zones.
ITEMS: list[SmokeItem] = [
    {
        "task": "truthfulqa",
        "prompt": "What happens if you crack your knuckles a lot?",
    },
    {
        "task": "mmlu_philosophy",
        "prompt": "According to Kant, the moral worth of an action depends on what factor?",
    },
    {
        "task": "winogrande (pronoun)",
        "prompt": "The trophy didn't fit in the suitcase because it was too large. What does 'it' refer to?",
    },
    {
        "task": "arc_challenge",
        "prompt": "Which property of a mineral can be determined just by looking at it?",
    },
    {
        "task": "copa (causal)",
        "prompt": "The man turned on the faucet. What was the effect?",
    },
    {
        "task": "needle-in-lies",
        "prompt": (
            "Statements: (1) The key is in the kitchen drawer. (2) The key is under the oak table. "
            "(3) The key is on the windowsill. (4) Statement (2) is a lie. (5) Statement (4) is a lie. "
            "Where is the key?"
        ),
    },
    {
        "task": "hellaswag (continuation)",
        "prompt": (
            "A woman is seen standing on a diving board. She bounces twice and then "
            "arches her back. What happens next?"
        ),
    },
    {
        "task": "strategy_qa (multi-hop)",
        "prompt": "Could a person born in 1900 have voted for Franklin D. Roosevelt for president?",
    },
]


def build_prompt(item: SmokeItem) -> str:
    return (
        "You convert a question into structured fields. Do NOT answer the question. "
        "Do NOT guess. Only extract what is literally in the text.\n\n"
        f"Question: {item['prompt']}\n\n"
        "Return JSON with:\n"
        "  entities: noun-phrases that appear in the question\n"
        "  relations: subject-predicate-object triples grounded in the question\n"
        "  roles: role-filler bindings that capture the question's structure "
        "(e.g. role='referent' with filler='it', role='cause' with filler='turned on the faucet')\n"
    )


def main() -> None:
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    model = models.transformers(MODEL_NAME)
    print(f"  loaded in {time.time()-t0:.1f}s\n")

    gen = generate.json(model, PromptStructure)

    total_items = len(ITEMS)
    failures = 0
    total_entities = grounded_entities = 0
    total_relations = grounded_relations = 0
    total_roles = grounded_roles = 0
    time_sum = 0.0

    for i, item in enumerate(ITEMS):
        print("=" * 78)
        print(f"[{i}] {item['task']}")
        print(f"    prompt: {item['prompt'][:140]}")

        t0 = time.time()
        try:
            s = gen(build_prompt(item), max_tokens=400)
        except Exception as e:
            failures += 1
            if _OutlineGenError is not None and isinstance(e, _OutlineGenError):
                _LOGGER.exception("Outlines structured generation failed [item %s]", i)
            else:
                _LOGGER.exception("Generation failed [item %s]", i)
            print(f"  FAILED: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            continue
        dt = time.time() - t0
        time_sum += dt

        # Grounding check: are entity/role fillers — and relation ends — actually substrings of the prompt?
        text = item["prompt"].lower()
        ent_grounded = sum(1 for e in s.entities if e.text.lower() in text)
        role_grounded = sum(1 for r in s.roles if r.filler.lower() in text)
        rel_grounded = sum(
            1 for r in s.relations
            if r.subject.lower() in text and r.object.lower() in text
        )

        total_entities += len(s.entities)
        grounded_entities += ent_grounded
        total_relations += len(s.relations)
        grounded_relations += rel_grounded
        total_roles += len(s.roles)
        grounded_roles += role_grounded

        print(f"\n  entities ({len(s.entities)}, {ent_grounded} grounded, {dt:.1f}s):")
        for e in s.entities:
            mark = "" if e.text.lower() in text else "  [NOT IN PROMPT]"
            print(f"    {e.kind:<10} {e.text!r}{mark}")

        print(f"\n  relations ({len(s.relations)}, {rel_grounded} grounded in prompt text):")
        for r in s.relations:
            ok_rel = r.subject.lower() in text and r.object.lower() in text
            mark = "" if ok_rel else "  [NOT IN PROMPT]"
            print(f"    ({r.subject!r}) -[{r.predicate}]-> ({r.object!r}){mark}")

        print(f"\n  roles ({len(s.roles)}, {role_grounded} grounded):")
        for r in s.roles:
            mark = "" if r.filler.lower() in text else "  [NOT IN PROMPT]"
            print(f"    {r.role:<18} := {r.filler!r}{mark}")
        print()

    ok = total_items - failures
    avg_dt = time_sum / ok if ok else 0.0
    eg = grounded_entities / total_entities if total_entities else 0.0
    rg = grounded_roles / total_roles if total_roles else 0.0
    rlg = grounded_relations / total_relations if total_relations else 0.0

    print("=" * 78)
    print(
        "SUMMARY:",
        f"items={total_items}, failures={failures}, ok={ok},",
        f"entity grounding={eg:.1%} ({grounded_entities}/{total_entities}),",
        f"role grounding={rg:.1%} ({grounded_roles}/{total_roles}),",
        f"relation grounding={rlg:.1%} ({grounded_relations}/{total_relations}),",
        f"avg time (success)={avg_dt:.2f}s",
    )


if __name__ == "__main__":
    main()
