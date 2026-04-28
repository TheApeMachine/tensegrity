"""
Task Registry: Adapters that normalize HuggingFace datasets into a common format.

Every benchmark task, regardless of its native schema, gets normalized into:

    TaskSample(
        id:        str              -- unique identifier
        prompt:    str              -- the question / context to present
        choices:   list[str]        -- the candidate answers
        gold:      int              -- index of the correct answer in choices
        keywords:  dict[str, list]  -- per-choice keywords for vocabulary grounding
        metadata:  dict             -- task name, difficulty, domain, etc.
    )

The harness doesn't care where the data came from. It only sees TaskSamples.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
import logging
import random


logger = logging.getLogger(__name__)


@dataclass
class TaskSample:
    """A single evaluation sample normalized from any HF dataset."""
    id: str
    prompt: str
    choices: List[str]
    gold: int
    keywords: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a benchmark task."""
    name: str
    hf_id: str
    hf_config: Optional[str]
    split: str
    adapter_fn: Callable  # (row) -> TaskSample
    description: str = ""
    domain: str = "general"
    n_choices: int = 4
    max_samples: Optional[int] = None  # Cap for fast runs


# ═══════════════════════════════════════════════════════════════
# ADAPTER FUNCTIONS: dataset row → TaskSample
# ═══════════════════════════════════════════════════════════════

def _adapt_arc(row: dict, task_name: str = "arc_challenge") -> TaskSample:
    """ARC Challenge / Easy: scientific multi-choice reasoning."""
    question = row["question"]
    choices = row["choices"]["text"]
    labels = row["choices"]["label"]
    answer = row["answerKey"]

    # Handle numeric answer keys ("1","2","3","4") and alpha ("A","B","C","D")
    if answer.isdigit():
        gold = int(answer) - 1
    else:
        gold = ord(answer) - ord("A")
    
    gold = min(gold, len(choices) - 1)

    # Extract keywords from each choice for vocabulary grounding
    keywords = {f"choice_{i}": c.lower().split()[:6] for i, c in enumerate(choices)}

    return TaskSample(
        id=row.get("id", ""),
        prompt=f"Question: {question}\nAnswer:",
        choices=choices,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "science"},
    )


def _adapt_hellaswag(row: dict, task_name: str = "hellaswag") -> TaskSample:
    """HellaSwag: commonsense continuation."""
    ctx = row["ctx"]
    endings = row["endings"]
    gold = int(row["label"])
    keywords = {f"choice_{i}": e.lower().split()[:6] for i, e in enumerate(endings)}

    return TaskSample(
        id=str(row.get("ind", "")),
        prompt=f"{ctx}",
        choices=endings,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "commonsense"},
    )


def _adapt_winogrande(row: dict, task_name: str = "winogrande") -> TaskSample:
    """WinoGrande: coreference / relational reasoning."""
    sentence = row["sentence"]
    choices = [row["option1"], row["option2"]]
    gold = int(row["answer"]) - 1  # 1-indexed

    filled = [sentence.replace("_", c) for c in choices]
    keywords = {f"choice_{i}": c.lower().split() for i, c in enumerate(choices)}

    return TaskSample(
        id="",
        prompt=f"Which makes more sense?\n",
        choices=filled,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "reasoning"},
    )


def _adapt_boolq(row: dict, task_name: str = "boolq") -> TaskSample:
    """BoolQ: yes/no reading comprehension."""
    question = row["question"]
    passage = row["passage"]
    gold = int(row["answer"])  # False→0, True→1

    return TaskSample(
        id="",
        prompt=f"{passage}\n\nQuestion: {question}\nAnswer (yes or no):",
        choices=["no", "yes"],
        gold=gold,
        keywords={"choice_0": ["no", "false", "not"], "choice_1": ["yes", "true"]},
        metadata={"task": task_name, "domain": "comprehension"},
    )


def _adapt_copa(row: dict, task_name: str = "copa") -> TaskSample:
    """COPA: causal reasoning (cause/effect)."""
    premise = row["premise"]
    q_type = row["question"]
    choices = [row["choice1"], row["choice2"]]
    gold = int(row["label"])

    if q_type == "cause":
        connector = "What was the CAUSE of this?"
    else:
        connector = "What happened as a RESULT?"

    keywords = {f"choice_{i}": c.lower().split()[:6] for i, c in enumerate(choices)}

    return TaskSample(
        id=str(row.get("id", "")),
        prompt=f"{premise}\n{connector}",
        choices=choices,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "causal", "q_type": q_type},
    )


def _adapt_logiqa(row: dict, task_name: str = "logiqa") -> TaskSample:
    """LogiQA: formal logical deduction."""
    context = row["context"]
    question = row["query"]
    options = row["options"]
    gold = int(row["correct_option"])

    keywords = {f"choice_{i}": o.lower().split()[:6] for i, o in enumerate(options)}

    return TaskSample(
        id="",
        prompt=f"{context}\n\nQuestion: {question}",
        choices=options,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "logic"},
    )


def _adapt_truthfulqa(row: dict, task_name: str = "truthfulqa") -> TaskSample:
    """TruthfulQA: hallucination resistance (MC1: single correct)."""
    question = row["question"]
    mc1 = row["mc1_targets"]
    choices = mc1["choices"]
    labels = mc1["labels"]
    gold = labels.index(1)

    keywords = {f"choice_{i}": c.lower().split()[:6] for i, c in enumerate(choices)}

    return TaskSample(
        id="",
        prompt=f"Question: {question}\nAnswer:",
        choices=choices,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "truthfulness"},
    )


def _adapt_mmlu(row: dict, task_name: str = "mmlu") -> TaskSample:
    """MMLU: multi-domain knowledge + reasoning."""
    question = row["question"]
    choices = row["choices"]
    gold = int(row["answer"])

    keywords = {f"choice_{i}": c.lower().split()[:6] for i, c in enumerate(choices)}
    subject = row.get("subject", "unknown")

    return TaskSample(
        id="",
        prompt=f"Question: {question}\nAnswer:",
        choices=choices,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "knowledge", "subject": subject},
    )


def _adapt_bigbench_mc(row: dict, task_name: str = "bigbench") -> TaskSample:
    """BigBench multiple-choice tasks (logical_deduction, causal_judgment, etc.)."""
    prompt = row["inputs"]
    choices = row["multiple_choice_targets"]
    scores = row["multiple_choice_scores"]
    gold = scores.index(1) if 1 in scores else 0

    keywords = {f"choice_{i}": c.lower().split()[:6] for i, c in enumerate(choices)}

    return TaskSample(
        id=str(row.get("idx", "")),
        prompt=prompt,
        choices=choices,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "logic"},
    )


def _adapt_sciq(row: dict, task_name: str = "sciq") -> TaskSample:
    """SciQ: scientific reasoning with support passage."""
    question = row["question"]
    support = row["support"]
    correct = row["correct_answer"]
    choices = [row["distractor1"], row["distractor2"], row["distractor3"], correct]
    random.shuffle(choices)
    gold = choices.index(correct)

    keywords = {f"choice_{i}": c.lower().split()[:6] for i, c in enumerate(choices)}

    return TaskSample(
        id="",
        prompt=f"{support}\n\nQuestion: {question}\nAnswer:",
        choices=choices,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "science"},
    )


def _adapt_strategy_qa(row: dict, task_name: str = "strategy_qa") -> TaskSample:
    """StrategyQA: multi-hop implicit reasoning."""
    question = row["question"]
    gold = int(row["answer"])  # False→0, True→1

    return TaskSample(
        id=row.get("qid", ""),
        prompt=f"Answer yes or no: {question}",
        choices=["no", "yes"],
        gold=gold,
        keywords={"choice_0": ["no", "false"], "choice_1": ["yes", "true"]},
        metadata={
            "task": task_name, "domain": "multi_hop",
            "decomposition": row.get("decomposition", []),
        },
    )


def _adapt_reclor(row: dict, task_name: str = "reclor") -> TaskSample:
    """ReClor: LSAT/GRE logical reasoning."""
    context = row["context"]
    question = row["question"]
    answers = row["answers"]
    gold = int(row["label"])

    keywords = {f"choice_{i}": a.lower().split()[:6] for i, a in enumerate(answers)}

    return TaskSample(
        id=row.get("id_string", ""),
        prompt=f"{context}\n\nQuestion: {question}",
        choices=answers,
        gold=gold,
        keywords=keywords,
        metadata={"task": task_name, "domain": "formal_logic"},
    )


# ═══════════════════════════════════════════════════════════════
# TASK REGISTRY
# ═══════════════════════════════════════════════════════════════

TASK_REGISTRY: Dict[str, TaskConfig] = {
    "arc_challenge": TaskConfig(
        name="arc_challenge", hf_id="allenai/ai2_arc", hf_config="ARC-Challenge",
        split="test", adapter_fn=_adapt_arc, n_choices=4,
        description="Scientific reasoning (multi-step deduction)",
        domain="science",
    ),
    "arc_easy": TaskConfig(
        name="arc_easy", hf_id="allenai/ai2_arc", hf_config="ARC-Easy",
        split="test", adapter_fn=_adapt_arc, n_choices=4,
        description="Scientific reasoning (easier)",
        domain="science",
    ),
    "hellaswag": TaskConfig(
        name="hellaswag", hf_id="Rowan/hellaswag", hf_config="default",
        split="validation", adapter_fn=_adapt_hellaswag, n_choices=4,
        description="Commonsense continuation prediction",
        domain="commonsense",
    ),
    "winogrande": TaskConfig(
        name="winogrande", hf_id="allenai/winogrande", hf_config="winogrande_xl",
        split="validation", adapter_fn=_adapt_winogrande, n_choices=2,
        description="Coreference / relational reasoning",
        domain="reasoning",
    ),
    "boolq": TaskConfig(
        name="boolq", hf_id="google/boolq", hf_config="default",
        split="validation", adapter_fn=_adapt_boolq, n_choices=2,
        description="Yes/no reading comprehension",
        domain="comprehension",
    ),
    "copa": TaskConfig(
        name="copa", hf_id="pkavumba/balanced-copa", hf_config="default",
        split="test", adapter_fn=_adapt_copa, n_choices=2,
        description="Causal reasoning (cause/effect)",
        domain="causal",
    ),
    "logical_deduction": TaskConfig(
        name="logical_deduction", hf_id="tasksource/bigbench", hf_config="logical_deduction",
        split="validation", adapter_fn=_adapt_bigbench_mc, n_choices=5,
        description="BigBench — Logical deduction (ordering puzzles)",
        domain="logic",
    ),
    "causal_judgment": TaskConfig(
        name="causal_judgment", hf_id="tasksource/bigbench", hf_config="causal_judgment",
        split="validation", adapter_fn=_adapt_bigbench_mc, n_choices=2,
        description="BigBench — Causal judgment (counterfactual)",
        domain="causal",
    ),
    "truthfulqa": TaskConfig(
        name="truthfulqa", hf_id="truthfulqa/truthful_qa", hf_config="multiple_choice",
        split="validation", adapter_fn=_adapt_truthfulqa, n_choices=4,
        description="Hallucination / belief resistance",
        domain="truthfulness",
    ),
    "mmlu_formal_logic": TaskConfig(
        name="mmlu_formal_logic", hf_id="cais/mmlu", hf_config="formal_logic",
        split="test", adapter_fn=_adapt_mmlu, n_choices=4,
        description="MMLU — Formal Logic",
        domain="logic",
    ),
    "mmlu_philosophy": TaskConfig(
        name="mmlu_philosophy", hf_id="cais/mmlu", hf_config="philosophy",
        split="test", adapter_fn=_adapt_mmlu, n_choices=4,
        description="MMLU — Philosophy",
        domain="reasoning",
    ),
    "sciq": TaskConfig(
        name="sciq", hf_id="allenai/sciq", hf_config="default",
        split="test", adapter_fn=_adapt_sciq, n_choices=4,
        description="Scientific reasoning with support passage",
        domain="science",
    ),
    "strategy_qa": TaskConfig(
        name="strategy_qa", hf_id="tasksource/bigbench", hf_config="strategyqa",
        split="validation", adapter_fn=_adapt_bigbench_mc, n_choices=2,
        description="Multi-hop implicit reasoning (StrategyQA via BigBench)",
        domain="multi_hop",
    ),
    "reclor": TaskConfig(
        name="reclor", hf_id="metaeval/reclor", hf_config="default",
        split="validation", adapter_fn=_adapt_reclor, n_choices=4,
        description="LSAT/GRE logical reasoning",
        domain="formal_logic",
    ),
}


def list_tasks() -> List[str]:
    """List all registered task names."""
    return list(TASK_REGISTRY.keys())


def get_task(name: str) -> TaskConfig:
    """Get a task config by name."""
    if name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise KeyError(f"Unknown task '{name}'. Available: {available}")

    return TASK_REGISTRY[name]


def load_task_samples(name: str, max_samples: Optional[int] = None) -> List[TaskSample]:
    """
    Load a task's dataset and convert all rows to TaskSamples.

    Args:
        name: Task name from TASK_REGISTRY
        max_samples: Cap the number of samples (for fast dev runs)

    Returns:
        List of normalized TaskSample instances
    """
    from datasets import load_dataset

    config = get_task(name)
    cap = max_samples or config.max_samples

    kwargs = {"split": config.split}

    if config.hf_config and config.hf_config != "default":
        ds = load_dataset(config.hf_id, config.hf_config, **kwargs)
    else:
        ds = load_dataset(config.hf_id, **kwargs)

    samples = []

    for i, row in enumerate(ds):
        if cap and i >= cap:
            break

        try:
            sample = config.adapter_fn(row, config.name)

            if not sample.id:
                sample.id = f"{config.name}_{i}"

            samples.append(sample)
        except Exception as e:
            logger.exception("Error adapting task %s at row %s: %s", name, i, e)
            continue  # Skip malformed rows

    return samples
