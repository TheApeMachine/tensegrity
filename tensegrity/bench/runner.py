"""
Evaluation Runner: raw LLM readout vs one integrated Tensegrity agent.

The local LLM still exposes per-choice log-likelihoods, but those logits are
now sensory evidence inside the canonical agent. The final benchmark answer is
the agent's posterior commitment after integrating linguistic evidence,
predictive-coding falsification, causal energy, and persistent memory. The
benchmark label is consumed only afterward as feedback for online learning.
"""

import numpy as np
import time
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from tensegrity.bench.tasks import TaskSample, TASK_REGISTRY, load_task_samples
from tensegrity.torch_device import inference_load_settings

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""
    sample_id: str
    task: str
    gold: int
    n_choices: int
    baseline_pred: int
    grafted_pred: int
    baseline_correct: bool
    grafted_correct: bool
    baseline_scores: List[float]
    grafted_scores: List[float]
    tensegrity_scores: List[float]
    graft_entropy: float
    bias_applied: bool
    bias_magnitude: float
    flip_type: str
    lam: float
    wall_time: float
    emitted_answer: str = ""
    emission_mode: str = ""
    emission_graft_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlipAccounting:
    """Flip analysis for one task."""
    good_flips: int = 0
    bad_flips: int = 0
    preserved: int = 0
    neutral: int = 0
    no_flip: int = 0

    @property
    def total(self):
        return self.good_flips + self.bad_flips + self.preserved + self.neutral

    @property
    def flip_rate(self):
        return (self.good_flips + self.bad_flips) / max(self.total, 1)

    @property
    def good_bad_ratio(self):
        if self.bad_flips == 0:
            return float('inf') if self.good_flips > 0 else 0.0

        return self.good_flips / self.bad_flips

    def to_dict(self):
        return {
            "good_flips": self.good_flips, "bad_flips": self.bad_flips,
            "preserved": self.preserved, "neutral": self.neutral,
            "flip_rate": round(self.flip_rate, 4),
            "good_bad_ratio": round(self.good_bad_ratio, 2) if self.good_bad_ratio != float('inf') else "inf",
        }


@dataclass
class TaskResult:
    """Aggregated result for one task."""
    task: str
    domain: str
    n_samples: int
    lam: float
    baseline_accuracy: float
    grafted_accuracy: float
    delta: float
    baseline_correct: int
    grafted_correct: int
    coverage: float
    cond_acc_biased: float
    cond_acc_unbiased: float
    mean_bias_magnitude: float
    mean_graft_entropy: float
    flips: FlipAccounting
    mean_wall_time: float


@dataclass
class BenchmarkResult:
    """Full benchmark result across all tasks."""
    model_name: str
    mode: str
    lam: float
    tasks: List[TaskResult]
    overall_baseline_accuracy: float
    overall_grafted_accuracy: float
    overall_delta: float
    overall_flips: FlipAccounting
    total_samples: int
    total_wall_time: float
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model_name, "mode": self.mode, "lambda": self.lam,
            "overall": {
                "baseline_accuracy": round(self.overall_baseline_accuracy, 4),
                "grafted_accuracy": round(self.overall_grafted_accuracy, 4),
                "delta": round(self.overall_delta, 4),
                "total_samples": self.total_samples,
                "wall_time_s": round(self.total_wall_time, 1),
                "flips": self.overall_flips.to_dict(),
            },
            "tasks": [{
                "task": t.task, "domain": t.domain, "n": t.n_samples,
                "lambda": t.lam,
                "baseline": round(t.baseline_accuracy, 4),
                "grafted": round(t.grafted_accuracy, 4),
                "delta": round(t.delta, 4),
                "coverage": round(t.coverage, 3),
                "cond_acc_biased": round(t.cond_acc_biased, 4),
                "mean_bias_mag": round(t.mean_bias_magnitude, 4),
                "mean_entropy": round(t.mean_graft_entropy, 3),
                "flips": t.flips.to_dict(),
            } for t in self.tasks],
        }

    def summary_table(self) -> str:
        lines = []
        lines.append(
            f"{'Task':<22} {'N':>5} {'Base':>7} {'Graft':>7} {'Δ':>7}"
            f" {'Cov':>5} {'G/B':>6} {'G→✓':>4} {'G→✗':>4}"
        )
        lines.append("─" * 75)

        for t in sorted(self.tasks, key=lambda x: x.delta, reverse=True):
            sign = "+" if t.delta >= 0 else ""
            gb = t.flips.good_bad_ratio
            gb_str = f"{gb:.1f}" if gb != float('inf') else "∞"
        
            lines.append(
                f"{t.task:<22} {t.n_samples:>5} {t.baseline_accuracy:>6.1%} "
                f"{t.grafted_accuracy:>6.1%} {sign}{t.delta:>6.1%}"
                f" {t.coverage:>4.0%} {gb_str:>6} {t.flips.good_flips:>4} {t.flips.bad_flips:>4}"
            )
        
        lines.append("─" * 75)
        sign = "+" if self.overall_delta >= 0 else ""
        gb = self.overall_flips.good_bad_ratio
        gb_str = f"{gb:.1f}" if gb != float('inf') else "∞"
        
        lines.append(
            f"{'OVERALL':<22} {self.total_samples:>5} {self.overall_baseline_accuracy:>6.1%} "
            f"{self.overall_grafted_accuracy:>6.1%} {sign}{self.overall_delta:>6.1%}"
            f" {'':>5} {gb_str:>6} "
            f"{self.overall_flips.good_flips:>4} {self.overall_flips.bad_flips:>4}"
        )
        
        return "\n".join(lines)


class EvalRunner:
    """
    Runs raw LLM readout vs the persistent integrated agent.

    Modes:
      "local"   — Uses transformers log-likelihoods as linguistic evidence
      "offline" — No LLM; runs the same canonical agent without that channel
    """

    def __init__(
        self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        mode: str = "offline",
        lam: float = 1.0,
        seed: int = 42,
        state_path: Optional[str] = ".tensegrity/agent_state.pkl",
    ):
        self.model_name = model_name
        self.mode = mode
        self.lam = lam
        self.seed = seed
        self.state_path = state_path
        self._model = None
        self._tokenizer = None

    def _init_model(self):
        if self._model is not None or self.mode != "local":
            return
        
        from transformers import AutoTokenizer, AutoModelForCausalLM

        dtype, device_map, move_to = inference_load_settings()
        logger.info(f"Loading model {self.model_name}...")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype, device_map=device_map)
        
        if move_to is not None:
            self._model = self._model.to(move_to)
        
        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is not None:
            generation_config.do_sample = False
            if hasattr(generation_config, "temperature"):
                generation_config.temperature = 1.0
            if hasattr(generation_config, "top_p"):
                generation_config.top_p = 1.0

        self._model.eval()

    def _score_choices_local(self, prompt: str, choices: List[str]) -> List[float]:
        """Score each choice by log P(choice | prompt)."""
        import torch
        scores = []
        
        for choice in choices:
            full_text = f"{prompt} {choice}"
            
            inputs = self._tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=512
            )
            
            if hasattr(self._model, 'device'):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
            
            prompt_ids = self._tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=512,
            )["input_ids"]

            n_prompt = prompt_ids.shape[1]
            n_total = inputs["input_ids"].shape[1]
            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            choice_log_prob = 0.0
            
            for pos in range(n_prompt, n_total):
                token_id = inputs["input_ids"][0, pos].item()
                choice_log_prob += log_probs[pos - 1, token_id].item()
            
            n_choice_tokens = max(n_total - n_prompt, 1)
            scores.append(choice_log_prob / n_choice_tokens)
        return scores

    def _broca_for_canonical(self):
        """Build the Broca transducer used by the canonical benchmark path."""
        if hasattr(self, "_broca"):
            return self._broca
        if self.mode == "local":
            self._init_model()
            from tensegrity.broca.interface import LocalBrocaInterface

            self._broca = LocalBrocaInterface(
                model=self._model,
                tokenizer=self._tokenizer,
                model_name=self.model_name,
            )
        else:
            from tensegrity.broca.interface import DeterministicBrocaInterface

            self._broca = DeterministicBrocaInterface()
        return self._broca

    def _get_tensegrity_scores(
        self,
        sample: TaskSample,
        linguistic_scores: Optional[List[float]],
    ) -> Tuple[List[float], float, Any]:
        """Run the canonical agent pipeline on one item.

        Wires every shipped subsystem: CognitiveController + TensegrityAgent
        (UnifiedField, FreeEnergyEngine, EpistemicMemory, EpisodicMemory,
        AssociativeMemory, log-lik CausalArena), Broca dynamic SCM injection,
        EnergyCausalArena + TopologyMapper for per-choice causal competition,
        NGC top-down falsification.

        One :class:`CanonicalPipeline` instance is reused for all samples and
        tasks. Per-item hypotheses are rewritten in place; field weights,
        episodic memory, Hopfield attractors, epistemic counts, and the agent
        causal arena persist. ``linguistic_scores`` are LLM log-likelihoods and
        are integrated inside the posterior update, not added afterward.
        """
        import numpy as np

        from tensegrity.pipeline.canonical import CanonicalPipeline

        if not hasattr(self, "_canonical"):
            # One CanonicalPipeline instance for all samples on this Runner. Hypothesis texts
            # and per-choice SCMs are updated per sample inside ``reset_for_item`` /
            # ``_soft_reset_in_place``; rebuilding ``CanonicalPipeline`` whenever multi-choice
            # strings changed would recreate ``TensegrityAgent`` / FHRR SBERT loaders and spam
            # "Loading weights" for each benchmark row (see CanonicalPipeline docs).
            self._canonical = CanonicalPipeline(
                hypothesis_labels=None,
                broca=self._broca_for_canonical(),
                use_llm_broca=True,
                enable_hypothesis_generation=True,
                model_name=self.model_name,
                max_iterations=3,
                commit_ratio=2.0,
                falsify_settle_steps=15,
                falsify_update_strength=1.0,
                energy_arena_precision=1.0,
                energy_arena_beta=1.0,
                max_hypotheses=8,
                llm_evidence_weight=self.lam,
                memory_evidence_weight=0.75,
                persistent_state_path=self.state_path,
            )
        else:
            self._canonical.llm_evidence_weight = self.lam

        result = self._canonical.score_multichoice(
            sample,
            linguistic_scores=linguistic_scores,
        )
        n = len(result.scores)

        if n == 0:
            return [], 1.0, result

        # Belief-entropy normalized to [0, 1] for reporting.
        b = np.asarray(result.belief, dtype=np.float64)
        bm = b[b > 0]
        entropy = float(-np.sum(bm * np.log(bm)) / np.log(n)) if n > 1 and len(bm) > 1 else 0.0

        return list(result.scores), entropy, result

    def _choice_token_sequences(
        self,
        choices: List[str],
    ) -> Tuple[Dict[int, List[List[int]]], Dict[Tuple[int, ...], int]]:
        """Tokenize exact answer choices for constrained local emission."""
        sequences: Dict[int, List[List[int]]] = {}
        seq_to_choice: Dict[Tuple[int, ...], int] = {}
        for i, choice in enumerate(choices):
            variants = [choice, f" {choice}", f"\n{choice}"]
            seen: Set[Tuple[int, ...]] = set()
            for variant in variants:
                ids = self._tokenizer.encode(variant, add_special_tokens=False)
                seq = tuple(int(t) for t in ids if int(t) >= 0)
                if not seq or seq in seen:
                    continue
                seen.add(seq)
                sequences.setdefault(i, []).append(list(seq))
                seq_to_choice[seq] = i
        return sequences, seq_to_choice

    @staticmethod
    def _choice_token_grounding(choice_sequences: Dict[int, List[List[int]]]) -> Dict[str, Set[int]]:
        """Convert tokenized choices into hypothesis token sets for the live graft."""
        grounding: Dict[str, Set[int]] = {}
        for i, seqs in choice_sequences.items():
            toks: Set[int] = set()
            for seq in seqs:
                toks.update(int(t) for t in seq)
            grounding[f"H{i}"] = toks
        return grounding

    @staticmethod
    def _belief_mapping(commit: Any, n_choices: int) -> Dict[str, float]:
        vals = np.asarray(list(getattr(commit, "belief", []))[:n_choices], dtype=np.float64)
        if vals.shape[0] != n_choices or not np.all(np.isfinite(vals)):
            vals = np.full(n_choices, 1.0 / max(n_choices, 1), dtype=np.float64)
        vals = np.maximum(vals, 0.0)
        total = float(vals.sum())
        if total <= 0.0:
            vals = np.full(n_choices, 1.0 / max(n_choices, 1), dtype=np.float64)
        else:
            vals = vals / total
        return {f"H{i}": float(vals[i]) for i in range(n_choices)}

    @staticmethod
    def _build_prefix_allowed_fn(
        prompt_len: int,
        choice_sequences: Dict[int, List[List[int]]],
        eos_token_id: Optional[int],
    ):
        trie: Dict[int, Any] = {}
        terminal = "_terminal"
        for seqs in choice_sequences.values():
            for seq in seqs:
                node = trie
                for token_id in seq:
                    node = node.setdefault(int(token_id), {})
                node[terminal] = True

        def allowed(_batch_id: int, input_ids):
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            generated = ids[prompt_len:]
            if eos_token_id is not None and generated and generated[-1] == eos_token_id:
                return [int(eos_token_id)]
            node = trie
            for token_id in generated:
                token_id = int(token_id)
                if token_id not in node:
                    return [int(eos_token_id)] if eos_token_id is not None else []
                node = node[token_id]
            next_tokens = [int(k) for k in node.keys() if k != terminal]
            if node.get(terminal) and eos_token_id is not None:
                next_tokens.append(int(eos_token_id))
            return next_tokens or ([int(eos_token_id)] if eos_token_id is not None else [])

        return allowed

    @staticmethod
    def _parse_emitted_choice(
        generated_ids: List[int],
        seq_to_choice: Dict[Tuple[int, ...], int],
        eos_token_id: Optional[int],
    ) -> Optional[int]:
        seq = [int(t) for t in generated_ids]
        if eos_token_id is not None:
            seq = [t for t in seq if t != int(eos_token_id)]
        while seq and seq[-1] == 0:
            seq.pop()
        return seq_to_choice.get(tuple(seq))

    def _emit_answer(
        self,
        sample: TaskSample,
        commit: Any,
    ) -> Tuple[int, str, str, Dict[str, Any]]:
        """Emit the final local answer through live logit grafting."""
        n = len(sample.choices)
        committed_idx = int(getattr(commit, "committed_idx", -1))
        if self.mode != "local" or n == 0:
            text = sample.choices[committed_idx] if 0 <= committed_idx < n else ""
            return committed_idx, text, "posterior_verbalizer", {}

        self._init_model()
        import torch
        from transformers import LogitsProcessorList
        from tensegrity.graft.logit_bias import TensegrityLogitsProcessor

        choice_sequences, _ = self._choice_token_sequences(sample.choices)
        if not choice_sequences:
            raise ValueError("No tokenized answer choices available for local emission")
        if 0 <= committed_idx < n and committed_idx in choice_sequences:
            emission_sequences = {committed_idx: choice_sequences[committed_idx]}
        else:
            emission_sequences = choice_sequences
        seq_to_choice = {
            tuple(seq): i
            for i, seqs in emission_sequences.items()
            for seq in seqs
        }

        hypothesis_tokens = self._choice_token_grounding(choice_sequences)
        beliefs = self._belief_mapping(commit, n)
        model_config = getattr(self._model, "config", None)
        model_vocab_size = getattr(model_config, "vocab_size", None)
        vocab_size_attr = getattr(self._tokenizer, "vocab_size", None)
        vocab_size = int(
            model_vocab_size
            if model_vocab_size is not None
            else vocab_size_attr if vocab_size_attr is not None
            else len(self._tokenizer)
        )
        processor = TensegrityLogitsProcessor(
            hypothesis_tokens=hypothesis_tokens,
            belief_fn=lambda: beliefs,
            vocab_size=vocab_size,
            scale=max(0.0, float(self.lam)) * 2.5,
            suppress_threshold=0.0,
            entropy_gate=1.01,
            min_confidence=0.0,
            async_beliefs=False,
        )

        choice_lines = "\n".join(f"- {choice}" for choice in sample.choices)
        prompt = (
            "Return exactly one answer choice and no explanation.\n\n"
            f"Question:\n{sample.prompt}\n\n"
            f"Choices:\n{choice_lines}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            attention_mask = None
            if hasattr(self._tokenizer, "apply_chat_template"):
                encoded = self._tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                if isinstance(encoded, torch.Tensor):
                    input_ids = encoded
                elif hasattr(encoded, "input_ids"):
                    input_ids = encoded.input_ids
                    attention_mask = getattr(encoded, "attention_mask", None)
                else:
                    input_ids = encoded["input_ids"]
                    attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
            else:
                encoded = self._tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)

            if hasattr(self._model, "device"):
                input_ids = input_ids.to(self._model.device)
                attention_mask = attention_mask.to(self._model.device)

            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
            pad_token_id = getattr(self._tokenizer, "pad_token_id", None) or eos_token_id
            max_choice_len = max(len(seq) for seqs in emission_sequences.values() for seq in seqs)
            prefix_allowed = self._build_prefix_allowed_fn(
                prompt_len=int(input_ids.shape[1]),
                choice_sequences=emission_sequences,
                eos_token_id=eos_token_id,
            )
            outputs = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                logits_processor=LogitsProcessorList([processor]),
                prefix_allowed_tokens_fn=prefix_allowed,
                max_new_tokens=max_choice_len + 1,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            new_tokens = [int(t) for t in outputs[0][input_ids.shape[1]:].tolist()]
            emitted_idx = self._parse_emitted_choice(new_tokens, seq_to_choice, eos_token_id)
            emitted_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            graft_state = {
                "step": processor.state.step,
                "bias_emitted": processor.state.bias_emitted,
                "belief_entropy": processor.state.belief_entropy,
                "convergence_met": processor.state.convergence_met,
                "max_bias_magnitude": processor.state.max_bias_magnitude,
                "boosted_tokens": processor.state.boosted_tokens,
                "suppressed_tokens": processor.state.suppressed_tokens,
                "beliefs": beliefs,
            }
            if emitted_idx is None:
                emitted_idx = committed_idx
                return emitted_idx, emitted_text, "logit_grafted_unparsed_commit", graft_state
            return int(emitted_idx), emitted_text, "logit_grafted_verbalizer", graft_state
        finally:
            processor.close()

    def evaluate_sample(self, sample: TaskSample) -> SampleResult:
        """Evaluate one sample through the persistent integrated agent."""
        t0 = time.time()
        n = len(sample.choices)

        if self.mode == "local":
            self._init_model()
            baseline_scores = self._score_choices_local(sample.prompt, sample.choices)
        else:
            seed_bytes = hashlib.sha256(sample.id.encode("utf-8")).digest()
            sample_seed = int.from_bytes(seed_bytes[:8], "big", signed=False) % (2**31)
            rng = np.random.RandomState(sample_seed)
            baseline_scores = rng.randn(n).tolist()

        tensegrity_scores, entropy, commit = self._get_tensegrity_scores(
            sample,
            linguistic_scores=baseline_scores if self.mode == "local" else None,
        )
        scores_arr = np.array(tensegrity_scores, dtype=np.float64)
        score_spread = float(scores_arr.max() - scores_arr.min()) if len(scores_arr) > 0 else 0.0
        bias_magnitude = score_spread
        bias_applied = True
        grafted_scores = list(tensegrity_scores)
        baseline_pred = int(np.argmax(baseline_scores))
        try:
            grafted_pred, emitted_answer, emission_mode, emission_graft_state = self._emit_answer(
                sample, commit
            )
        except Exception as e:
            logger.warning("Integrated answer emission failed; using committed posterior: %s", e)
            grafted_pred = int(commit.committed_idx)
            emitted_answer = sample.choices[grafted_pred] if 0 <= grafted_pred < n else ""
            emission_mode = "emission_error_commit"
            emission_graft_state = {"error": str(e)}
        baseline_correct = (baseline_pred == sample.gold)
        grafted_correct = (grafted_pred == sample.gold)

        if baseline_pred == grafted_pred:
            flip_type = "preserved" if baseline_correct else "neutral"
        elif not baseline_correct and grafted_correct:
            flip_type = "good_flip"
        elif baseline_correct and not grafted_correct:
            flip_type = "bad_flip"
        else:
            flip_type = "neutral"

        sample_result = SampleResult(
            sample_id=sample.id, task=sample.metadata.get("task", ""),
            gold=sample.gold, n_choices=n,
            baseline_pred=baseline_pred, grafted_pred=grafted_pred,
            baseline_correct=baseline_correct, grafted_correct=grafted_correct,
            baseline_scores=baseline_scores, grafted_scores=grafted_scores,
            tensegrity_scores=tensegrity_scores, graft_entropy=entropy,
            bias_applied=bias_applied, bias_magnitude=bias_magnitude,
            flip_type=flip_type, lam=self.lam, wall_time=time.time() - t0,
            emitted_answer=emitted_answer, emission_mode=emission_mode,
            emission_graft_state=emission_graft_state,
        )
        if hasattr(self, "_canonical"):
            self._canonical.learn_from_feedback(sample, grafted_pred)
        return sample_result

    def evaluate_task(
        self, task_name: str, max_samples: Optional[int] = None, verbose: bool = False,
    ) -> TaskResult:
        config = TASK_REGISTRY[task_name]
        samples = load_task_samples(task_name, max_samples)

        if verbose:
            print(f"  [{task_name}] Loaded {len(samples)} samples")

        results = []
        for i, sample in enumerate(samples):
            results.append(self.evaluate_sample(sample))
            if verbose and (i + 1) % 100 == 0:
                acc_b = sum(1 for x in results if x.baseline_correct) / len(results)
                acc_g = sum(1 for x in results if x.grafted_correct) / len(results)
                print(f"    {i+1}/{len(samples)}: base={acc_b:.1%} graft={acc_g:.1%}")

        n = len(results)
        if n == 0:
            return TaskResult(
                task=task_name, domain=config.domain, n_samples=0, lam=self.lam,
                baseline_accuracy=0, grafted_accuracy=0, delta=0,
                baseline_correct=0, grafted_correct=0,
                coverage=0, cond_acc_biased=0, cond_acc_unbiased=0,
                mean_bias_magnitude=0, mean_graft_entropy=0,
                flips=FlipAccounting(), mean_wall_time=0,
            )

        bl_correct = sum(1 for r in results if r.baseline_correct)
        gr_correct = sum(1 for r in results if r.grafted_correct)

        flips = FlipAccounting()
        
        for r in results:
            if r.flip_type == "good_flip": flips.good_flips += 1
            elif r.flip_type == "bad_flip": flips.bad_flips += 1
            elif r.flip_type == "preserved": flips.preserved += 1
            elif r.flip_type == "neutral": flips.neutral += 1

        biased = [r for r in results if r.bias_applied]
        coverage = len(biased) / n
        cond_acc_biased = (sum(1 for r in biased if r.grafted_correct) / len(biased)) if biased else 0.0
        unbiased = [r for r in results if not r.bias_applied]
        cond_acc_unbiased = (sum(1 for r in unbiased if r.grafted_correct) / len(unbiased)) if unbiased else 0.0

        return TaskResult(
            task=task_name, domain=config.domain, n_samples=n, lam=self.lam,
            baseline_accuracy=bl_correct / n, grafted_accuracy=gr_correct / n,
            delta=(gr_correct - bl_correct) / n,
            baseline_correct=bl_correct, grafted_correct=gr_correct,
            coverage=coverage, cond_acc_biased=cond_acc_biased,
            cond_acc_unbiased=cond_acc_unbiased,
            mean_bias_magnitude=np.mean([r.bias_magnitude for r in results]),
            mean_graft_entropy=np.mean([r.graft_entropy for r in results]),
            flips=flips, mean_wall_time=np.mean([r.wall_time for r in results]),
        )

    def run_benchmark(
        self, tasks=None, max_samples_per_task=None, verbose=True,
    ) -> BenchmarkResult:
        if tasks is None:
            tasks = list(TASK_REGISTRY.keys())
        
        if verbose:
            print(f"\n{'█' * 60}")
            print(f"  TENSEGRITY BENCHMARK")
            print(f"  Model:  {self.model_name}")
            print(f"  Mode:   {self.mode}")
            print(f"  λ:      {self.lam}")
            print(f"  Tasks:  {len(tasks)}")
            cap_str = str(max_samples_per_task) if max_samples_per_task else "all"
            print(f"  N/task: {cap_str}")
            print(f"{'█' * 60}")

        t_start = time.time()
        task_results = []
        
        for task_name in tasks:
            if verbose:
                print(f"\n  ▸ {task_name}: {TASK_REGISTRY[task_name].description}")
            try:
                tr = self.evaluate_task(task_name, max_samples_per_task, verbose)
                task_results.append(tr)
                
                if verbose:
                    sign = "+" if tr.delta >= 0 else ""
                    gb = tr.flips.good_bad_ratio
                    gb_str = f"{gb:.1f}" if gb != float('inf') else "∞"
                    
                    print(f"    base={tr.baseline_accuracy:.1%}  graft={tr.grafted_accuracy:.1%}  "
                          f"Δ={sign}{tr.delta:.1%}  cov={tr.coverage:.0%}  "
                          f"flips={tr.flips.good_flips}↑/{tr.flips.bad_flips}↓  G/B={gb_str}")
            except Exception as e:
                logger.error(f"Task {task_name} failed: {e}")
                
                if verbose:
                    print(f"    ✗ FAILED: {e}")
                    import traceback; traceback.print_exc()

        total_time = time.time() - t_start
        total_bl = sum(t.baseline_correct for t in task_results)
        total_gr = sum(t.grafted_correct for t in task_results)
        total_n = sum(t.n_samples for t in task_results)

        overall_flips = FlipAccounting()
        
        for t in task_results:
            overall_flips.good_flips += t.flips.good_flips
            overall_flips.bad_flips += t.flips.bad_flips
            overall_flips.preserved += t.flips.preserved
            overall_flips.neutral += t.flips.neutral

        result = BenchmarkResult(
            model_name=self.model_name, mode=self.mode, lam=self.lam,
            tasks=task_results,
            overall_baseline_accuracy=total_bl / max(total_n, 1),
            overall_grafted_accuracy=total_gr / max(total_n, 1),
            overall_delta=(total_gr - total_bl) / max(total_n, 1),
            overall_flips=overall_flips, total_samples=total_n,
            total_wall_time=total_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if verbose:
            print(f"\n{'═' * 75}")
            print(result.summary_table())
            print(f"\n  λ={self.lam}  Time={total_time:.1f}s")
            print(
                f"  Total flips: {overall_flips.good_flips}↑ good, "
                f"{overall_flips.bad_flips}↓ bad, "
                f"{overall_flips.preserved} preserved, "
                f"{overall_flips.neutral} neutral"
            )
            print(f"{'═' * 75}")

        return result

    def sweep_lambda(
        self, tasks=None, lambdas=None, max_samples_per_task=None, verbose=True,
    ) -> List[BenchmarkResult]:
        if lambdas is None:
            lambdas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

        if verbose:
            print(f"\n{'█' * 60}")
            print(f"  λ SWEEP: {lambdas}")
            print(f"{'█' * 60}")

        results = []

        for lam_val in lambdas:
            self.lam = lam_val
            result = self.run_benchmark(tasks, max_samples_per_task, verbose=False)
            results.append(result)

            if verbose:
                sign = "+" if result.overall_delta >= 0 else ""
                gb = result.overall_flips.good_bad_ratio
                gb_str = f"{gb:.1f}" if gb != float('inf') else "∞"
                
                print(
                    f"  λ={lam_val:<5}  base={result.overall_baseline_accuracy:.1%}  "
                    f"graft={result.overall_grafted_accuracy:.1%}  "
                    f"Δ={sign}{result.overall_delta:.1%}  G/B={gb_str}  "
                    f"({result.overall_flips.good_flips}↑/{result.overall_flips.bad_flips}↓)"
                )

        if verbose:
            best = max(results, key=lambda r: r.overall_delta)
            print(f"\n  Best λ = {best.lam} → Δ = {best.overall_delta:+.1%}")

        return results

    def save_results(self, result: BenchmarkResult, path: str):
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Results saved to {path}")
