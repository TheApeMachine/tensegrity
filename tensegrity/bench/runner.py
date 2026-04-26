"""
Evaluation Runner: Baseline (plain LLM) vs Grafted (Tensegrity+LLM).

Two evaluation modes per sample:

  BASELINE: Model scores each choice via log-probability.
            P(choice | prompt) computed from raw logits.
            Prediction = argmax over choices.

  GRAFTED:  score(choice) = llm_logprob(choice) + λ * tensegrity_score(choice)
            Where λ controls the graft weight. λ=0 recovers baseline.

The ONLY difference is the additive Tensegrity term.
This is a controlled A/B comparison.

Metrics per task:
  - raw_acc:         baseline accuracy
  - grafted_acc:     grafted accuracy
  - delta:           grafted - baseline
  - coverage:        fraction of samples where graft posteriors are non-uniform
  - cond_acc_biased: accuracy on the subset where graft was non-uniform
  - mean_bias_mag:   mean max absolute Tensegrity score deviation from uniform
  - flip_rate:       fraction of samples where baseline_pred != grafted_pred
  - good_flips:      LLM wrong → graft right
  - bad_flips:       LLM right → graft wrong
  - preserved:       LLM right → graft right
  - neutral:         LLM wrong → graft wrong
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from tensegrity.bench.tasks import TaskSample, TaskConfig, TASK_REGISTRY, load_task_samples

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
    tensegrity_scores: List[float]  # Raw Tensegrity posteriors (pre-blend)
    graft_entropy: float            # Normalized entropy of Tensegrity posteriors
    bias_applied: bool              # Did Tensegrity posteriors differ from uniform?
    bias_magnitude: float           # Max absolute deviation from uniform
    flip_type: str                  # "good_flip", "bad_flip", "preserved", "neutral", "no_flip"
    lam: float                      # λ used for this evaluation
    wall_time: float


@dataclass
class FlipAccounting:
    """Flip analysis for one task."""
    good_flips: int = 0     # LLM wrong → graft right
    bad_flips: int = 0      # LLM right → graft wrong
    preserved: int = 0      # LLM right → graft right
    neutral: int = 0        # LLM wrong → graft wrong (no change)
    no_flip: int = 0        # Same prediction (subset of preserved + neutral)

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
            "good_flips": self.good_flips,
            "bad_flips": self.bad_flips,
            "preserved": self.preserved,
            "neutral": self.neutral,
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
    # Core accuracy
    baseline_accuracy: float
    grafted_accuracy: float
    delta: float
    baseline_correct: int
    grafted_correct: int
    # Graft diagnostics
    coverage: float             # Fraction where bias_applied=True
    cond_acc_biased: float      # Accuracy only on samples where bias was applied
    cond_acc_unbiased: float    # Accuracy only on samples where bias was NOT applied
    mean_bias_magnitude: float
    mean_graft_entropy: float
    # Flips
    flips: FlipAccounting
    # Timing
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
            "model": self.model_name,
            "mode": self.mode,
            "lambda": self.lam,
            "overall": {
                "baseline_accuracy": round(self.overall_baseline_accuracy, 4),
                "grafted_accuracy": round(self.overall_grafted_accuracy, 4),
                "delta": round(self.overall_delta, 4),
                "total_samples": self.total_samples,
                "wall_time_s": round(self.total_wall_time, 1),
                "flips": self.overall_flips.to_dict(),
            },
            "tasks": [
                {
                    "task": t.task,
                    "domain": t.domain,
                    "n": t.n_samples,
                    "lambda": t.lam,
                    "baseline": round(t.baseline_accuracy, 4),
                    "grafted": round(t.grafted_accuracy, 4),
                    "delta": round(t.delta, 4),
                    "coverage": round(t.coverage, 3),
                    "cond_acc_biased": round(t.cond_acc_biased, 4),
                    "mean_bias_mag": round(t.mean_bias_magnitude, 4),
                    "mean_entropy": round(t.mean_graft_entropy, 3),
                    "flips": t.flips.to_dict(),
                }
                for t in self.tasks
            ],
        }

    def summary_table(self) -> str:
        lines = []
        lines.append(f"{'Task':<22} {'N':>5} {'Base':>7} {'Graft':>7} {'Δ':>7}"
                      f" {'Cov':>5} {'G/B':>6} {'G→✓':>4} {'G→✗':>4}")
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
    Runs baseline vs grafted evaluation on any set of tasks.

    Modes:
      "local"   — Uses transformers model with LogitsProcessor
      "offline"  — No LLM; baseline = random, grafted = Tensegrity posteriors
                  (tests the cognitive layer in isolation)

    λ parameter:
      score(choice) = baseline_score(choice) + λ * tensegrity_score(choice)
      λ=0 → pure baseline. λ>0 → graft contributes. Sweep to find optimal.
    """

    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
                 mode: str = "offline",
                 lam: float = 1.0,
                 seed: int = 42):
        """
        Args:
            model_name: HF model ID for local mode
            mode: "offline" or "local"
            lam: λ — graft weight. score = baseline + λ * tensegrity
            seed: Random seed
        """
        self.model_name = model_name
        self.mode = mode
        self.lam = lam
        self.seed = seed

        self._model = None
        self._tokenizer = None

    def _init_model(self):
        if self._model is not None or self.mode != "local":
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        logger.info(f"Loading model {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()

    # ─── SCORING ────────────────────────────────────────────

    def _score_choices_local(self, prompt: str, choices: List[str]) -> List[float]:
        """Score each choice by log P(choice | prompt)."""
        import torch
        scores = []
        for choice in choices:
            full_text = f"{prompt} {choice}"
            inputs = self._tokenizer(full_text, return_tensors="pt",
                                     truncation=True, max_length=512)
            if hasattr(self._model, 'device'):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
            prompt_ids = self._tokenizer(prompt, return_tensors="pt",
                                         truncation=True, max_length=512)["input_ids"]
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

    def _get_tensegrity_scores(self, sample: TaskSample) -> Tuple[List[float], float]:
        """
        Run Tensegrity cognitive layer on a sample.
        Returns (posteriors_list, normalized_entropy).
        """
        from tensegrity.broca.controller import CognitiveController

        n = len(sample.choices)
        controller = CognitiveController(
            n_hypotheses=n,
            hypothesis_labels=[f"choice_{i}" for i in range(n)],
            use_llm=False,
        )
        for i, hyp in enumerate(controller.belief_state.hypotheses):
            hyp.description = sample.choices[i][:50]

        controller.step(sample.prompt)

        posteriors = [
            controller.belief_state.hypotheses[i].probability
            for i in range(n)
        ]

        probs = np.array(posteriors)
        probs = probs[probs > 0]
        if len(probs) > 1:
            entropy = float(-np.sum(probs * np.log(probs + 1e-16)) / np.log(len(probs)))
        else:
            entropy = 0.0

        return posteriors, entropy

    # ─── EVALUATION ─────────────────────────────────────────

    def evaluate_sample(self, sample: TaskSample) -> SampleResult:
        """Evaluate a single sample with full diagnostics."""
        t0 = time.time()
        n = len(sample.choices)
        uniform = 1.0 / n

        # Get Tensegrity scores
        tensegrity_scores, entropy = self._get_tensegrity_scores(sample)

        # Compute bias diagnostics
        deviations = [abs(s - uniform) for s in tensegrity_scores]
        bias_magnitude = max(deviations)
        # bias_applied = posteriors are meaningfully non-uniform
        bias_applied = bias_magnitude > 0.02  # More than 2% deviation from uniform

        # Get baseline scores
        if self.mode == "local":
            self._init_model()
            baseline_scores = self._score_choices_local(sample.prompt, sample.choices)
        else:
            # Offline: random baseline (seeded by sample ID for reproducibility)
            rng = np.random.RandomState(hash(sample.id) % 2**31)
            baseline_scores = rng.randn(n).tolist()

        # Grafted: baseline + λ * tensegrity
        # Normalize tensegrity scores to be on comparable scale to baseline
        # In offline mode, baseline is N(0,1), tensegrity is [0,1] probabilities
        # In local mode, baseline is log-probs (~[-5, 0]), tensegrity is [0,1]
        # Convert tensegrity to log-odds for better scale matching
        tensegrity_logodds = [
            np.log(max(s, 1e-9)) - np.log(uniform)
            for s in tensegrity_scores
        ]

        grafted_scores = [
            b + self.lam * t
            for b, t in zip(baseline_scores, tensegrity_logodds)
        ]

        baseline_pred = int(np.argmax(baseline_scores))
        grafted_pred = int(np.argmax(grafted_scores))

        baseline_correct = (baseline_pred == sample.gold)
        grafted_correct = (grafted_pred == sample.gold)

        # Flip classification
        if baseline_pred == grafted_pred:
            flip_type = "preserved" if baseline_correct else "neutral"
        elif not baseline_correct and grafted_correct:
            flip_type = "good_flip"
        elif baseline_correct and not grafted_correct:
            flip_type = "bad_flip"
        else:
            flip_type = "neutral"  # Both wrong, different wrong answers

        wall_time = time.time() - t0

        return SampleResult(
            sample_id=sample.id,
            task=sample.metadata.get("task", ""),
            gold=sample.gold,
            n_choices=n,
            baseline_pred=baseline_pred,
            grafted_pred=grafted_pred,
            baseline_correct=baseline_correct,
            grafted_correct=grafted_correct,
            baseline_scores=baseline_scores,
            grafted_scores=grafted_scores,
            tensegrity_scores=tensegrity_scores,
            graft_entropy=entropy,
            bias_applied=bias_applied,
            bias_magnitude=bias_magnitude,
            flip_type=flip_type,
            lam=self.lam,
            wall_time=wall_time,
        )

    def evaluate_task(self, task_name: str,
                      max_samples: Optional[int] = None,
                      verbose: bool = False) -> TaskResult:
        """Evaluate all samples in a task with full flip accounting."""
        config = TASK_REGISTRY[task_name]
        samples = load_task_samples(task_name, max_samples)

        if verbose:
            print(f"  [{task_name}] Loaded {len(samples)} samples")

        results = []
        for i, sample in enumerate(samples):
            r = self.evaluate_sample(sample)
            results.append(r)
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

        # Flip accounting
        flips = FlipAccounting()
        for r in results:
            if r.flip_type == "good_flip":
                flips.good_flips += 1
            elif r.flip_type == "bad_flip":
                flips.bad_flips += 1
            elif r.flip_type == "preserved":
                flips.preserved += 1
            elif r.flip_type == "neutral":
                flips.neutral += 1

        # Coverage: fraction where bias was non-trivial
        biased = [r for r in results if r.bias_applied]
        coverage = len(biased) / n

        # Conditional accuracy
        cond_acc_biased = (sum(1 for r in biased if r.grafted_correct) / len(biased)) if biased else 0.0
        unbiased = [r for r in results if not r.bias_applied]
        cond_acc_unbiased = (sum(1 for r in unbiased if r.grafted_correct) / len(unbiased)) if unbiased else 0.0

        return TaskResult(
            task=task_name,
            domain=config.domain,
            n_samples=n,
            lam=self.lam,
            baseline_accuracy=bl_correct / n,
            grafted_accuracy=gr_correct / n,
            delta=(gr_correct - bl_correct) / n,
            baseline_correct=bl_correct,
            grafted_correct=gr_correct,
            coverage=coverage,
            cond_acc_biased=cond_acc_biased,
            cond_acc_unbiased=cond_acc_unbiased,
            mean_bias_magnitude=np.mean([r.bias_magnitude for r in results]),
            mean_graft_entropy=np.mean([r.graft_entropy for r in results]),
            flips=flips,
            mean_wall_time=np.mean([r.wall_time for r in results]),
        )

    def run_benchmark(self, tasks: Optional[List[str]] = None,
                      max_samples_per_task: Optional[int] = None,
                      verbose: bool = True) -> BenchmarkResult:
        """Run the full benchmark across multiple tasks."""
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
                desc = TASK_REGISTRY[task_name].description
                print(f"\n  ▸ {task_name}: {desc}")
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
            model_name=self.model_name,
            mode=self.mode,
            lam=self.lam,
            tasks=task_results,
            overall_baseline_accuracy=total_bl / max(total_n, 1),
            overall_grafted_accuracy=total_gr / max(total_n, 1),
            overall_delta=(total_gr - total_bl) / max(total_n, 1),
            overall_flips=overall_flips,
            total_samples=total_n,
            total_wall_time=total_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if verbose:
            print(f"\n{'═' * 75}")
            print(result.summary_table())
            print(f"\n  λ={self.lam}  Time={total_time:.1f}s")
            print(f"  Total flips: {overall_flips.good_flips}↑ good, "
                  f"{overall_flips.bad_flips}↓ bad, "
                  f"{overall_flips.preserved} preserved, "
                  f"{overall_flips.neutral} neutral")
            print(f"{'═' * 75}")

        return result

    def sweep_lambda(self, tasks: Optional[List[str]] = None,
                     lambdas: Optional[List[float]] = None,
                     max_samples_per_task: Optional[int] = None,
                     verbose: bool = True) -> List[BenchmarkResult]:
        """
        Sweep λ to find optimal graft weight.

        Args:
            lambdas: Values to sweep. Default: [0, 0.1, 0.25, 0.5, 1.0, 2.0]
        """
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
                print(f"  λ={lam_val:<5}  base={result.overall_baseline_accuracy:.1%}  "
                      f"graft={result.overall_grafted_accuracy:.1%}  "
                      f"Δ={sign}{result.overall_delta:.1%}  G/B={gb_str}  "
                      f"({result.overall_flips.good_flips}↑/{result.overall_flips.bad_flips}↓)")

        if verbose:
            # Find optimal λ
            best = max(results, key=lambda r: r.overall_delta)
            print(f"\n  Best λ = {best.lam} → Δ = {best.overall_delta:+.1%}")

        return results

    def save_results(self, result: BenchmarkResult, path: str):
        """Save benchmark results to JSON."""
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")
