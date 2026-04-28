"""
Evaluation Runner: Baseline (plain LLM) vs Grafted (Tensegrity+LLM).

Two evaluation modes per sample:

  BASELINE: Model scores each choice via log-probability.
            P(choice | prompt) computed from raw logits.
            Prediction = argmax over choices.

  GRAFTED:  score(choice) = llm_logprob(choice) + λ_eff * tensegrity_score(choice)
            Where λ_eff = λ * (1 - LLM_confidence/threshold) in local mode.
            The graft is a confidence-gated tiebreaker, not an equal-weight competitor.

The ONLY difference is the additive Tensegrity term.
This is a controlled A/B comparison.

Local mode blending:
  1. Compute LLM confidence from normalized entropy of softmax(log_probs)
  2. If LLM confident (norm_entropy < 0.4): don't apply graft (preserve good preds)
  3. If LLM uncertain: apply graft scaled by uncertainty and proportional to LLM score range
  4. Graft magnitude capped at 0.3× of the LLM's own score spread
"""

import numpy as np
import time
import json
import logging
import hashlib
from typing import List, Optional, Tuple
from dataclasses import dataclass

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
    Runs baseline vs grafted evaluation on any set of tasks.

    Modes:
      "local"   — Uses transformers model + confidence-gated semantic field scoring
      "offline"  — No LLM; baseline = random, grafted = field scoring

    Local mode blending:
      effective_λ = λ * (1 - LLM_confidence / confidence_gate_threshold)
      graft_score is scaled to 0.3× of LLM score range
      If LLM is confident (entropy < 0.4×max): effective_λ = 0 (don't interfere)
    """

    def __init__(
        self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        mode: str = "offline",
        lam: float = 1.0,
        seed: int = 42,
    ):
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

        dtype, device_map, move_to = inference_load_settings()
        logger.info(f"Loading model {self.model_name}...")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype, device_map=device_map)
        
        if move_to is not None:
            self._model = self._model.to(move_to)
        
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

    def _get_tensegrity_scores(self, sample: TaskSample) -> Tuple[List[float], float]:
        """Run the canonical agent pipeline on one item.

        Wires every shipped subsystem: CognitiveController + TensegrityAgent
        (UnifiedField, FreeEnergyEngine, EpistemicMemory, EpisodicMemory,
        AssociativeMemory, log-lik CausalArena), Broca dynamic SCM injection,
        EnergyCausalArena + TopologyMapper for per-choice causal competition,
        NGC top-down falsification.

        **Bench-specific behavior**: In ``single`` scorer mode (`TENSEGRITY_SCORER` env),
        :meth:`ScoringBridge.reset` is called **once per benchmark sample**, so episodic /
        Hopfield state does not accumulate across MC items — each example is isolated.

        In the default canonical mode, reuse a single :class:`CanonicalPipeline` for all
        samples — per-item hypotheses and SMCs come from ``reset_for_item`` /
        ``_soft_reset_in_place``. Rebuilding the pipeline on each row would recreate
        the agent stack and repeatedly load sentence-transformer weights into memory.
        :meth:`CanonicalPipeline.reset_session` is invoked **once per task**
        (``EvalRunner.evaluate_task``), wiping cross-task leakage while permitting
        within-task learning where applicable.

        Prefer ``canonical`` for behavior aligned with HybridPipeline/session semantics;
        use ``single`` for a deterministic, isolated field snapshot per sample.
        """
        import os
        import numpy as np

        # Escape hatch: opt back into the legacy single-shot field scorer.
        if os.environ.get("TENSEGRITY_SCORER", "canonical").lower() == "single":
            from tensegrity.engine.scoring import ScoringBridge

            if not hasattr(self, "_field_scorer"):
                self._field_scorer = ScoringBridge(
                    obs_dim=256, hidden_dims=[128, 32], fhrr_dim=2048,
                    ngc_settle_steps=30, ngc_learning_rate=0.01,
                    hopfield_beta=0.05, confidence_threshold=0.15,
                    context_settle_steps=40, choice_settle_steps=25,
                    context_learning_epochs=3,
                )
            self._field_scorer.reset()
            return self._field_scorer.score_choices(sample.prompt, sample.choices)

        from tensegrity.pipeline.canonical import CanonicalPipeline

        if not hasattr(self, "_canonical"):
            # One CanonicalPipeline instance for all samples on this Runner. Hypothesis texts
            # and per-choice SCMs are updated per sample inside ``reset_for_item`` /
            # ``_soft_reset_in_place``; rebuilding ``CanonicalPipeline`` whenever multi-choice
            # strings changed would recreate ``TensegrityAgent`` / FHRR SBERT loaders and spam
            # "Loading weights" for each benchmark row (see CanonicalPipeline docs).
            self._canonical = CanonicalPipeline(
                hypothesis_labels=None,
                use_llm_broca=False,
                enable_hypothesis_generation=False,
                model_name=self.model_name,
                max_iterations=3,
                commit_ratio=2.0,
                falsify_settle_steps=15,
                falsify_update_strength=1.0,
                energy_arena_precision=1.0,
                energy_arena_beta=1.0,
            )

        result = self._canonical.score_multichoice(sample)
        n = len(result.scores)

        if n == 0:
            return [], 1.0

        # Belief-entropy normalized to [0, 1] for the harness's confidence gate.
        b = np.asarray(result.belief, dtype=np.float64)
        bm = b[b > 0]
        entropy = float(-np.sum(bm * np.log(bm)) / np.log(n)) if n > 1 and len(bm) > 1 else 0.0

        # Hard gate: if the agent loop did not converge, signal uninformative
        # (the LLM should lead). Mirrors the single-shot scorer's gate-to-zero.
        if not result.converged:
            return [0.0] * n, 1.0

        return list(result.scores), entropy

    def evaluate_sample(self, sample: TaskSample) -> SampleResult:
        """Evaluate a single sample with confidence-gated blending."""
        t0 = time.time()
        n = len(sample.choices)

        tensegrity_scores, entropy = self._get_tensegrity_scores(sample)

        scores_arr = np.array(tensegrity_scores)
        score_spread = float(scores_arr.max() - scores_arr.min()) if len(scores_arr) > 0 else 0.0
        bias_magnitude = score_spread
        bias_applied = score_spread > 0.01

        if self.mode == "local":
            self._init_model()
            baseline_scores = self._score_choices_local(sample.prompt, sample.choices)
        else:
            seed_bytes = hashlib.sha256(sample.id.encode("utf-8")).digest()
            sample_seed = int.from_bytes(seed_bytes[:8], "big", signed=False) % (2**31)
            rng = np.random.RandomState(sample_seed)
            baseline_scores = rng.randn(n).tolist()

        # === CONFIDENCE-GATED BLENDING ===
        base_arr = np.array(baseline_scores)

        if self.mode == "local":
            # LLM confidence from softmax entropy
            shifted_base = base_arr - base_arr.max()
            base_probs = np.exp(shifted_base)
            base_probs_sum = base_probs.sum()
            base_probs = base_probs / base_probs_sum if base_probs_sum > 0 else np.ones(n) / n

            if n > 1:
                base_entropy = float(-np.sum(base_probs * np.log(base_probs + 1e-16)))
                max_entropy = float(np.log(n))
                llm_norm_entropy = base_entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                llm_norm_entropy = 0.0

            llm_confidence = 1.0 - llm_norm_entropy

            # Gate: if LLM is confident, don't interfere
            confidence_gate_threshold = 0.6

            if llm_confidence > confidence_gate_threshold:
                effective_lam = 0.0
                bias_applied = False
            else:
                uncertainty_scale = (1.0 - llm_confidence / confidence_gate_threshold)
                effective_lam = self.lam * uncertainty_scale

            # Scale graft relative to LLM score range (0.3× = gentle nudge)
            base_spread = float(base_arr.max() - base_arr.min())

            if base_spread > 1e-8 and bias_applied:
                t_std = float(scores_arr.std())
                if t_std > 1e-8:
                    normalized_t = [(s - float(scores_arr.mean())) / t_std * base_spread * 0.3
                                    for s in tensegrity_scores]
                else:
                    normalized_t = [0.0] * n
            else:
                normalized_t = [0.0] * n
        else:
            # Offline: simple z-normalization
            effective_lam = self.lam

            if bias_applied and score_spread > 0:
                t_std = float(scores_arr.std())
                if t_std > 1e-8:
                    normalized_t = [(s - float(scores_arr.mean())) / t_std for s in tensegrity_scores]
                else:
                    normalized_t = [0.0] * n
            else:
                normalized_t = [0.0] * n

        grafted_scores = [b + effective_lam * t for b, t in zip(baseline_scores, normalized_t)]

        baseline_pred = int(np.argmax(baseline_scores))
        grafted_pred = int(np.argmax(grafted_scores))
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

        return SampleResult(
            sample_id=sample.id, task=sample.metadata.get("task", ""),
            gold=sample.gold, n_choices=n,
            baseline_pred=baseline_pred, grafted_pred=grafted_pred,
            baseline_correct=baseline_correct, grafted_correct=grafted_correct,
            baseline_scores=baseline_scores, grafted_scores=grafted_scores,
            tensegrity_scores=tensegrity_scores, graft_entropy=entropy,
            bias_applied=bias_applied, bias_magnitude=bias_magnitude,
            flip_type=flip_type, lam=self.lam, wall_time=time.time() - t0,
        )

    def evaluate_task(
        self, task_name: str, max_samples: Optional[int] = None, verbose: bool = False,
    ) -> TaskResult:
        config = TASK_REGISTRY[task_name]
        samples = load_task_samples(task_name, max_samples)

        if verbose:
            print(f"  [{task_name}] Loaded {len(samples)} samples")

        # Start a fresh memory session per task: episodic + Hopfield + energy
        # arena wiped, so cross-item learning operates within a task but
        # doesn't leak priors across tasks (different label spaces).
        if hasattr(self, "_canonical") and hasattr(self._canonical, "reset_session"):
            self._canonical.reset_session()
        if hasattr(self, "_field_scorer"):
            if hasattr(self._field_scorer, "reset_session"):
                self._field_scorer.reset_session()
            elif hasattr(self._field_scorer, "reset"):
                self._field_scorer.reset()

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
