"""
Compare single-shot ScoringBridge vs iterative cognitive scorer on a slice
of the benchmark. No LLM in either path — this isolates the cognitive
contribution.
"""
from __future__ import annotations

import time
import argparse
import logging
import warnings

import numpy as np

from tensegrity.bench.tasks import load_task_samples
from tensegrity.engine.scoring import ScoringBridge
from tensegrity.pipeline.iterative import IterativeCognitiveScorer

logging.basicConfig(level=logging.WARNING)


TASKS_TO_RUN = [
    ("truthfulqa", 30),         # graft-friendly today
    ("mmlu_philosophy", 30),    # graft-hostile
    ("winogrande", 30),         # graft-dead
    ("arc_challenge", 30),      # mid
    ("copa", 20),               # causal, small
    ("logical_deduction", 30),  # logic
]


def run_task(task_name: str, n: int):
    samples = load_task_samples(task_name, max_samples=n)
    if not samples:
        print(f"  [{task_name}] no samples")
        return None

    shared_params = {
        "obs_dim": 256,
        "hidden_dims": [128, 32],
        "fhrr_dim": 2048,
        "ngc_settle_steps": 30,
        "ngc_learning_rate": 0.01,
        "hopfield_beta": 0.05,
        "context_settle_steps": 40,
        "choice_settle_steps": 25,
        "context_learning_epochs": 3,
    }
    single = ScoringBridge(
        **shared_params,
        confidence_threshold=0.15,
    )
    iterative = IterativeCognitiveScorer(
        **shared_params,
        max_iterations=6,
        convergence_top_p=0.75,
        w_sbert=1.0,
        w_fhrr=0.3,
        w_ngc=0.6,
        belief_step=0.6,
        shaping_lr_scale=0.5,
        use_hopfield=True,
        hopfield_steps=2,
    )

    n_total = len(samples)
    n_single_correct = 0
    n_iter_correct = 0
    n_iter_used_total = 0
    n_iter_converged = 0
    n_disagree = 0
    n_iter_better = 0
    n_single_better = 0

    t_single = 0.0
    t_iter = 0.0

    for s in samples:
        # Single-shot
        single.reset()
        t0 = time.time()
        scores_s, _ = single.score_choices(s.prompt, s.choices)
        t_single += time.time() - t0
        # If gated to all zeros, fall back to sbert-only argmax — matches benchmark.
        sa = np.array(scores_s)
        if np.allclose(sa, 0.0):
            # use raw sbert sim as tiebreaker (single's gate = uninformative)
            if hasattr(single, "sentence_similarities"):
                sims = single.sentence_similarities(s.prompt, s.choices)
            elif hasattr(single, "_sentence_similarities"):
                warnings.warn(
                    "ScoringBridge has no public sentence_similarities(); using "
                    "_sentence_similarities (private). Prefer adding a stable public API.",
                    UserWarning,
                    stacklevel=2,
                )
                sims = single._sentence_similarities(s.prompt, s.choices)
            else:
                raise AttributeError(
                    "ScoringBridge exposes no sentence_similarities() or "
                    "_sentence_similarities(); add a public API on ScoringBridge for tie-breaks.",
                )
            pred_s = int(np.argmax(sims))
        else:
            pred_s = int(np.argmax(sa))

        # Iterative
        iterative.reset()
        t0 = time.time()
        result = iterative.score(s.prompt, s.choices)
        t_iter += time.time() - t0
        pred_i = result.committed_idx

        ok_s = (pred_s == s.gold)
        ok_i = (pred_i == s.gold)
        n_single_correct += int(ok_s)
        n_iter_correct += int(ok_i)
        n_iter_used_total += result.iterations_used
        n_iter_converged += int(result.converged)
        if pred_s != pred_i:
            n_disagree += 1
            if ok_i and not ok_s:
                n_iter_better += 1
            elif ok_s and not ok_i:
                n_single_better += 1

    acc_s = n_single_correct / n_total
    acc_i = n_iter_correct / n_total
    print(
        f"  [{task_name:<22}] N={n_total:3d}  "
        f"single={acc_s:5.1%}  iter={acc_i:5.1%}  "
        f"Δ={(acc_i-acc_s):+5.1%}  "
        f"disagree={n_disagree:2d}  "
        f"iter→✓={n_iter_better}  iter→✗={n_single_better}  "
        f"avg_iters={n_iter_used_total/n_total:.1f}  "
        f"conv={n_iter_converged}/{n_total}  "
        f"t_s={t_single:.1f}s  t_i={t_iter:.1f}s"
    )
    return {
        "task": task_name, "n": n_total,
        "single": acc_s, "iter": acc_i, "delta": acc_i - acc_s,
        "disagree": n_disagree, "iter_better": n_iter_better,
        "single_better": n_single_better,
        "avg_iters": n_iter_used_total / n_total,
        "converged": n_iter_converged,
        "t_single_s": t_single, "t_iter_s": t_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task names; default = small fixed slice")
    ap.add_argument("--n", type=int, default=None,
                    help="override per-task sample count")
    args = ap.parse_args()

    if args.tasks:
        plan = [(t, args.n or 30) for t in args.tasks]
    else:
        plan = TASKS_TO_RUN
        if args.n is not None:
            plan = [(t, args.n) for t, _ in plan]

    print("=" * 110)
    print("Single-shot ScoringBridge vs Iterative cognitive scorer (LLM-free)")
    print("=" * 110)
    rows = []
    for t, n in plan:
        try:
            r = run_task(t, n)
        except Exception as e:
            print(f"  [{t}] FAILED: {type(e).__name__}: {e}")
            continue
        if r is not None:
            rows.append(r)

    if not rows:
        return

    print("-" * 110)
    total_n = sum(r["n"] for r in rows)
    sum_s = sum(r["single"] * r["n"] for r in rows) / total_n
    sum_i = sum(r["iter"] * r["n"] for r in rows) / total_n
    print(
        f"  {'OVERALL':<24} N={total_n:3d}  "
        f"single={sum_s:5.1%}  iter={sum_i:5.1%}  "
        f"Δ={(sum_i-sum_s):+5.1%}  "
        f"disagree={sum(r['disagree'] for r in rows):3d}  "
        f"iter→✓={sum(r['iter_better'] for r in rows):3d}  "
        f"iter→✗={sum(r['single_better'] for r in rows):3d}"
    )


if __name__ == "__main__":
    main()
