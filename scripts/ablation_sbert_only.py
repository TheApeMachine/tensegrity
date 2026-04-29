#!/usr/bin/env python3
"""
SBERT-Only Ablation Baseline.

This script answers the most important question about Tensegrity:
"Does the cognitive layer add value above SBERT-alone?"

It runs the same benchmark tasks but uses ONLY SBERT cosine similarity
to score choices — no NGC, no causal arena, no Hopfield memory, no
belief updates, no falsification. Just:

    score(choice_i) = cosine_sim(sbert(prompt), sbert(prompt + choice_i))

This is the honest baseline the cognitive layer must beat. If the
cognitive layer's Δ over SBERT-alone is positive, the manifold is
doing real work. If it's zero, the manifold is expensive SBERT.

Usage:
    python scripts/ablation_sbert_only.py --max-samples 100
    python scripts/ablation_sbert_only.py --tasks copa,boolq,sciq
"""
import sys
import os
import time
import json
import argparse
import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SBERT-only ablation baseline")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per task")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2", help="SBERT model name")
    parser.add_argument("--output", default=None, help="Save JSON results to file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from tensegrity.bench.tasks import TASK_REGISTRY, load_task_samples

    # Load SBERT
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(args.sbert_model)
        print(f"Loaded SBERT: {args.sbert_model}")
    except Exception as e:
        print(f"FATAL: Could not load SBERT: {e}")
        sys.exit(1)

    tasks = args.tasks.split(",") if args.tasks else list(TASK_REGISTRY.keys())

    print(f"\n{'█' * 60}")
    print(f"  SBERT-ONLY ABLATION BASELINE")
    print(f"  Model: {args.sbert_model}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  N/task: {args.max_samples or 'all'}")
    print(f"{'█' * 60}")

    t_start = time.time()
    all_results = []
    total_correct_sbert = 0
    total_correct_random = 0
    total_n = 0

    for task_name in tasks:
        config = TASK_REGISTRY[task_name]
        samples = load_task_samples(task_name, args.max_samples)
        print(f"\n  ▸ {task_name}: {config.description} ({len(samples)} samples)")

        task_correct_sbert = 0
        task_correct_random = 0
        task_n = len(samples)

        for sample in samples:
            n = len(sample.choices)
            if n == 0:
                continue

            # SBERT-only scoring: cosine(prompt, prompt+choice)
            texts = [sample.prompt] + [f"{sample.prompt} {c}" for c in sample.choices]
            embs = sbert.encode(texts, show_progress_bar=False)
            pe = embs[0]
            pn = np.linalg.norm(pe)
            scores = np.zeros(n)
            if pn > 1e-8:
                for i in range(n):
                    ce = embs[i + 1]
                    cn = np.linalg.norm(ce)
                    if cn > 1e-8:
                        scores[i] = np.dot(pe, ce) / (pn * cn)

            sbert_pred = int(np.argmax(scores))
            if sbert_pred == sample.gold:
                task_correct_sbert += 1

            # Random baseline for comparison
            seed_bytes = hashlib.sha256(sample.id.encode("utf-8")).digest()
            sample_seed = int.from_bytes(seed_bytes[:8], "big", signed=False) % (2**31)
            rng = np.random.RandomState(sample_seed)
            random_pred = int(np.argmax(rng.randn(n)))
            if random_pred == sample.gold:
                task_correct_random += 1

        sbert_acc = task_correct_sbert / max(task_n, 1)
        random_acc = task_correct_random / max(task_n, 1)
        chance = 1.0 / config.n_choices if config.n_choices > 0 else 0.25

        total_correct_sbert += task_correct_sbert
        total_correct_random += task_correct_random
        total_n += task_n

        result = {
            "task": task_name, "domain": config.domain, "n": task_n,
            "sbert_accuracy": round(sbert_acc, 4),
            "random_accuracy": round(random_acc, 4),
            "chance": round(chance, 4),
            "sbert_over_chance": round(sbert_acc - chance, 4),
        }
        all_results.append(result)
        print(f"    SBERT={sbert_acc:.1%}  random={random_acc:.1%}  "
              f"chance={chance:.1%}  SBERT-chance={sbert_acc-chance:+.1%}")

    total_time = time.time() - t_start
    overall_sbert = total_correct_sbert / max(total_n, 1)
    overall_random = total_correct_random / max(total_n, 1)

    print(f"\n{'═' * 75}")
    print(f"  SBERT-only overall: {overall_sbert:.1%}  (random: {overall_random:.1%})")
    print(f"  Total: {total_n} samples, {total_time:.1f}s")
    print(f"{'═' * 75}")

    # Print comparison table
    print(f"\n{'Task':<22} {'N':>5} {'SBERT':>7} {'Random':>7} {'Chance':>7} {'SBERT-Chance':>12}")
    print("─" * 65)
    for r in sorted(all_results, key=lambda x: x["sbert_over_chance"], reverse=True):
        print(f"{r['task']:<22} {r['n']:>5} {r['sbert_accuracy']:>6.1%} "
              f"{r['random_accuracy']:>6.1%} {r['chance']:>6.1%} "
              f"{r['sbert_over_chance']:>+11.1%}")
    print("─" * 65)
    print(f"{'OVERALL':<22} {total_n:>5} {overall_sbert:>6.1%} {overall_random:>6.1%}")

    output = {
        "mode": "sbert_only_ablation",
        "sbert_model": args.sbert_model,
        "overall_sbert_accuracy": round(overall_sbert, 4),
        "overall_random_accuracy": round(overall_random, 4),
        "total_samples": total_n,
        "wall_time_s": round(total_time, 1),
        "tasks": all_results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print(f"\n{json.dumps(output, indent=2)}")


if __name__ == "__main__":
    main()
