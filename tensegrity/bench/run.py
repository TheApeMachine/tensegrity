#!/usr/bin/env python3
"""
Tensegrity Benchmark CLI.

Usage:
    # Quick dev run (offline, 20 samples/task, 3 tasks):
    python -m tensegrity.bench.run --mode offline --max-samples 20 --tasks copa,boolq,logiqa

    # Full offline benchmark (all tasks, all samples):
    python -m tensegrity.bench.run --mode offline

    # Local model benchmark (requires GPU):
    python -m tensegrity.bench.run --mode local --model meta-llama/Llama-3.2-1B-Instruct --max-samples 50

    # Save results:
    python -m tensegrity.bench.run --mode offline --output results.json
"""

import sys
sys.path.insert(0, "/app")
import argparse
import json

from tensegrity.bench.tasks import list_tasks
from tensegrity.bench.runner import EvalRunner


def main():
    parser = argparse.ArgumentParser(description="Tensegrity Benchmark Harness")
    parser.add_argument("--mode", choices=["offline", "local"], default="offline",
                        help="Evaluation mode (default: offline)")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HF model ID for local mode")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated task names (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per task (default: all)")
    parser.add_argument("--scale", type=float, default=2.5,
                        help="Graft logit bias scale")
    parser.add_argument("--entropy-gate", type=float, default=0.85,
                        help="Convergence gate threshold")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available tasks and exit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    if args.list_tasks:
        from tensegrity.bench.tasks import TASK_REGISTRY
        print("\nAvailable tasks:")
        for name, cfg in TASK_REGISTRY.items():
            print(f"  {name:<25} {cfg.description:<45} [{cfg.domain}]")
        return

    tasks = args.tasks.split(",") if args.tasks else None

    runner = EvalRunner(
        model_name=args.model,
        mode=args.mode,
        graft_scale=args.scale,
        graft_entropy_gate=args.entropy_gate,
        seed=args.seed,
    )

    result = runner.run_benchmark(
        tasks=tasks,
        max_samples_per_task=args.max_samples,
        verbose=not args.quiet,
    )

    if args.output:
        runner.save_results(result, args.output)
        print(f"\nResults saved to {args.output}")
    else:
        print(f"\n{json.dumps(result.to_dict(), indent=2)}")


if __name__ == "__main__":
    main()
