#!/usr/bin/env python3
"""
Tensegrity Benchmark CLI.

Usage:
    # Quick benchmark (offline, 50 samples/task):
    python -m tensegrity.bench.run --mode offline --max-samples 50 --tasks copa,boolq,sciq

    # Full offline benchmark:
    python -m tensegrity.bench.run --mode offline

    # λ sweep (find optimal graft weight):
    python -m tensegrity.bench.run --sweep --max-samples 100 --tasks copa,sciq,truthfulqa

    # Local model benchmark (requires GPU):
    python -m tensegrity.bench.run --mode local --model meta-llama/Llama-3.2-1B-Instruct

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

    parser.add_argument(
        "--mode", choices=["offline", "local"], default="offline", help="Evaluation mode (default: offline)"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-1B-Instruct", help="HF model ID for local mode"
    )
    parser.add_argument(
        "--tasks", default=None, help="Comma-separated task names (default: all)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples per task (default: all)"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="λ — precision of the LLM-likelihood evidence channel inside the canonical posterior",
    )
    parser.add_argument(
        "--state-path",
        default=".tensegrity/agent_state.pkl",
        help="Persistent canonical agent state path (default: .tensegrity/agent_state.pkl)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run λ sweep over [0, 0.1, 0.25, 0.5, 1.0, 2.0]"
    )
    parser.add_argument(
        "--sweep-lambdas", default=None, help="Custom λ values for sweep (comma-separated, e.g. 0,0.5,1,2,4)"
    )
    parser.add_argument(
        "--output", default=None, help="Save results to JSON file"
    )
    parser.add_argument(
        "--list-tasks", action="store_true", help="List available tasks and exit"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output"
    )

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
        lam=args.lam,
        seed=args.seed,
        state_path=args.state_path,
    )

    if args.sweep:
        lambdas = None
        
        if args.sweep_lambdas:
            lambdas = [float(x) for x in args.sweep_lambdas.split(",")]
        
        results = runner.sweep_lambda(
            tasks=tasks,
            lambdas=lambdas,
            max_samples_per_task=args.max_samples,
            verbose=not args.quiet,
        )

        if args.output:
            sweep_data = [r.to_dict() for r in results]
        
            with open(args.output, "w") as f:
                json.dump(sweep_data, f, indent=2)
        
            print(f"\nSweep results saved to {args.output}")
    else:
        result = runner.run_benchmark(
            tasks=tasks,
            max_samples_per_task=args.max_samples,
            verbose=not args.quiet,
        )
        
        if args.output:
            runner.save_results(result, args.output)
            print(f"\nResults saved to {args.output}")
        elif not args.quiet:
            print(f"\n{json.dumps(result.to_dict(), indent=2)}")


if __name__ == "__main__":
    main()
