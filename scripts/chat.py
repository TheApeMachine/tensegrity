"""
Free-chat mode for Tensegrity.

The cognitive layer is the agent. Each user turn is a perception cycle that
runs the full agent stack (UnifiedField + FreeEnergyEngine + EpistemicMemory
+ EpisodicMemory + AssociativeMemory + log-likelihood CausalArena +
EnergyCausalArena + TopologyMapper). The LLM enters at exactly one place:
the final Broca verbalization, where logit grafting under semantic vocabulary
grounding shapes the LLM's tokens to be coherent with the agent's converged
beliefs.

This is the architecture from the Sensorium paper: the manifold reasons; the
LLM narrates.

Usage:
    python scripts/chat.py
    python scripts/chat.py --hypotheses "explanation_a,explanation_b,explanation_c"
    python scripts/chat.py --offline   # no LLM; agent prints converged belief

Type :state to dump the agent's BeliefState. Type :memory to dump episodic
memory. Type :quit to exit.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from tensegrity.graft.pipeline import HybridPipeline


def parse_args():
    ap = argparse.ArgumentParser(description="Tensegrity free-chat mode")
    ap.add_argument(
        "--hypotheses",
        default="positive,neutral,negative,uncertain",
        help="Comma-separated initial hypothesis labels (the agent reasons over these).",
    )
    ap.add_argument(
        "--mode",
        default="local",
        choices=["local", "remote", "offline"],
        help="LLM mode for narration. 'offline' bypasses the LLM entirely.",
    )
    ap.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id for narration.",
    )
    ap.add_argument(
        "--scale", type=float, default=2.5, help="Logit graft scale.",
    )
    ap.add_argument(
        "--entropy-gate", type=float, default=0.85,
        help="Above this normalized entropy, no graft is applied (LLM speaks freely).",
    )
    return ap.parse_args()


def banner(args):
    print("=" * 78)
    print("  TENSEGRITY CHAT")
    print(f"  hypotheses : {args.hypotheses}")
    print(f"  mode       : {args.mode}")
    if args.mode != "offline":
        print(f"  model      : {args.model}")
        print(f"  graft      : semantic grounding (sbert phrase projection)")
    print("  commands   : :state  :memory  :quit")
    print("=" * 78)


def dump_state(pipe: HybridPipeline) -> None:
    bs = pipe.controller.belief_state
    rows = [
        {
            "hypothesis": h.description,
            "p": round(h.probability, 3),
            "supports": len(h.supporting_evidence),
            "contradicts": len(h.contradicting_evidence),
        }
        for h in bs.hypotheses
    ]
    rows.sort(key=lambda r: r["p"], reverse=True)
    print(json.dumps({
        "turn": bs.turn,
        "tension": round(bs.current_tension, 3),
        "free_energy": round(bs.free_energy, 3),
        "epistemic_urgency": round(bs.epistemic_urgency, 3),
        "eliminated": bs.eliminated_hypotheses,
        "hypotheses": rows,
        "confirmed_facts": bs.confirmed_facts[-5:],
    }, indent=2))


def dump_memory(pipe: HybridPipeline) -> None:
    ep = pipe.controller.agent.episodic
    print(json.dumps({
        "n_episodes": len(ep.episodes),
        "stats": ep.statistics,
    }, indent=2, default=str))


def main():
    args = parse_args()
    hypotheses = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
    if len(hypotheses) < 2:
        print("error: need at least two hypotheses", file=sys.stderr)
        sys.exit(2)

    pipe = HybridPipeline(
        hypothesis_labels=hypotheses,
        model_name=args.model,
        mode=args.mode,
        scale=args.scale,
        entropy_gate=args.entropy_gate,
        async_graft=True,
        semantic_grounding=(args.mode != "offline"),
    )
    banner(args)

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == ":quit":
            break
        if line == ":state":
            dump_state(pipe)
            continue
        if line == ":memory":
            dump_memory(pipe)
            continue

        # Perception updates the agent's belief state. This runs the full
        # cognitive stack — no LLM in this step.
        try:
            pipe.process_observation(line)
        except Exception as e:
            print(f"[perception failed: {type(e).__name__}: {e}]")
            traceback.print_exc()
            continue

        # Generation: LLM narrates the converged belief, with semantic-
        # grounded logit grafting from the cognitive layer.
        try:
            res = pipe.generate_response(
                "Given everything observed so far, what is the agent's best summary?",
                max_tokens=100,
            )
        except Exception as e:
            print(f"[generation failed: {type(e).__name__}: {e}]")
            traceback.print_exc()
            continue

        text = res.get("text", "").strip() or "(no narration)"
        beliefs = res.get("beliefs", {})
        mode_label = res.get("mode", "?")
        top = max(beliefs, key=beliefs.get) if beliefs else "(none)"
        top_p = beliefs.get(top, 0.0)
        print(f"agent[{mode_label}] {text}")
        print(f"  → top hypothesis: {top}  (p={top_p:.2f})")


if __name__ == "__main__":
    main()
