"""
Benchmark: "Hypothesis Elimination" — a diagnostic reasoning game.

The game:
  - There are N possible answers (hypotheses).
  - The environment gives clues, one per turn.
  - Some clues CONFIRM certain hypotheses.
  - Some clues CONTRADICT certain hypotheses.
  - Some clues are MISLEADING (noise).
  - The agent must identify the correct answer using as few turns as possible.
  - The agent can also ASK questions to get targeted clues.

This tests:
  1. Belief tracking across turns (can the agent maintain state?)
  2. Bayesian updating (does contradicting evidence actually eliminate hypotheses?)
  3. Epistemic action selection (does the agent ask informative questions?)
  4. Robustness to misleading clues (does the agent recover from bad evidence?)

A plain LLM fails here because:
  - It has no persistent belief state (beliefs live in the rolling context window)
  - It cannot track 8 probabilities simultaneously
  - Misleading clues corrupt its "reasoning" permanently
  - It doesn't know WHAT to ask — it asks random or obvious questions

Tensegrity+LLM should win because:
  - Beliefs are explicit Dirichlet distributions, updated by math, not vibes
  - Elimination is permanent and correct (Bayesian, not associative)
  - Question selection is driven by EFE (maximize information gain)
  - Misleading clues are down-weighted by the causal arena's falsification
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import sys
sys.path.insert(0, '/app')


@dataclass
class GameScenario:
    """A single game scenario with a hidden answer and clue sequence."""
    answer: str
    hypotheses: List[str]
    clues: List[Dict[str, Any]]  # {"text": str, "confirms": [str], "contradicts": [str], "misleading": bool}
    difficulty: str  # "easy", "medium", "hard"
    name: str = ""


def make_animal_scenario() -> GameScenario:
    """Classic: guess the animal."""
    hypotheses = ["cat", "dog", "parrot", "snake", "goldfish", "hamster", "turtle", "rabbit"]
    answer = "parrot"
    
    clues = [
        {"text": "It's a common household pet.", "confirms": hypotheses, "contradicts": [], "misleading": False},
        {"text": "It can make sounds that resemble speech.", "confirms": ["parrot"], "contradicts": ["goldfish", "snake", "turtle"], "misleading": False},
        {"text": "It has fur.", "confirms": ["cat", "dog", "hamster", "rabbit"], "contradicts": ["parrot", "snake", "goldfish", "turtle"], "misleading": True},  # MISLEADING
        {"text": "It does not have four legs.", "confirms": ["parrot", "snake", "goldfish"], "contradicts": ["cat", "dog", "hamster", "rabbit"], "misleading": False},
        {"text": "It can live for over 50 years.", "confirms": ["parrot", "turtle"], "contradicts": ["hamster", "goldfish"], "misleading": False},
        {"text": "It has feathers.", "confirms": ["parrot"], "contradicts": ["cat", "dog", "snake", "goldfish", "hamster", "turtle", "rabbit"], "misleading": False},
    ]
    
    return GameScenario(answer=answer, hypotheses=hypotheses, clues=clues, difficulty="medium", name="animal_guess")


def make_diagnosis_scenario() -> GameScenario:
    """Medical-ish: diagnose the cause of system failure."""
    hypotheses = ["memory_leak", "disk_full", "network_timeout", "cpu_overload", 
                  "dns_failure", "config_error", "dependency_crash", "deadlock"]
    answer = "dependency_crash"
    
    clues = [
        {"text": "The system was working fine until a deployment 2 hours ago.", "confirms": ["config_error", "dependency_crash"], "contradicts": ["disk_full"], "misleading": False},
        {"text": "CPU usage is normal.", "confirms": ["dependency_crash", "dns_failure", "config_error", "disk_full"], "contradicts": ["cpu_overload", "deadlock"], "misleading": False},
        {"text": "Memory usage is gradually increasing.", "confirms": ["memory_leak", "deadlock"], "contradicts": [], "misleading": True},  # MISLEADING — it's correlation, not cause
        {"text": "The error logs show 'ConnectionRefusedError' to an upstream service.", "confirms": ["dependency_crash", "network_timeout"], "contradicts": ["memory_leak", "disk_full", "config_error"], "misleading": False},
        {"text": "Other services on the same network are healthy.", "confirms": ["dependency_crash", "config_error"], "contradicts": ["network_timeout", "dns_failure"], "misleading": False},
        {"text": "The upstream service's health check is returning 503.", "confirms": ["dependency_crash"], "contradicts": ["network_timeout", "dns_failure", "config_error", "memory_leak"], "misleading": False},
    ]
    
    return GameScenario(answer=answer, hypotheses=hypotheses, clues=clues, difficulty="hard", name="root_cause")


def make_murder_mystery() -> GameScenario:
    """Who did it?"""
    hypotheses = ["butler", "gardener", "chef", "librarian", "driver", "maid"]
    answer = "librarian"
    
    clues = [
        {"text": "The incident happened in the study.", "confirms": ["librarian", "butler"], "contradicts": ["gardener", "chef"], "misleading": False},
        {"text": "The butler was seen in the kitchen at the time.", "confirms": ["chef"], "contradicts": ["butler"], "misleading": False},
        {"text": "Muddy footprints were found near the scene.", "confirms": ["gardener", "driver"], "contradicts": [], "misleading": True},  # MISLEADING
        {"text": "A rare book was found with the victim.", "confirms": ["librarian"], "contradicts": ["driver", "maid", "chef"], "misleading": False},
        {"text": "The perpetrator knew the combination to the safe.", "confirms": ["librarian", "butler"], "contradicts": ["gardener", "driver", "maid"], "misleading": False},
        {"text": "The butler confirms the librarian had been arguing with the victim.", "confirms": ["librarian"], "contradicts": [], "misleading": False},
    ]
    
    return GameScenario(answer=answer, hypotheses=hypotheses, clues=clues, difficulty="medium", name="mystery")


class HypothesisEliminationGame:
    """
    Game engine for the hypothesis elimination benchmark.
    
    Manages the clue sequence, tracks agent performance,
    and provides a standardized evaluation interface.
    """
    
    def __init__(self, scenario: GameScenario, noise_rate: float = 0.0):
        """
        Args:
            scenario: The game scenario to play
            noise_rate: Probability of injecting additional misleading clues
        """
        self.scenario = scenario
        self.noise_rate = noise_rate
        
        self.clue_index = 0
        self.turns_taken = 0
        self.concluded = False
        self.agent_answer: Optional[str] = None
        
        # Tracking
        self.clue_history: List[str] = []
        self.belief_snapshots: List[Dict[str, float]] = []
    
    def get_next_clue(self) -> Optional[str]:
        """Get the next clue in sequence."""
        if self.clue_index >= len(self.scenario.clues):
            return None
        
        clue = self.scenario.clues[self.clue_index]
        self.clue_index += 1
        self.turns_taken += 1
        self.clue_history.append(clue["text"])
        
        return clue["text"]
    
    def answer_question(self, question: str) -> str:
        """
        Answer an agent's question about the hypotheses.
        
        Simple keyword matching — the game oracle provides targeted feedback.
        """
        self.turns_taken += 1
        question_lower = question.lower()
        
        # Check each hypothesis
        for hyp in self.scenario.hypotheses:
            if hyp.lower() in question_lower:
                if hyp == self.scenario.answer:
                    return f"That's a possibility worth considering. There is evidence pointing toward {hyp}."
                else:
                    # Give a hint that this isn't it
                    return f"The evidence doesn't strongly support {hyp}."
        
        # Generic response
        return "I can't answer that specifically. Let me give you another clue instead."
    
    def submit_answer(self, answer: str) -> Dict[str, Any]:
        """
        Agent submits their final answer.
        
        Returns evaluation metrics.
        """
        self.concluded = True
        self.agent_answer = answer.lower().strip()
        correct_answer = self.scenario.answer.lower().strip()
        
        correct = self.agent_answer == correct_answer or correct_answer in self.agent_answer
        
        return {
            "correct": correct,
            "answer": self.agent_answer,
            "true_answer": self.scenario.answer,
            "turns_used": self.turns_taken,
            "clues_seen": self.clue_index,
            "total_clues": len(self.scenario.clues),
            "efficiency": 1.0 - (self.turns_taken / (len(self.scenario.clues) + 3)),
            "difficulty": self.scenario.difficulty,
        }
    
    def get_ground_truth_posteriors(self, after_n_clues: int) -> Dict[str, float]:
        """
        Compute the Bayesian-optimal posterior after seeing n clues.
        
        This is the gold standard: what a perfect Bayesian agent would believe.
        """
        # Start with uniform prior
        probs = {h: 1.0 / len(self.scenario.hypotheses) for h in self.scenario.hypotheses}
        
        for i in range(min(after_n_clues, len(self.scenario.clues))):
            clue = self.scenario.clues[i]
            
            if clue["misleading"]:
                continue  # A perfect Bayesian would detect and ignore misleading clues
                          # (this is generous — even the gold standard isn't trivial)
            
            # Update based on confirms/contradicts
            for h in clue["contradicts"]:
                probs[h] *= 0.1  # Strong evidence against
            
            for h in clue["confirms"]:
                probs[h] *= 2.0  # Evidence for
            
            # Normalize
            total = sum(probs.values())
            if total > 0:
                probs = {h: p / total for h, p in probs.items()}
        
        return probs


def run_tensegrity_agent(scenario: GameScenario, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the Tensegrity+LLM agent on a scenario.
    
    Uses template mode (no API calls) for testing.
    """
    from tensegrity.broca.controller import CognitiveController
    
    controller = CognitiveController(
        n_hypotheses=len(scenario.hypotheses),
        hypothesis_labels=scenario.hypotheses,
        use_llm=False,  # Template mode for testing
    )
    
    game = HypothesisEliminationGame(scenario)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  SCENARIO: {scenario.name} (difficulty: {scenario.difficulty})")
        print(f"  True answer: {scenario.answer}")
        print(f"  Hypotheses: {scenario.hypotheses}")
        print(f"{'='*60}")
    
    # Play the game
    concluded = False
    for turn in range(len(scenario.clues) + 3):  # Extra turns for questions
        clue = game.get_next_clue()
        if clue is None:
            break
        
        # Feed clue to cognitive controller
        result = controller.step(clue)
        
        if verbose:
            action = result["action"]
            hyps = result["hypotheses"]
            top = max(hyps, key=lambda h: h["prob"]) if hyps else {"desc": "?", "prob": 0}
            
            print(f"\n  Turn {result['turn']}:")
            print(f"    Clue: \"{clue}\"")
            print(f"    Action: {action['action_type']} → {action.get('content', 'N/A')}")
            print(f"    Top hypothesis: {top['desc']} (p={top['prob']:.4f})")
            print(f"    Tension: {result['perception']['tension']:.4f}")
            print(f"    Epistemic urgency: {result['belief_state']['epistemic_urgency']:.4f}")
        
        # Check if agent wants to conclude
        if result["action"]["action_type"] == "state_conclusion":
            answer = result["action"].get("content", "")
            eval_result = game.submit_answer(answer)
            concluded = True
            break
    
    # If agent didn't conclude, force a conclusion from top hypothesis
    if not concluded:
        if controller.belief_state.hypotheses:
            top = max(controller.belief_state.hypotheses, key=lambda h: h.probability)
            eval_result = game.submit_answer(top.description)
        else:
            eval_result = game.submit_answer("unknown")
    
    # Compute belief accuracy vs Bayesian optimal
    gold = game.get_ground_truth_posteriors(game.clue_index)
    agent_probs = {h.description: h.probability for h in controller.belief_state.hypotheses}
    
    # KL divergence from gold standard
    kl_div = 0.0
    for h in gold:
        p = gold[h]
        q = agent_probs.get(h, 1e-16)
        if p > 0:
            kl_div += p * np.log(p / max(q, 1e-16))
    
    eval_result["kl_from_optimal"] = float(kl_div)
    eval_result["gold_posteriors"] = gold
    eval_result["agent_posteriors"] = agent_probs
    
    if verbose:
        print(f"\n  {'='*40}")
        print(f"  RESULT: {'✓ CORRECT' if eval_result['correct'] else '✗ WRONG'}")
        print(f"  Agent answer: {eval_result['answer']}")
        print(f"  True answer:  {eval_result['true_answer']}")
        print(f"  Turns used:   {eval_result['turns_used']}")
        print(f"  KL from optimal: {eval_result['kl_from_optimal']:.4f}")
        print(f"  Agent posteriors: {json.dumps({k: round(v, 3) for k, v in sorted(agent_probs.items(), key=lambda x: x[1], reverse=True)[:4]})}")
        print(f"  Gold posteriors:  {json.dumps({k: round(v, 3) for k, v in sorted(gold.items(), key=lambda x: x[1], reverse=True)[:4]})}")
    
    return eval_result


def run_benchmark(verbose: bool = True):
    """Run the full benchmark across all scenarios."""
    scenarios = [
        make_animal_scenario(),
        make_diagnosis_scenario(),
        make_murder_mystery(),
    ]
    
    print("\n" + "█" * 60)
    print("  TENSEGRITY BENCHMARK: Hypothesis Elimination Game")
    print("  Testing: Belief tracking, Bayesian updating, Epistemic action")
    print("█" * 60)
    
    results = []
    for scenario in scenarios:
        result = run_tensegrity_agent(scenario, verbose=verbose)
        results.append(result)
    
    # Summary
    n_correct = sum(1 for r in results if r["correct"])
    avg_turns = np.mean([r["turns_used"] for r in results])
    avg_kl = np.mean([r["kl_from_optimal"] for r in results])
    
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.0f}%)")
    print(f"  Avg turns used: {avg_turns:.1f}")
    print(f"  Avg KL from optimal: {avg_kl:.4f}")
    print(f"  Scenarios: {[s.name for s in scenarios]}")
    
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"    {status} {r['true_answer']:20s} → agent said '{r['answer'][:30]}' "
              f"({r['turns_used']} turns, KL={r['kl_from_optimal']:.3f})")
    
    return results


if __name__ == "__main__":
    results = run_benchmark(verbose=True)
