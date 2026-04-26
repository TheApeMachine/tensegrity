"""
Test the hybrid pipeline: Tensegrity cognitive layer + logit-bias graft.

Tests:
  1. Offline mode: cognitive layer resolves without LLM
  2. Logit bias computation: beliefs → vocabulary biases
  3. Convergence gating: no bias when beliefs are uncertain
  4. Full pipeline on the benchmark scenarios
"""

import sys
sys.path.insert(0, '/app')
import numpy as np
import json

np.random.seed(42)


def test_offline_pipeline():
    """Test the full pipeline in offline mode (no LLM)."""
    print("=" * 60)
    print("TEST 1: Offline Pipeline (Cognitive Resolution Without LLM)")
    print("=" * 60)
    
    from tensegrity.graft.pipeline import HybridPipeline
    
    pipeline = HybridPipeline(
        hypothesis_labels=["parrot", "cat", "dog", "snake", "goldfish", "turtle"],
        hypothesis_keywords={
            "parrot": ["parrot", "bird", "feather", "beak", "wings", "speech", "talk"],
            "cat": ["cat", "fur", "meow", "purr", "whiskers", "paws"],
            "dog": ["dog", "fur", "bark", "wag", "tail", "paws"],
            "snake": ["snake", "scales", "slither", "venom", "cold"],
            "goldfish": ["goldfish", "fish", "swim", "water", "bowl"],
            "turtle": ["turtle", "shell", "slow", "reptile"],
        },
        mode="offline",
        scale=2.5,
        entropy_gate=0.85,
    )
    
    clues = [
        "It can make sounds that resemble speech.",
        "It does not have four legs.",
        "It can live for over 50 years.",
        "It has feathers.",
    ]
    
    result = pipeline.run_scenario(clues, verbose=True)
    
    final = result["final_beliefs"]
    winner = max(final, key=final.get) if final else "?"
    print(f"\n  Winner: {winner} (p={final.get(winner, 0):.3f})")
    print(f"  Output: \"{result['generation']['text']}\"")
    assert winner == "parrot", f"Expected 'parrot', got '{winner}'"
    print(f"  ✓ Correct answer identified via pure cognitive resolution")
    
    return True


def test_logit_bias_computation():
    """Test that beliefs correctly map to logit biases."""
    print("\n" + "=" * 60)
    print("TEST 2: Logit Bias Computation (Beliefs → Token Biases)")
    print("=" * 60)
    
    from tensegrity.graft.logit_bias import StaticLogitBiasBuilder
    
    hypothesis_tokens = {
        "parrot": {100, 101, 102},
        "cat": {200, 201},
        "snake": {300, 301, 302, 303},
    }
    
    builder = StaticLogitBiasBuilder(
        hypothesis_tokens=hypothesis_tokens,
        scale=2.5,
        suppress_threshold=0.01,
    )
    
    # Case 1: Parrot dominant
    posteriors = {"parrot": 0.8, "cat": 0.15, "snake": 0.05}
    bias = builder.build(posteriors)
    
    print(f"  Posteriors: {posteriors}")
    print(f"  Bias for token 100 (parrot): {bias.get(100, 0):.3f}")
    print(f"  Bias for token 200 (cat):    {bias.get(200, 0):.3f}")
    print(f"  Bias for token 300 (snake):  {bias.get(300, 0):.3f}")
    
    assert bias[100] > 0, "Parrot tokens should be boosted"
    assert bias[200] < bias[100], "Cat tokens should be less boosted than parrot"
    assert bias[300] < 0, "Snake tokens should be negative (below uniform)"
    print(f"  ✓ Dominant hypothesis → positive bias, suppressed → negative bias")
    
    # Case 2: Snake eliminated
    posteriors_eliminated = {"parrot": 0.85, "cat": 0.14, "snake": 0.005}
    bias_elim = builder.build(posteriors_eliminated)
    
    print(f"\n  Posteriors with elimination: {posteriors_eliminated}")
    print(f"  Bias for token 300 (eliminated snake): {bias_elim.get(300, 0):.1f}")
    assert bias_elim[300] == -100.0, "Eliminated hypothesis should get -100"
    print(f"  ✓ Eliminated hypothesis → hard suppress (-100)")
    
    # Case 3: Uniform (no information)
    posteriors_uniform = {"parrot": 0.333, "cat": 0.333, "snake": 0.334}
    bias_uniform = builder.build(posteriors_uniform)
    
    # All biases should be near zero
    max_bias = max(abs(v) for v in bias_uniform.values())
    print(f"\n  Uniform posteriors: max |bias| = {max_bias:.4f}")
    assert max_bias < 0.1, "Uniform posteriors should produce near-zero biases"
    print(f"  ✓ Uniform beliefs → near-zero biases (no information to inject)")
    
    return True


def test_convergence_gating():
    """Test that the graft doesn't emit when beliefs are uncertain."""
    print("\n" + "=" * 60)
    print("TEST 3: Convergence Gating (Never Worse Than Base)")
    print("=" * 60)
    
    import torch
    from tensegrity.graft.logit_bias import TensegrityLogitsProcessor
    
    hypothesis_tokens = {
        "parrot": {100, 101},
        "cat": {200, 201},
    }
    
    # Case 1: Uncertain beliefs → should NOT emit
    uncertain_beliefs = {"parrot": 0.55, "cat": 0.45}
    processor = TensegrityLogitsProcessor(
        hypothesis_tokens=hypothesis_tokens,
        belief_fn=lambda: uncertain_beliefs,
        vocab_size=1000,
        scale=2.5,
        entropy_gate=0.6,  # Require fairly resolved beliefs
        min_confidence=0.3,
    )
    
    fake_input_ids = torch.zeros(1, 5, dtype=torch.long)
    fake_scores = torch.randn(1, 1000)
    
    modified = processor(fake_input_ids, fake_scores)
    
    # Should be unmodified
    diff = (modified - fake_scores).abs().max().item()
    print(f"  Uncertain beliefs (0.55/0.45): bias emitted = {processor.state.bias_emitted}")
    print(f"  Max score change: {diff:.6f}")
    assert not processor.state.bias_emitted, "Should NOT emit when uncertain"
    assert diff < 1e-6, "Scores should be unmodified"
    print(f"  ✓ Convergence gate blocked emission — LLM gets native logits")
    
    # Case 2: Resolved beliefs → should emit
    resolved_beliefs = {"parrot": 0.92, "cat": 0.08}
    processor2 = TensegrityLogitsProcessor(
        hypothesis_tokens=hypothesis_tokens,
        belief_fn=lambda: resolved_beliefs,
        vocab_size=1000,
        scale=2.5,
        entropy_gate=0.6,
        min_confidence=0.3,
    )
    
    modified2 = processor2(fake_input_ids, fake_scores.clone())
    diff2 = (modified2 - fake_scores).abs().max().item()
    
    print(f"\n  Resolved beliefs (0.92/0.08): bias emitted = {processor2.state.bias_emitted}")
    print(f"  Max score change: {diff2:.4f}")
    assert processor2.state.bias_emitted, "Should emit when resolved"
    assert diff2 > 0.1, "Scores should be modified"
    
    # Check parrot tokens got boosted, cat tokens got suppressed
    parrot_boost = modified2[0, 100].item() - fake_scores[0, 100].item()
    cat_change = modified2[0, 200].item() - fake_scores[0, 200].item()
    print(f"  Parrot token 100 boost: {parrot_boost:+.4f}")
    print(f"  Cat token 200 change:   {cat_change:+.4f}")
    assert parrot_boost > 0, "Winning hypothesis tokens should be boosted"
    print(f"  ✓ Resolved beliefs → biases emitted, winner boosted")
    
    return True


def test_rca_pipeline():
    """Run the root cause analysis scenario end-to-end."""
    print("\n" + "=" * 60)
    print("TEST 4: Root Cause Analysis Pipeline (Full Scenario)")
    print("=" * 60)
    
    from tensegrity.graft.pipeline import HybridPipeline
    
    pipeline = HybridPipeline(
        hypothesis_labels=["memory_leak", "disk_full", "network_timeout", "cpu_overload",
                          "dns_failure", "config_error", "dependency_crash", "deadlock"],
        hypothesis_keywords={
            "memory_leak": ["memory", "leak", "OOM", "heap", "allocation"],
            "disk_full": ["disk", "storage", "full", "space", "filesystem"],
            "network_timeout": ["network", "timeout", "latency", "packet", "connection"],
            "cpu_overload": ["CPU", "processor", "load", "utilization", "compute"],
            "dns_failure": ["DNS", "resolution", "hostname", "lookup", "nameserver"],
            "config_error": ["config", "configuration", "setting", "parameter", "deploy"],
            "dependency_crash": ["dependency", "upstream", "service", "503", "refused", "crash"],
            "deadlock": ["deadlock", "lock", "mutex", "contention", "blocked"],
        },
        mode="offline",
        scale=2.5,
        entropy_gate=0.8,
    )
    
    clues = [
        "The system was working fine until a deployment 2 hours ago.",
        "CPU usage is normal.",
        "Memory usage is gradually increasing.",  # MISLEADING
        "The error logs show 'ConnectionRefusedError' to an upstream service.",
        "Other services on the same network are healthy.",
        "The upstream service's health check is returning 503.",
    ]
    
    print(f"  True root cause: dependency_crash")
    print(f"  Note: Clue 3 is misleading (memory is correlated, not causal)")
    print()
    
    result = pipeline.run_scenario(clues, verbose=True)
    
    final = result["final_beliefs"]
    winner = max(final, key=final.get) if final else "?"
    
    print(f"\n  Final answer: {winner}")
    assert winner == "dependency_crash", f"Expected 'dependency_crash', got '{winner}'"
    
    # Check that misleading clue was recovered from
    traj = result["belief_trajectory"]
    ml_prob_after_misleading = traj[2]["posteriors"].get("memory_leak", 0)
    ml_prob_final = final.get("memory_leak", 0)
    print(f"  memory_leak after misleading clue: p={ml_prob_after_misleading:.3f}")
    print(f"  memory_leak final: p={ml_prob_final:.3f}")
    assert ml_prob_final < ml_prob_after_misleading, "Should recover from misleading evidence"
    print(f"  ✓ Recovered from misleading evidence via Bayesian updating")
    print(f"  ✓ Correct root cause identified: {winner}")
    
    return True


def main():
    tests = [
        ("Offline Pipeline", test_offline_pipeline),
        ("Logit Bias Computation", test_logit_bias_computation),
        ("Convergence Gating", test_convergence_gating),
        ("RCA Pipeline", test_rca_pipeline),
    ]
    
    print("\n" + "█" * 60)
    print("  TENSEGRITY GRAFT: Logit-Bias Integration Tests")
    print("█" * 60)
    
    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    
    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  {n_pass}/{len(results)} passed")
    return all(ok for _, ok in results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
