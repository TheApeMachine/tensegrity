"""
Test v2 scoring bridge against benchmarks.
"""
import sys
sys.path.insert(0, '/app')
import numpy as np
np.random.seed(42)


def test_v2_scoring():
    """Test v2 NGC-based scoring on benchmark samples."""
    print("=" * 60)
    print("TEST: v2 NGC Scoring vs v1 Baseline on Sample Tasks")
    print("=" * 60)
    
    from tensegrity.v2.graft import V2ScoringBridge
    from tensegrity.bench.tasks import load_task_samples
    
    bridge = V2ScoringBridge(obs_dim=128, hidden_dims=[64, 16])
    
    tasks = ["copa", "sciq", "arc_challenge"]
    
    for task_name in tasks:
        try:
            samples = load_task_samples(task_name, max_samples=30)
        except Exception as e:
            print(f"\n  {task_name}: SKIP ({e})")
            continue
        
        correct = 0
        total = 0
        
        for sample in samples:
            bridge.reset()
            scores, entropy = bridge.score_choices(sample.prompt, sample.choices)
            pred = int(np.argmax(scores))
            if pred == sample.gold:
                correct += 1
            total += 1
        
        acc = correct / max(total, 1)
        print(f"\n  {task_name}: {correct}/{total} = {acc:.1%}")
    
    print(f"\n  ✓ v2 scoring bridge functional")
    return True


def test_causal_energy():
    """Test the causal energy term."""
    print("\n" + "=" * 60)
    print("TEST: Causal Energy Arena v2")
    print("=" * 60)
    
    from tensegrity.causal.scm import StructuralCausalModel
    from tensegrity.v2.causal_energy import CausalArenaV2
    
    # Two competing models
    m_correct = StructuralCausalModel("correct")
    m_correct.add_variable("X", n_values=3)
    m_correct.add_variable("Y", n_values=3, parents=["X"])
    
    m_wrong = StructuralCausalModel("wrong")
    m_wrong.add_variable("X", n_values=3)
    m_wrong.add_variable("Y", n_values=3)  # No causal link
    
    # Train correct model on data where X causes Y
    data = m_correct.sample(100)
    m_correct.update_from_data(data)
    m_wrong.update_from_data(data)
    
    arena = CausalArenaV2(precision=1.0, beta=2.0)
    arena.register(m_correct)
    arena.register(m_wrong)
    
    # Test on 20 observations
    test_data = m_correct.sample(20)
    winners = []
    for obs in test_data:
        result = arena.compete(obs)
        winners.append(result["winner"])
        arena.update_models(obs)
    
    correct_wins = sum(1 for w in winners if w == "correct")
    print(f"  Correct model wins: {correct_wins}/{len(winners)}")
    print(f"  Final tension: {arena.tension:.3f}")
    
    # Energy comparison
    last_result = arena.compete(test_data[-1])
    print(f"  Last energies: {last_result['energies']}")
    print(f"  Last posteriors: {last_result['posteriors']}")
    
    print(f"  ✓ Causal energy arena functional")
    return True


if __name__ == "__main__":
    tests = [
        ("v2 Scoring", test_v2_scoring),
        ("Causal Energy", test_causal_energy),
    ]
    
    print("\n" + "█" * 60)
    print("  v2 Integration Tests")
    print("█" * 60)
    
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
    
    print()
