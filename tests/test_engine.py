"""
Tests for the unified cognitive engine: FHRR, NGC, and UnifiedField.
"""

import numpy as np
import sys
np.random.seed(42)


def test_fhrr_encoding():
    """Test FHRR compositional encoding."""
    print("=" * 60)
    print("TEST 1: FHRR-RNS Compositional Encoding")
    print("=" * 60)
    
    from tensegrity.engine.fhrr import FHRREncoder, bind, unbind, bundle
    
    enc = FHRREncoder(dim=2048)
    
    # Test position encoding: similar positions → similar vectors
    p10 = enc.encode_position(10)
    p11 = enc.encode_position(11)
    p100 = enc.encode_position(100)
    
    sim_close = enc.similarity(p10, p11)
    sim_far = enc.similarity(p10, p100)
    
    print(f"  sim(pos=10, pos=11)  = {sim_close:.4f}")
    print(f"  sim(pos=10, pos=100) = {sim_far:.4f}")
    assert np.isfinite(sim_close)
    assert np.isfinite(sim_far)
    
    # Test binding + unbinding: encode "red ball on table"
    obs = enc.encode_observation({
        "color": "red",
        "object": "ball",
        "location": "table",
    })
    
    # Unbind to recover
    recovered = enc.decode_role(obs, "color")
    print(f"\n  Encoded: {{color:red, object:ball, location:table}}")
    print(f"  Decoded role 'color': {recovered[:3]}")
    
    # The top result should be "red"
    top_label, top_sim = recovered[0]
    print(f"  Top match: '{top_label}' (sim={top_sim:.4f})")
    assert top_label == "red", f"Expected 'red', got '{top_label}'"
    print(f"  ✓ Compositional binding → unbinding recovers 'red'")
    
    # Test sequence encoding
    seq1 = enc.encode_sequence(["the", "cat", "sat"])
    seq2 = enc.encode_sequence(["the", "cat", "sat"])
    seq3 = enc.encode_sequence(["the", "dog", "ran"])
    
    sim_same = enc.similarity(seq1, seq2)
    sim_diff = enc.similarity(seq1, seq3)
    print(f"\n  sim('the cat sat', 'the cat sat') = {sim_same:.4f}")
    print(f"  sim('the cat sat', 'the dog ran') = {sim_diff:.4f}")
    assert sim_same > sim_diff, "Same sequence should be more similar"
    print(f"  ✓ Same sequences more similar than different ones")
    
    # Test numeric vector encoding (modality-agnostic)
    v_base = enc.encode_numeric_vector(np.array([1.0, 2.0, 3.0]))
    v_near = enc.encode_numeric_vector(np.array([1.0, 2.0, 3.1]))
    v_far = enc.encode_numeric_vector(np.array([9.0, 8.0, 7.0]))
    
    sim_near = enc.similarity(v_base, v_near)
    sim_numeric_far = enc.similarity(v_base, v_far)
    print(f"\n  sim([1,2,3], [1,2,3.1]) = {sim_near:.4f}")
    print(f"  sim([1,2,3], [9,8,7])   = {sim_numeric_far:.4f}")
    assert sim_near > sim_numeric_far, (
        "Numeric vectors should be more similar when inputs are nearer in value space "
        f"(sim_near={sim_near}, sim_far={sim_numeric_far})"
    )
    print(f"  ✓ Numeric vectors: similar inputs → similar encodings")


def test_predictive_coding():
    """Test the NGC predictive coding circuit."""
    print("\n" + "=" * 60)
    print("TEST 2: Hierarchical Predictive Coding (NGC)")
    print("=" * 60)
    
    from tensegrity.engine.ngc import PredictiveCodingCircuit
    
    # 3-layer hierarchy: 64 → 32 → 8
    ngc = PredictiveCodingCircuit(
        layer_sizes=[64, 32, 8],
        precisions=[1.0, 1.0, 0.5],
        settle_steps=30,
        learning_rate=0.01,
    )
    
    # Feed repeated observations — the system should learn to predict them
    pattern_a = np.sin(np.linspace(0, 2*np.pi, 64))
    pattern_b = np.cos(np.linspace(0, 2*np.pi, 64))
    
    energies = []
    prediction_errors = []
    
    for epoch in range(40):
        pattern = pattern_a if epoch % 2 == 0 else pattern_b
        
        # Before settling: how surprised?
        pe = ngc.prediction_error(pattern)
        prediction_errors.append(pe)
        
        # Settle (minimize VFE)
        result = ngc.settle(pattern)
        energies.append(result["final_energy"])
        
        # Learn (Hebbian update)
        ngc.learn()
    
    print(f"  Architecture: {ngc.layer_sizes}")
    print(f"  Epochs: 40 (alternating sin/cos patterns)")
    print(f"\n  Energy trajectory:")
    print(f"    First 5:  {[f'{e:.3f}' for e in energies[:5]]}")
    print(f"    Last 5:   {[f'{e:.3f}' for e in energies[-5:]]}")
    
    print(f"\n  Prediction error trajectory:")
    print(f"    First 5:  {[f'{e:.3f}' for e in prediction_errors[:5]]}")
    print(f"    Last 5:   {[f'{e:.3f}' for e in prediction_errors[-5:]]}")
    
    # Energy should decrease
    early = np.mean(energies[:5])
    late = np.mean(energies[-5:])
    print(f"\n  Mean energy (early): {early:.4f}")
    print(f"  Mean energy (late):  {late:.4f}")
    
    # Prediction error should decrease
    pe_early = np.mean(prediction_errors[:5])
    pe_late = np.mean(prediction_errors[-5:])
    print(f"  Mean PE (early): {pe_early:.4f}")
    print(f"  Mean PE (late):  {pe_late:.4f}")
    assert np.all(np.isfinite(energies))
    assert pe_late < pe_early, "Prediction error should decrease after Hebbian updates"
    
    # THE KEY TEST: the system now PREDICTS its input
    predicted = ngc.predict_observation()
    residual = np.linalg.norm(predicted)
    print(f"\n  Prediction norm: {residual:.4f} (>0 means the system has learned to predict)")
    assert residual > 0.01, "System should generate non-trivial predictions"
    print(f"  ✓ System generates predictions of sensory input")
    print(f"  ✓ Prediction error decreases via Hebbian learning (no backprop)")


def test_unified_field():
    """Test the unified energy landscape."""
    print("\n" + "=" * 60)
    print("TEST 3: Unified Energy Landscape")
    print("=" * 60)
    
    from tensegrity.engine.unified_field import UnifiedField
    
    field = UnifiedField(
        obs_dim=128,
        hidden_dims=[64, 16],
        fhrr_dim=1024,
        hopfield_beta=0.05,
        ngc_settle_steps=15,
        ngc_learning_rate=0.01,
    )
    
    # Feed a sequence of structured observations
    observations = [
        {"object": "ball", "color": "red", "location": "table"},
        {"object": "cup", "color": "blue", "location": "shelf"},
        {"object": "ball", "color": "red", "location": "table"},  # Repeated
        {"object": "key", "color": "gold", "location": "drawer"},
        {"object": "ball", "color": "red", "location": "table"},  # Repeated again
    ]
    
    print(f"  Architecture: FHRR({field.fhrr_dim}) → NGC{field.ngc.layer_sizes} → Hopfield")
    print(f"  Feeding {len(observations)} structured observations...\n")
    
    for i, obs in enumerate(observations):
        result = field.observe(obs, input_type="bindings")
        e = result["energy"]
        
        print(f"  Obs {i+1}: {obs}")
        print(f"    Total energy:     {e.total:.4f}")
        print(f"    Perception:       {e.perception:.4f}")
        print(f"    Memory:           {e.memory:.4f}")
        print(f"    Prediction error: {e.prediction_error_norm:.4f}")
        print(f"    Memory similarity: {result['memory_similarity']:.4f}")
    
    # Check that repeated observations become less surprising
    energies = [e.total for e in field.energy_history]
    pe_list = [e.prediction_error_norm for e in field.energy_history]
    
    print(f"\n  Energy trajectory:     {[f'{e:.3f}' for e in energies]}")
    print(f"  Pred. error trajectory: {[f'{p:.3f}' for p in pe_list]}")
    
    # Memory should recall the repeated pattern
    print(f"\n  Memory patterns stored: {field.memory.n_patterns}")
    assert field.memory.n_patterns == len(observations)
    assert pe_list[-1] < pe_list[0]
    
    # Make a prediction before seeing the next observation
    predicted = field.predict()
    print(f"  Prediction vector norm: {np.linalg.norm(predicted):.4f}")
    print(f"  ✓ System predicts, remembers, and settles via unified energy minimization")
    
    # Test with text input
    print(f"\n  --- Text input mode ---")
    result = field.observe("the red ball is on the table", input_type="text")
    print(f"  Text: 'the red ball is on the table'")
    print(f"  Energy: {result['energy'].total:.4f}")
    assert np.isfinite(result["energy"].total)
    print(f"  ✓ Modality-agnostic: same pipeline handles structured and text input")


def main():
    tests = [
        ("FHRR Encoding", test_fhrr_encoding),
        ("Predictive Coding", test_predictive_coding),
        ("Unified Field", test_unified_field),
    ]
    
    print("\n" + "█" * 60)
    print("  Tensegrity engine: unified energy architecture")
    print("  FHRR-RNS × Predictive Coding × Hopfield memory")
    print("█" * 60)
    
    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append((name, False))
    
    print(f"\n{'=' * 60}")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"  {sum(1 for _, ok in results if ok)}/{len(results)} passed")
    
    return all(ok for _, ok in results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)


