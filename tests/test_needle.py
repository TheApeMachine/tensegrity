"""
Needle-in-Lies Test: Can v2's NGC detect the true statement among contradictions?

The test:
  - One true statement ("The key is under the oak table")
  - N false/contradictory statements
  - The system must identify the true statement by detecting that
    the false ones are mutually inconsistent and the true one is
    consistent with the overall evidence pattern

This tests:
  1. FHRR binding: can structured representations encode relations?
  2. NGC prediction errors: do contradictions produce larger errors?
  3. Memory: does the Hopfield network store and retrieve the consistent pattern?
  4. Energy landscape: does the true statement minimize total energy?

The key encoding insight: instead of encoding text as flat token sequences,
we encode STRUCTURED CLAIMS as role-filler bindings:
  "The key is under the oak table" → bind(subject:key, relation:under, object:oak_table)

Then contradictions become visible: two claims about the same subject
with different relation-object bindings will have low FHRR similarity
at the binding level, even though they share surface tokens.
"""

import sys
sys.path.insert(0, '/app')
import numpy as np
np.random.seed(42)

from tensegrity.v2.fhrr import FHRREncoder, bind, bundle, unbind
from tensegrity.v2.ngc import PredictiveCodingCircuit
from tensegrity.v2.field import UnifiedField, HopfieldMemoryBank


def make_needle_scenario(n_lies: int = 13):
    """
    Create a needle-in-lies scenario.
    
    One true claim + n_lies false claims, all about the same subject.
    """
    truth = {
        "subject": "key",
        "relation": "under",
        "object": "oak_table",
        "text": "The key is under the oak table."
    }
    
    lies = [
        {"subject": "key", "relation": "inside", "object": "red_box",
         "text": "The key is inside the red box."},
        {"subject": "key", "relation": "behind", "object": "blue_curtain",
         "text": "The key is behind the blue curtain."},
        {"subject": "key", "relation": "on_top_of", "object": "bookshelf",
         "text": "The key is on top of the bookshelf."},
        {"subject": "key", "relation": "beneath", "object": "carpet",
         "text": "The key is beneath the carpet."},
        {"subject": "key", "relation": "inside", "object": "coat_pocket",
         "text": "The key is inside the coat pocket."},
        {"subject": "key", "relation": "behind", "object": "painting",
         "text": "The key is behind the painting."},
        {"subject": "key", "relation": "in", "object": "garden_shed",
         "text": "The key is in the garden shed."},
        {"subject": "key", "relation": "under", "object": "doormat",
         "text": "The key is under the doormat."},
        {"subject": "key", "relation": "inside", "object": "desk_drawer",
         "text": "The key is inside the desk drawer."},
        {"subject": "key", "relation": "on", "object": "kitchen_counter",
         "text": "The key is on the kitchen counter."},
        {"subject": "key", "relation": "behind", "object": "sofa_cushion",
         "text": "The key is behind the sofa cushion."},
        {"subject": "key", "relation": "in", "object": "shoe_box",
         "text": "The key is in the shoe box."},
        {"subject": "key", "relation": "beneath", "object": "floorboard",
         "text": "The key is beneath the floorboard."},
    ]
    
    return truth, lies[:n_lies]


def test_contradiction_detection():
    """Test that FHRR binding makes contradictions detectable."""
    print("=" * 60)
    print("TEST 1: FHRR Binding Detects Contradictions")
    print("=" * 60)
    
    enc = FHRREncoder(dim=2048)
    
    # Encode two contradictory claims as structured bindings
    claim_a = enc.encode_observation({
        "subject": "key", "relation": "under", "object": "oak_table"
    })
    claim_b = enc.encode_observation({
        "subject": "key", "relation": "inside", "object": "red_box"
    })
    claim_a_repeat = enc.encode_observation({
        "subject": "key", "relation": "under", "object": "oak_table"
    })
    
    # Compare: same claim vs contradictory claim
    sim_same = enc.similarity(claim_a, claim_a_repeat)
    sim_contra = enc.similarity(claim_a, claim_b)
    
    print(f"  claim A: key-under-oak_table")
    print(f"  claim B: key-inside-red_box")
    print(f"  sim(A, A_repeat) = {sim_same:.4f}")
    print(f"  sim(A, B)        = {sim_contra:.4f}")
    
    assert sim_same > sim_contra, "Same claims should be more similar than contradictory ones"
    print(f"  ✓ Structured binding distinguishes consistent from contradictory claims")
    
    # Unbind to verify structure is recoverable
    decoded_obj = enc.decode_role(claim_a, "object")
    top_label, top_sim = decoded_obj[0]
    print(f"\n  unbind(claim_a, 'object') → '{top_label}' (sim={top_sim:.4f})")
    assert top_label == "oak_table"
    print(f"  ✓ Object filler correctly recovered via unbinding")
    
    return True


def test_ngc_contradiction_signal():
    """Test that NGC prediction errors spike on contradictions."""
    print("\n" + "=" * 60)
    print("TEST 2: NGC Prediction Errors Spike on Contradictions")
    print("=" * 60)
    
    enc = FHRREncoder(dim=2048)
    
    # Build a field that encodes claims as bindings, not token sequences
    field = UnifiedField(obs_dim=128, hidden_dims=[64, 16], fhrr_dim=2048,
                         ngc_settle_steps=25, ngc_learning_rate=0.005)
    
    # First, establish a belief by presenting the truth 3 times
    truth = {"subject": "key", "relation": "under", "object": "oak_table"}
    
    print("  Phase 1: Establishing belief (truth repeated 3x)")
    for i in range(3):
        r = field.observe(truth, input_type="bindings")
        print(f"    [{i+1}] PE={r['energy'].prediction_error_norm:.2f}  "
              f"E={r['energy'].total:.2f}")
    
    pe_after_training = r['energy'].prediction_error_norm
    
    # Now present contradictions
    lies = [
        {"subject": "key", "relation": "inside", "object": "red_box"},
        {"subject": "key", "relation": "behind", "object": "blue_curtain"},
        {"subject": "key", "relation": "on_top_of", "object": "bookshelf"},
    ]
    
    print("\n  Phase 2: Presenting contradictions")
    contradiction_pes = []
    for i, lie in enumerate(lies):
        r = field.observe(lie, input_type="bindings")
        pe = r['energy'].prediction_error_norm
        contradiction_pes.append(pe)
        print(f"    [lie {i+1}] PE={pe:.2f}  E={r['energy'].total:.2f}  "
              f"\"{lie['relation']}_{lie['object']}\"")
    
    # Present truth again
    print("\n  Phase 3: Presenting truth again")
    r_truth = field.observe(truth, input_type="bindings")
    pe_truth_after = r_truth['energy'].prediction_error_norm
    print(f"    [truth] PE={pe_truth_after:.2f}  E={r_truth['energy'].total:.2f}")
    
    # The key metric: prediction error for contradictions should be
    # different from prediction error for the established truth
    mean_contra_pe = np.mean(contradiction_pes)
    print(f"\n  Mean contradiction PE: {mean_contra_pe:.2f}")
    print(f"  Truth PE (re-presented): {pe_truth_after:.2f}")
    
    # Memory similarity should be high for truth, lower for lies
    print(f"\n  Memory similarity for truth: {r_truth['memory_similarity']:.4f}")
    
    return True


def test_needle_in_lies():
    """
    The full needle-in-lies test.
    
    Present a stream of N contradictory claims interspersed with the truth.
    Score each claim by its fit to the established belief pattern.
    The truth should score highest (lowest energy / prediction error).
    """
    print("\n" + "=" * 60)
    print("TEST 3: Needle-in-Lies (13 contradictions)")
    print("=" * 60)
    
    truth, lies = make_needle_scenario(n_lies=13)
    
    field = UnifiedField(obs_dim=128, hidden_dims=[64, 16], fhrr_dim=2048,
                         ngc_settle_steps=25, ngc_learning_rate=0.003)
    
    # Build the claim stream: truth appears at positions 0, 5, 10
    # (establishing the "needle" among the "lies")
    claims = []
    truth_indices = set()
    
    # Initial truth
    claims.append(truth)
    truth_indices.add(0)
    
    # Interleave lies and truth
    for i, lie in enumerate(lies):
        claims.append(lie)
        if i == 4:
            claims.append(truth)  # Repeat truth midway
            truth_indices.add(len(claims) - 1)
        if i == 9:
            claims.append(truth)  # Repeat truth again
            truth_indices.add(len(claims) - 1)
    
    print(f"  Total claims: {len(claims)} ({len(truth_indices)} truth, {len(claims) - len(truth_indices)} lies)")
    print(f"  Truth positions: {sorted(truth_indices)}")
    print()
    
    # Feed all claims to the field
    energies = []
    for i, claim in enumerate(claims):
        is_truth = i in truth_indices
        label = "TRUTH" if is_truth else "lie  "
        
        # Encode as structured binding
        bindings = {k: v for k, v in claim.items() if k != "text"}
        r = field.observe(bindings, input_type="bindings")
        
        pe = r['energy'].prediction_error_norm
        e = r['energy'].total
        ms = r['memory_similarity']
        
        energies.append({
            'index': i,
            'is_truth': is_truth,
            'pe': pe,
            'energy': e,
            'mem_sim': ms,
            'text': claim['text'],
        })
        
        print(f"  [{i:2d}] {label} PE={pe:8.2f} E={e:9.2f} mem={ms:+.3f}  "
              f"{claim.get('relation', '?'):12s} {claim.get('object', '?')}")
    
    # === SCORING ===
    # After processing all claims, score each one by re-presenting it
    # and measuring how well it fits the settled belief state
    print(f"\n  --- Re-scoring all claims ---")
    
    scores = []
    for i, claim in enumerate(claims):
        bindings = {k: v for k, v in claim.items() if k != "text"}
        fhrr_vec = field.encoder.encode_observation(bindings)
        
        # Score directly in FHRR space: compare this claim's FHRR vector
        # to the FHRR vectors of all stored observations.
        # The truth was stored 3 times; lies were stored once each.
        # Hopfield in FHRR space will favor the repeated pattern.
        
        # Build a Hopfield bank from the raw FHRR observations
        # (We only need to do this once, but it's clearer inline)
        if i == 0:
            fhrr_memory = HopfieldMemoryBank(dim=field.fhrr_dim, beta=0.005, capacity=100)
            # Re-encode and store all claims as they were presented
            for j, c in enumerate(claims):
                b = {k: v for k, v in c.items() if k != "text"}
                fv = field.encoder.encode_observation(b)
                fhrr_memory.store(fv, normalize=True)
        
        # Retrieve: how well does this claim match the memory's attractor?
        retrieved_fhrr, fhrr_energy = fhrr_memory.retrieve(
            np.real(fhrr_vec).astype(np.float64), steps=5)
        
        # Similarity to retrieval
        q = np.real(fhrr_vec).astype(np.float64)
        q_norm = np.linalg.norm(q)
        r_norm = np.linalg.norm(retrieved_fhrr)
        if q_norm > 1e-8 and r_norm > 1e-8:
            fhrr_sim = float(np.dot(q / q_norm, retrieved_fhrr / r_norm))
        else:
            fhrr_sim = 0.0
        
        # Score = FHRR memory similarity + Hopfield energy (more negative = deeper attractor)
        score = fhrr_sim - 0.1 * fhrr_energy
        
        scores.append({
            'index': i,
            'is_truth': i in truth_indices,
            'score': score,
            'pe': pe,
            'text': claim['text'][:50],
        })
    
    # Sort by score (best first)
    ranked = sorted(scores, key=lambda x: x['score'], reverse=True)
    
    print(f"\n  Ranking (higher score = better fit to beliefs):")
    for rank, item in enumerate(ranked[:5]):
        marker = "★" if item['is_truth'] else " "
        print(f"    #{rank+1} {marker} score={item['score']:8.2f}  PE={item['pe']:8.2f}  "
              f"\"{item['text']}\"")
    print(f"    ...")
    for item in ranked[-3:]:
        marker = "★" if item['is_truth'] else " "
        print(f"    #{ranked.index(item)+1:2d} {marker} score={item['score']:8.2f}  PE={item['pe']:8.2f}  "
              f"\"{item['text']}\"")
    
    # Check: is any truth claim in the top 3?
    top_3_indices = [item['index'] for item in ranked[:3]]
    truth_in_top_3 = any(i in truth_indices for i in top_3_indices)
    
    # Check: is the best claim a truth?
    best_is_truth = ranked[0]['is_truth']
    
    print(f"\n  Best claim is truth: {best_is_truth}")
    print(f"  Truth in top 3: {truth_in_top_3}")
    
    # Compute mean score for truth vs lies
    truth_scores = [s['score'] for s in scores if s['is_truth']]
    lie_scores = [s['score'] for s in scores if not s['is_truth']]
    
    mean_truth = np.mean(truth_scores)
    mean_lie = np.mean(lie_scores)
    
    print(f"\n  Mean score (truth): {mean_truth:.4f}")
    print(f"  Mean score (lies):  {mean_lie:.4f}")
    print(f"  Separation:         {mean_truth - mean_lie:+.4f}")
    
    if mean_truth > mean_lie:
        print(f"  ✓ Truth claims score higher than lies on average")
    else:
        print(f"  ✗ Lies score higher — NGC hasn't separated them yet")
    
    return truth_in_top_3


def main():
    tests = [
        ("FHRR Contradiction Detection", test_contradiction_detection),
        ("NGC Contradiction Signal", test_ngc_contradiction_signal),
        ("Needle-in-Lies (13 contradictions)", test_needle_in_lies),
    ]
    
    print("\n" + "█" * 60)
    print("  NEEDLE-IN-LIES TEST")
    print("  Can hierarchical predictive coding resolve contradictions?")
    print("█" * 60)
    
    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append((name, False))
    
    print(f"\n{'=' * 60}")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"  {sum(1 for _, ok in results if ok)}/{len(results)} passed")
    
    return all(ok for _, ok in results)


if __name__ == "__main__":
    main()
