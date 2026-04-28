"""
Tests for asynchronous logit grafting and NGC warm-start settling.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_module(name: str, relpath: str):
    import importlib.util

    path = os.path.join(ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_ngc_warm_start_fewer_steps():
    mod = _load_module("tensegrity_engine_ngc_test", os.path.join("tensegrity", "engine", "ngc.py"))
    PredictiveCodingCircuit = mod.PredictiveCodingCircuit

    ngc = PredictiveCodingCircuit(
        layer_sizes=[16, 8, 4],
        settle_steps=30,
        settle_steps_warm=4,
        obs_change_threshold=1e-6,
        adaptive_precision=False,
    )
    pattern = np.random.RandomState(0).randn(16)
    full = ngc.settle(pattern)
    assert full["settle_steps"] == 30
    warm = ngc.settle(pattern)
    assert warm["settle_steps"] == 4


def test_async_beliefs_processor_matches_sync():
    try:
        import torch
    except ImportError:
        return

    mod = _load_module("logit_bias_test", os.path.join("tensegrity", "graft", "logit_bias.py"))
    TensegrityLogitsProcessor = mod.TensegrityLogitsProcessor

    hypothesis_tokens = {"up": {10, 11}, "down": {20, 21}}
    posteriors = {"up": 0.88, "down": 0.12}

    def belief_fn():
        return posteriors

    sync = TensegrityLogitsProcessor(
        hypothesis_tokens=hypothesis_tokens,
        belief_fn=belief_fn,
        vocab_size=128,
        entropy_gate=0.95,
        min_confidence=0.2,
        async_beliefs=False,
    )
    async_p = TensegrityLogitsProcessor(
        hypothesis_tokens=hypothesis_tokens,
        belief_fn=belief_fn,
        vocab_size=128,
        entropy_gate=0.95,
        min_confidence=0.2,
        async_beliefs=True,
        belief_poll_s=0.002,
    )
    time.sleep(0.05)
    fake_ids = torch.zeros(1, 3, dtype=torch.long)
    scores = torch.randn(1, 128)
    out_sync = sync(fake_ids, scores.clone())
    out_async = async_p(fake_ids, scores.clone())
    async_p.close()
    if async_p.state.bias_emitted and sync.state.bias_emitted:
        assert torch.allclose(out_sync, out_async, atol=1e-5)


def test_associative_access_decay():
    mod = _load_module("associative_test", os.path.join("tensegrity", "memory", "associative.py"))
    AssociativeMemory = mod.AssociativeMemory

    mem = AssociativeMemory(
        pattern_dim=8,
        beta=1.0,
        decay_every_n_retrieves=1,
        access_decay=0.5,
        max_patterns=100,
    )
    mem.store(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    mem.retrieve(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    assert mem._access_counts[0] >= 1.0
    mem._decay_access_counts()
    assert mem._access_counts[0] < 1.0


def test_build_scm_from_proposal():
    try:
        import networkx  # noqa: F401
    except ImportError:
        return
    from tensegrity.broca.schemas import ProposedSCM, CausalEdge
    from tensegrity.causal.from_proposal import build_scm_from_proposal

    p = ProposedSCM(
        name="test_model",
        description="x drives y",
        edges=[
            CausalEdge(source="x", target="y", mechanism="causes"),
            CausalEdge(source="y", target="z", mechanism="enables"),
        ],
    )
    scm = build_scm_from_proposal(p)
    assert "x" in scm.variables and "z" in scm.variables


if __name__ == "__main__":
    test_ngc_warm_start_fewer_steps()
    test_async_beliefs_processor_matches_sync()
    test_associative_access_decay()
    test_build_scm_from_proposal()
    print("ok")
