"""
Tests for asynchronous logit grafting and NGC warm-start settling.
"""

import os
import sys
import time

import numpy as np
import pytest

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


def test_scm_marginalizes_missing_parents_and_counterfactual_changes_descendants():
    pytest.importorskip("networkx")
    from tensegrity.causal.scm import StructuralCausalModel

    scm = StructuralCausalModel("two_node")
    scm.add_variable("X", n_values=2)
    scm.add_variable("Y", n_values=2, parents=["X"])
    scm.mechanisms["X"].cpt_params = np.array([[100.0], [1.0]])
    scm.mechanisms["Y"].cpt_params = np.array([
        [100.0, 1.0],
        [1.0, 100.0],
    ])

    p_x0 = 100.0 / 101.0
    p_x1 = 1.0 / 101.0
    expected_p_y1 = p_x0 * (1.0 / 101.0) + p_x1 * (100.0 / 101.0)
    assert np.isclose(
        scm.observe({"Y": 1})["log_likelihood"],
        np.log(expected_p_y1),
    )

    cf = scm.counterfactual({"X": 0, "Y": 0}, {"X": 1}, ["Y"])
    assert cf["Y"][1] > 0.98
    assert cf["Y"][0] < 0.02


def test_eliminated_hypothesis_stays_eliminated_after_engine_blend():
    from tensegrity.broca.controller import CognitiveController

    controller = CognitiveController(
        n_hypotheses=2,
        hypothesis_labels=["a", "b"],
        use_llm=False,
    )
    controller.belief_state.hypotheses[0].probability = 0.0
    controller.belief_state.hypotheses[1].probability = 1.0
    controller.belief_state.eliminated_hypotheses = ["H0"]

    controller._update_hypotheses_from_inference(
        {
            "belief_state": np.array([1.0, 0.0]),
            "arena": {"tension": 0.5},
            "free_energy": 0.0,
        },
        np.array([0.0, 0.0]),
    )

    assert controller.belief_state.hypotheses[0].probability == 0.0
    assert controller.belief_state.hypotheses[1].probability == 1.0
    assert controller.belief_state.eliminated_hypotheses == ["H0"]


def test_canonical_soft_reset_preserves_ngc_weights():
    from tensegrity.pipeline.canonical import CanonicalPipeline

    pipeline = CanonicalPipeline(["a", "b"])
    before = [w.copy() for w in pipeline.controller.agent.field.ngc.W]
    pipeline._soft_reset_in_place(["a", "b"])

    for old, new in zip(before, pipeline.controller.agent.field.ngc.W):
        assert np.allclose(old, new)


def test_canonical_feedback_persists_memory_and_reload(tmp_path):
    from tensegrity.bench.tasks import TaskSample
    from tensegrity.pipeline.canonical import CanonicalPipeline

    state_path = tmp_path / "agent_state.pkl"
    sample = TaskSample(
        id="demo_1",
        prompt="A coolant leak causes the engine to overheat.",
        choices=["low oil", "coolant leak"],
        gold=1,
        metadata={"task": "demo"},
    )

    pipeline = CanonicalPipeline(
        ["_empty_0", "_empty_1"],
        persistent_state_path=str(state_path),
    )
    before = len(pipeline.controller.agent.episodic.episodes)
    pipeline.learn_from_feedback(sample, committed_idx=0)

    assert state_path.exists()
    assert len(pipeline.controller.agent.episodic.episodes) > before

    reloaded = CanonicalPipeline(
        ["_empty_0", "_empty_1"],
        persistent_state_path=str(state_path),
    )
    assert len(reloaded.controller.agent.episodic.episodes) > 0
    memory_scores = reloaded._memory_choice_scores(sample)
    assert memory_scores.shape == (2,)
    assert np.all(np.isfinite(memory_scores))


def test_canonical_integrates_linguistic_scores_inside_posterior():
    from tensegrity.bench.tasks import TaskSample
    from tensegrity.pipeline.canonical import CanonicalPipeline

    sample = TaskSample(
        id="linguistic_1",
        prompt="Pick the candidate supported by the language evidence.",
        choices=["alpha", "beta"],
        gold=1,
    )
    pipeline = CanonicalPipeline(
        ["_empty_0", "_empty_1"],
        max_iterations=1,
        llm_evidence_weight=10.0,
    )
    result = pipeline.score_multichoice(sample, linguistic_scores=[-5.0, 5.0])
    assert result.committed_idx == 1
    assert result.linguistic_scores == [-5.0, 5.0]


def test_canonical_benchmark_path_uses_broca_transducer():
    from tensegrity.bench.tasks import TaskSample
    from tensegrity.broca.schemas import ParsedObservation
    from tensegrity.pipeline.canonical import CanonicalPipeline

    class FakeBroca:
        def __init__(self):
            self.parse_calls = 0

        def parse(self, text, context=None):
            self.parse_calls += 1
            return ParsedObservation(
                entities=[],
                relations=[],
                implicit_relations=[],
                is_question=text.strip().endswith("?"),
                is_assertion=not text.strip().endswith("?"),
                is_command=False,
                negation_present=False,
                temporal_marker=None,
                confidence_linguistic=0.5,
            )

    broca = FakeBroca()
    sample = TaskSample(
        id="broca_1",
        prompt="Which candidate is supported?",
        choices=["alpha", "beta"],
        gold=0,
    )
    pipeline = CanonicalPipeline(
        ["_empty_0", "_empty_1"],
        broca=broca,
        use_llm_broca=True,
        max_iterations=1,
    )
    pipeline.score_multichoice(sample, linguistic_scores=[1.0, 0.0])
    assert broca.parse_calls > 0


def test_deterministic_broca_proposes_arena_compatible_scm():
    from tensegrity.broca.interface import DeterministicBrocaInterface
    from tensegrity.causal.from_proposal import build_scm_from_proposal

    broca = DeterministicBrocaInterface()
    proposal = broca.propose_causal_hypothesis(
        "A blocked valve prevents pressure from reaching the chamber.",
        ["direct_causal", "mediated_causal", "broca_contextual_causal"],
    )
    scm = build_scm_from_proposal(proposal)

    assert proposal.name not in {"direct_causal", "mediated_causal", "broca_contextual_causal"}
    assert "state" in scm.variables
    assert "observation" in scm.variables
    assert ("state", "observation") in scm.edges


def test_runner_live_emission_choice_constraints():
    from tensegrity.bench.runner import EvalRunner

    seqs = {
        0: [[11, 12], [9, 11, 12]],
        1: [[21]],
    }
    grounding = EvalRunner._choice_token_grounding(seqs)
    assert grounding == {"H0": {9, 11, 12}, "H1": {21}}

    allowed = EvalRunner._build_prefix_allowed_fn(
        prompt_len=3,
        choice_sequences=seqs,
        eos_token_id=99,
    )
    assert set(allowed(0, [1, 2, 3])) == {9, 11, 21}
    assert allowed(0, [1, 2, 3, 11]) == [12]
    assert allowed(0, [1, 2, 3, 11, 12]) == [99]
    assert EvalRunner._parse_emitted_choice([11, 12, 99], {(11, 12): 0, (21,): 1}, 99) == 0


def test_runner_local_emission_verbalizes_committed_choice_under_graft():
    import torch
    from types import SimpleNamespace
    from tensegrity.bench.runner import EvalRunner
    from tensegrity.bench.tasks import TaskSample

    class FakeTokenizer:
        vocab_size = 128
        eos_token_id = 99
        pad_token_id = 99

        def __len__(self):
            return self.vocab_size

        def encode(self, text, add_special_tokens=False):
            stripped = text.strip()
            prefix = []
            if text.startswith(" "):
                prefix = [10]
            elif text.startswith("\n"):
                prefix = [13]
            if stripped == "alpha":
                return prefix + [11]
            if stripped == "beta":
                return prefix + [21]
            return [1]

        def apply_chat_template(self, messages, return_tensors="pt", add_generation_prompt=True):
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

        def decode(self, ids, skip_special_tokens=True):
            pieces = []
            for token_id in [int(t) for t in ids]:
                if token_id == 99 and skip_special_tokens:
                    continue
                if token_id in {10, 13}:
                    continue
                if token_id == 11:
                    pieces.append("alpha")
                elif token_id == 21:
                    pieces.append("beta")
            return "".join(pieces)

    class FakeModel:
        device = torch.device("cpu")

        def generate(
            self,
            input_ids,
            attention_mask,
            logits_processor,
            prefix_allowed_tokens_fn,
            max_new_tokens,
            do_sample,
            pad_token_id,
            eos_token_id,
        ):
            assert attention_mask is not None
            cur = input_ids.clone()
            vocab = 140
            for _ in range(max_new_tokens):
                allowed = prefix_allowed_tokens_fn(0, cur[0])
                scores = torch.full((1, vocab), -1e9, dtype=torch.float32)
                for token_id in allowed:
                    scores[0, int(token_id)] = 0.0
                scores = logits_processor(cur, scores)
                next_id = int(torch.argmax(scores[0]).item())
                cur = torch.cat([cur, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
                if next_id == eos_token_id:
                    break
            return cur

    runner = EvalRunner(mode="local", lam=1.0)
    runner._tokenizer = FakeTokenizer()
    runner._model = FakeModel()
    sample = TaskSample(
        id="emit_1",
        prompt="Choose the supported answer.",
        choices=["alpha", "beta"],
        gold=1,
    )
    commit = SimpleNamespace(committed_idx=1, belief=[0.1, 0.9])
    pred, text, mode, graft_state = runner._emit_answer(sample, commit)

    assert pred == 1
    assert text == "beta"
    assert mode == "logit_grafted_verbalizer"
    assert graft_state["bias_emitted"] is True


if __name__ == "__main__":
    test_ngc_warm_start_fewer_steps()
    test_async_beliefs_processor_matches_sync()
    test_associative_access_decay()
    test_build_scm_from_proposal()
    test_scm_marginalizes_missing_parents_and_counterfactual_changes_descendants()
    test_eliminated_hypothesis_stays_eliminated_after_engine_blend()
    test_canonical_soft_reset_preserves_ngc_weights()
    print("ok")
