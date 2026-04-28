import numpy as np
import networkx as nx


class FakeTokenizer:
    def __init__(self):
        self._vocab = {
            "dependency": 0,
            "crash": 1,
            "library": 2,
            "folded": 3,
            "pressure": 4,
            "cat": 5,
        }
        self.vocab_size = len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        return [self._vocab[t] for t in text.strip().split() if t in self._vocab]


def test_semantic_projection_grounding_captures_nonliteral_tokens():
    from tensegrity.graft.vocabulary import VocabularyGrounding

    vectors = {
        "dependency crash": np.array([1.0, 0.0]),
        "dependency": np.array([0.98, 0.02]),
        "crash": np.array([0.95, 0.05]),
        "library": np.array([0.92, 0.08]),
        "folded": np.array([0.88, 0.12]),
        "pressure": np.array([0.84, 0.16]),
        "cat": np.array([0.0, 1.0]),
    }

    def embed(text):
        return vectors.get(text, np.array([0.0, 1.0]))

    grounding = VocabularyGrounding.from_semantic_projection(
        {"dependency_crash": ["dependency crash"]},
        FakeTokenizer(),
        embedding_fn=embed,
        top_k=5,
        threshold=0.7,
    )

    token_ids = grounding.get_token_ids("dependency_crash")
    token_scores = grounding.get_token_scores("dependency_crash")

    assert 2 in token_ids  # "library"
    assert 3 in token_ids  # "folded"
    assert token_scores[2] > 0.7
    assert 5 not in token_ids  # unrelated "cat"


def test_topology_mapper_turns_horizontal_edge_into_virtual_parent():
    from tensegrity.engine.causal_energy import TopologyMapper

    dag = nx.DiGraph()
    dag.add_edge("A", "B")

    mapping = TopologyMapper().project_graph(
        dag,
        variable_layers={"A": 0, "B": 0},
    )

    assert len(mapping.virtual_parents) == 1
    vp = next(iter(mapping.virtual_parents.values()))
    assert set(vp.children) == {"A", "B"}
    assert mapping.embedded_layers[vp.name] == 1
    assert (vp.name, "A") in mapping.embedded_edges
    assert (vp.name, "B") in mapping.embedded_edges
    assert ("A", "B") not in mapping.embedded_edges


def test_topology_mapper_inserts_relay_nodes_for_bypass_edge():
    from tensegrity.engine.causal_energy import TopologyMapper

    dag = nx.DiGraph()
    dag.add_edge("root", "leaf")

    mapping = TopologyMapper().project_graph(
        dag,
        variable_layers={"root": 3, "leaf": 0},
    )

    relay_layers = sorted(mapping.embedded_layers[n] for n in mapping.relay_nodes)
    assert relay_layers == [1, 2]
    assert mapping.ngc_layer_sizes() == [1, 1, 1, 1]


def test_topology_mapper_keeps_virtual_parent_edges_adjacent():
    from tensegrity.engine.causal_energy import TopologyMapper

    dag = nx.DiGraph()
    dag.add_edge("low", "high")

    mapping = TopologyMapper().project_graph(
        dag,
        variable_layers={"low": 0, "high": 2},
    )

    assert len(mapping.virtual_parents) == 1
    for parent, child in mapping.embedded_edges:
        assert mapping.embedded_layers[parent] == mapping.embedded_layers[child] + 1
