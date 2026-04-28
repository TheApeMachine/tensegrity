"""
Build a StructuralCausalModel from a structured LLM proposal (ProposedSCM).
"""

from __future__ import annotations

import logging
import networkx as nx

from tensegrity.broca.schemas import ProposedSCM
from tensegrity.causal.scm import StructuralCausalModel

logger = logging.getLogger(__name__)


def build_scm_from_proposal(proposal: ProposedSCM, n_values: int = 4) -> StructuralCausalModel:
    """
    Convert a ProposedSCM into a StructuralCausalModel with discrete variables.
    Edges from ``proposal.edges`` are applied in **forward list order**. Each edge is
    kept unless adding it breaks acyclicity; if ``G.add_edge`` would introduce a cycle,
    that edge is dropped (``G.remove_edge``) and a debug log is emitted — earlier edges
    are never removed. Variable order follows a topological sort when the retained graph is non-empty.
    """
    if n_values <= 0:
        raise ValueError(f"n_values must be a positive integer, got {n_values}")

    G = nx.DiGraph()

    for e in proposal.edges:
        G.add_edge(e.source.strip(), e.target.strip())

    if G.number_of_nodes() == 0:
        logger.warning("ProposedSCM '%s' has no edges; returning empty SCM", proposal.name)
        scm = StructuralCausalModel(name=proposal.name[:60].replace(" ", "_"))
        scm.add_variable("observation", n_values=n_values, parents=[])
        return scm

    if not nx.is_directed_acyclic_graph(G):
        # Forward-order greedy cycle breaking: keep earlier edges_in_list, drop later ones that close a cycle.
        edges_list = [(e.source.strip(), e.target.strip()) for e in proposal.edges]
        G.clear()

        for s, t in edges_list:
            G.add_edge(s, t)

            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(s, t)
                logger.debug("Dropped cyclic edge %s -> %s", s, t)

    order = list(nx.topological_sort(G))
    scm = StructuralCausalModel(name=proposal.name[:60].replace(" ", "_"))

    for node in order:
        parents = sorted(G.predecessors(node))
        scm.add_variable(node, n_values=n_values, parents=parents)

    return scm
