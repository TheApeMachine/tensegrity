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
    Drops edges that would create cycles. Variable order follows a topological sort.
    """
    G = nx.DiGraph()
    for e in proposal.edges:
        G.add_edge(e.source.strip(), e.target.strip())

    if G.number_of_nodes() == 0:
        logger.warning("ProposedSCM '%s' has no edges; returning empty SCM", proposal.name)
        scm = StructuralCausalModel(name=proposal.name[:60])
        scm.add_variable("observation", n_values=n_values, parents=[])
        return scm

    if not nx.is_directed_acyclic_graph(G):
        # Greedily remove edges that introduce cycles (reverse insertion order)
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
