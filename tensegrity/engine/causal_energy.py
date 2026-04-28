"""
Causal Energy: Pearl's SCMs as energy terms in the unified landscape.

Each SCM contributes a prediction error to the total energy:
    E_causal(M_k) = Σ_v ||z_v - f_v(z_pa(v))||²

Where:
    z_v = observed value of variable v
    f_v(z_pa(v)) = structural equation's prediction from parents
    pa(v) = parents of v in the causal DAG

Multiple SCMs compete. The model with lowest causal energy provides
the best explanation. This complements the log-likelihood CausalArena in ``tensegrity.causal.arena``
when an energy-based readout of SCM fit is required.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from tensegrity.causal.scm import StructuralCausalModel


def _require_parent_observations(
    observations: Dict[str, int],
    parents: List[str],
    mechanism_name: str,
) -> Dict[str, int]:
    """Build parent_vals from observations or raise if any parent is missing."""
    parent_vals: Dict[str, int] = {}
    missing = [p for p in parents if p not in observations]

    if missing:
        raise ValueError(
            f"CausalEnergyTerm: mechanism '{mechanism_name}' requires parent values "
            f"{parents}; missing observations for {missing}"
        )
    
    for p in parents:
        parent_vals[p] = observations[p]
    
    return parent_vals


def _normalized_entropy_tension_from_energy_values(vals: np.ndarray, beta: float) -> float:
    """
    Map raw per-model energies to softmax weights over ``-beta * energy`` and return
    normalized entropy in [0, 1] (same convention as ``EnergyCausalArena.compete`` / ``tension``).
    """
    if vals.size == 0:
        return 1.0
    
    neg_e = -beta * vals
    neg_e -= neg_e.max()
    w = np.exp(neg_e)
    s = w.sum()
    
    if s <= 0:
        return 1.0
    
    w = w / s
    w = w[w > 0]
    
    if len(w) > 1:
        return float(-np.sum(w * np.log(w)) / np.log(len(w)))
    
    return 0.0


@dataclass(frozen=True)
class VirtualParent:
    """Virtual abstract node that turns a same-layer causal edge into top-down structure."""

    name: str
    source: str
    target: str
    layer: int
    children: Tuple[str, str]


@dataclass
class TopologyMapping:
    """
    Projection of a Pearl-style causal DAG into an NGC-compatible hierarchy.

    Layers follow the predictive-coding convention used by ``PredictiveCodingCircuit``:
    layer 0 is sensory/observed, and larger layer indices are more abstract. The
    embedded graph contains only top-down adjacent or shared-parent structure.
    """

    variable_layers: Dict[str, int]
    embedded_layers: Dict[str, int]
    embedded_edges: List[Tuple[str, str]]
    direct_edges: List[Tuple[str, str]] = field(default_factory=list)
    relay_nodes: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    virtual_parents: Dict[str, VirtualParent] = field(default_factory=dict)
    original_edges: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def layer_nodes(self) -> Dict[int, List[str]]:
        layers: Dict[int, List[str]] = {}
    
        for node, layer in self.embedded_layers.items():
            layers.setdefault(layer, []).append(node)
    
        for nodes in layers.values():
            nodes.sort()
    
        return dict(sorted(layers.items()))

    def ngc_layer_sizes(self, min_width: int = 1) -> List[int]:
        """Return layer widths from sensory layer 0 up to the top abstract layer."""
        max_layer = max(self.embedded_layers.values(), default=0)
        by_layer = self.layer_nodes
        return [max(min_width, len(by_layer.get(layer, []))) for layer in range(max_layer + 1)]

    def adjacent_edge_masks(self) -> Dict[Tuple[int, int], List[Tuple[str, str]]]:
        """
        Group embedded edges by adjacent NGC layers.

        Keys are ``(parent_layer, child_layer)``. Edges that still span more than
        one layer indicate an invalid mapping and should not be used for direct
        predictive-coding wiring.
        """
        grouped: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}
    
        for parent, child in self.embedded_edges:
            key = (self.embedded_layers[parent], self.embedded_layers[child])
            grouped.setdefault(key, []).append((parent, child))
    
        return grouped

    def as_dict(self) -> Dict[str, Any]:
        """Serializable diagnostics for documentation and tests."""
        return {
            "variable_layers": dict(self.variable_layers),
            "embedded_layers": dict(self.embedded_layers),
            "embedded_edges": list(self.embedded_edges),
            "direct_edges": list(self.direct_edges),
            "relay_nodes": dict(self.relay_nodes),
            "virtual_parents": {
                name: {
                    "source": vp.source,
                    "target": vp.target,
                    "layer": vp.layer,
                    "children": list(vp.children),
                }
                for name, vp in self.virtual_parents.items()
            },
            "original_edges": list(self.original_edges),
            "ngc_layer_sizes": self.ngc_layer_sizes(),
        }


class TopologyMapper:
    """
    Embed arbitrary SCM DAG topology into hierarchical predictive-coding wiring.

    The mapper makes the Friston/Pearl handshake explicit:

    * A causal edge from layer k to k-1 becomes a direct top-down prediction.
    * A bypass edge spanning multiple layers receives relay nodes, one per
      skipped layer.
    * A same-layer or inverted edge is encoded by a virtual parent node one
      layer above the endpoints. The virtual parent governs both variables,
      turning a lateral causal dependency into a shared vertical dependency.
    """

    def __init__(self, expand_layers: bool = True):
        self.expand_layers = bool(expand_layers)

    def from_scm(
        self,
        scm: StructuralCausalModel,
        n_layers: Optional[int] = None,
        variable_layers: Optional[Dict[str, int]] = None,
    ) -> TopologyMapping:
        """Project an SCM's DAG into an NGC hierarchy."""
        return self.project_graph(scm.graph, n_layers=n_layers, variable_layers=variable_layers)

    def project_graph(
        self,
        graph: nx.DiGraph,
        n_layers: Optional[int] = None,
        variable_layers: Optional[Dict[str, int]] = None,
    ) -> TopologyMapping:
        """
        Project a directed acyclic graph into NGC-compatible layers.

        If ``variable_layers`` is omitted, variables are placed by causal depth:
        leaves/effects at layer 0, their parents at layer 1, and so on. If
        ``n_layers`` is smaller than the DAG depth, depths are compressed into
        the available hierarchy; any resulting horizontal or bypass edges are
        repaired explicitly.
        """
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("TopologyMapper requires a directed acyclic graph")

        nodes = [str(n) for n in graph.nodes()]
        edges = [(str(u), str(v)) for u, v in graph.edges()]

        if variable_layers is None:
            variable_layers = self._layers_by_causal_depth(graph)
        else:
            missing = set(nodes) - set(variable_layers)
            if missing:
                raise ValueError(f"missing layer assignments for variables: {sorted(missing)}")
    
            variable_layers = {str(k): int(v) for k, v in variable_layers.items()}

        if n_layers is not None:
            variable_layers = self._compress_layers(variable_layers, n_layers)

        for node, layer in variable_layers.items():
            if layer < 0:
                raise ValueError(f"layer for {node!r} must be >= 0, got {layer}")

        embedded_layers = dict(variable_layers)
        embedded_edges: List[Tuple[str, str]] = []
        direct_edges: List[Tuple[str, str]] = []
        relay_nodes: Dict[str, Tuple[str, str]] = {}
        virtual_parents: Dict[str, VirtualParent] = {}

        for source, target in edges:
            source_layer = embedded_layers[source]
            target_layer = embedded_layers[target]

            if source_layer == target_layer + 1:
                embedded_edges.append((source, target))
                direct_edges.append((source, target))
                continue

            if source_layer > target_layer + 1:
                self._add_vertical_path(
                    source,
                    target,
                    source,
                    target,
                    embedded_layers,
                    embedded_edges,
                    relay_nodes,
                )
                continue

            virtual_layer = max(source_layer, target_layer) + 1
            
            if n_layers is not None and virtual_layer >= n_layers and not self.expand_layers:
                raise ValueError(
                    f"edge {source!r}->{target!r} needs virtual parent layer {virtual_layer}, "
                    f"but n_layers={n_layers} and expand_layers=False"
                )

            virtual = self._unique_name(
                f"vparent__{source}__{target}",
                embedded_layers,
            )

            embedded_layers[virtual] = virtual_layer

            vp = VirtualParent(
                name=virtual,
                source=source,
                target=target,
                layer=virtual_layer,
                children=(source, target),
            )

            virtual_parents[virtual] = vp

            self._add_vertical_path(
                virtual,
                source,
                source,
                target,
                embedded_layers,
                embedded_edges,
                relay_nodes,
            )

            self._add_vertical_path(
                virtual,
                target,
                source,
                target,
                embedded_layers,
                embedded_edges,
                relay_nodes,
            )

        return TopologyMapping(
            variable_layers=variable_layers,
            embedded_layers=embedded_layers,
            embedded_edges=embedded_edges,
            direct_edges=direct_edges,
            relay_nodes=relay_nodes,
            virtual_parents=virtual_parents,
            original_edges=edges,
        )

    def _layers_by_causal_depth(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Place leaves at layer 0 and causes at higher layers by longest path to a leaf."""
        layers = {str(node): 0 for node in graph.nodes()}

        for node in reversed(list(nx.topological_sort(graph))):
            children = list(graph.successors(node))

            if children:
                layers[str(node)] = 1 + max(layers[str(child)] for child in children)

        return layers

    def _compress_layers(self, layers: Dict[str, int], n_layers: int) -> Dict[str, int]:
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")

        if n_layers == 1:
            return {node: 0 for node in layers}

        max_layer = max(layers.values(), default=0)

        if max_layer <= n_layers - 1:
            return dict(layers)

        scale = (n_layers - 1) / max_layer

        return {node: int(round(layer * scale)) for node, layer in layers.items()}

    def _unique_name(self, base: str, layers: Dict[str, int]) -> str:
        if base not in layers:
            return base

        i = 2

        while f"{base}__{i}" in layers:
            i += 1

        return f"{base}__{i}"

    def _add_vertical_path(
        self,
        parent: str,
        child: str,
        original_source: str,
        original_target: str,
        layers: Dict[str, int],
        edges: List[Tuple[str, str]],
        relay_nodes: Dict[str, Tuple[str, str]],
    ) -> None:
        """Connect parent to child with adjacent top-down edges, inserting relays as needed."""
        parent_layer = layers[parent]
        child_layer = layers[child]

        if parent_layer <= child_layer:
            raise ValueError(
                f"embedded parent {parent!r} must be above child {child!r}; "
                f"got layers {parent_layer} <= {child_layer}"
            )

        prev = parent

        for layer in range(parent_layer - 1, child_layer, -1):
            relay = self._unique_name(
                f"relay__{parent}__{child}__L{layer}",
                layers,
            )

            layers[relay] = layer
            relay_nodes[relay] = (original_source, original_target)
            edges.append((prev, relay))
            prev = relay

        edges.append((prev, child))


class CausalEnergyTerm:
    """
    Computes causal prediction error energy for an SCM.
    
    Given observations of some variables, computes how well
    the SCM's structural equations predict them.
    """
    
    def __init__(self, scm: StructuralCausalModel, precision: float = 1.0):
        self.scm = scm
        self.precision = precision
    
    def energy(self, observations: Dict[str, int]) -> float:
        """
        Compute causal prediction error energy.
        
        E = Σ_v (1/2σ²) ||obs_v - predicted_v||²
        
        Where predicted_v = E[V | parents of V observed]
        """
        total_energy = 0.0
        order = self.scm.topological_order()
        
        for var in order:
            if var not in observations:
                continue
            
            mech = self.scm.mechanisms[var]
            parent_vals = _require_parent_observations(
                observations, mech.parents, mech.name,
            )
            
            # Expected value under the CPT
            cpt = mech.cpt
            config_idx = mech.parent_config_index(parent_vals)
            probs = cpt[:, config_idx]
            
            # Prediction = expected value index
            expected = np.sum(probs * np.arange(len(probs)))
            observed = float(observations[var])
            
            # Squared prediction error
            error = (observed - expected) ** 2
            total_energy += 0.5 * self.precision * error
        
        return total_energy
    
    def prediction(self, observations: Dict[str, int], 
                   target: str) -> np.ndarray:
        """Predict distribution over target given observed parents."""
        mech = self.scm.mechanisms.get(target)
        
        if mech is None:
            return np.array([1.0])
        
        parent_vals = _require_parent_observations(
            observations, mech.parents, mech.name,
        )
        
        config_idx = mech.parent_config_index(parent_vals)
        return mech.cpt[:, config_idx]


class EnergyCausalArena:
    """
    SCMs compete via prediction-error energy. Lowest energy wins.
    """
    
    def __init__(self, precision: float = 1.0, beta: float = 1.0):
        """
        Args:
            precision: Causal prediction error precision
            beta: Inverse temperature for model selection softmax
        """
        self.models: Dict[str, CausalEnergyTerm] = {}
        self.beta = beta
        self.precision = precision
        self._history: List[Dict[str, float]] = []
    
    def register(self, scm: StructuralCausalModel):
        """Add a competing causal model."""
        if scm.name in self.models:
            raise ValueError(
                f"EnergyCausalArena.register: model name {scm.name!r} is already registered"
            )
        
        self.models[scm.name] = CausalEnergyTerm(scm, self.precision)
    
    def compete(
        self, observations: Dict[str, int], record_history: bool = True,
    ) -> Dict[str, Any]:
        """
        All models compute their causal energy on the observation.
        Returns energies, posteriors, and tension.
        """
        energies = {}
        
        for name, term in self.models.items():
            energies[name] = term.energy(observations)
        
        if not energies:
            return {"winner": None, "tension": 1.0, "energies": {}}
        
        # Softmax over negative energies (lower energy = higher weight)
        vals = np.array(list(energies.values()))
        neg_e = -self.beta * vals
        neg_e -= neg_e.max()
        weights = np.exp(neg_e)
        weights /= weights.sum()
        
        posteriors = dict(zip(energies.keys(), weights.tolist()))
        tension = _normalized_entropy_tension_from_energy_values(vals, self.beta)
        
        winner = min(energies, key=energies.get)
        best_energy = energies[winner]
        
        result = {
            "winner": winner,
            "tension": tension,
            "posteriors": posteriors,
            "energies": energies,
            "best_energy": best_energy,
        }
        
        if record_history:
            self._history.append(energies)
        
        return result
    
    def best_energy(self, observations: Dict[str, int]) -> float:
        """Get the energy of the best-fitting model."""
        result = self.compete(observations, record_history=False)
        return result.get("best_energy", 0.0)
    
    def update_models(self, observations: Dict[str, int]):
        """Update all models' parameters from observation (Dirichlet counting)."""
        for name, term in self.models.items():
            term.scm.update_from_data([observations])
    
    @property
    def tension(self) -> float:
        """Current tension (from last competition)."""
        if not self._history:
            return 1.0

        last = self._history[-1]
        vals = np.array(list(last.values()))
        
        return _normalized_entropy_tension_from_energy_values(vals, self.beta)

