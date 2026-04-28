"""
Structural Causal Model: Pearl's M = (U, V, F) formalism.

A structural causal model defines the data-generating process of the world.
Each variable V_i is determined by a structural equation:
    V_i := f_i(Pa_i, U_i)
Where Pa_i are the parents of V_i in the causal DAG, and U_i is exogenous noise.

The three rungs of Pearl's causal hierarchy:
  1. Association: P(Y | X=x) — what is observed
  2. Intervention: P(Y | do(X=x)) — what happens when we act
  3. Counterfactual: P(Y_x | X=x', Y=y') — what would have happened

Implementation: Each SCM is a networkx DiGraph where nodes carry:
  - Structural equation (deterministic function of parents + noise)
  - Noise distribution (prior over exogenous U)
  - Current observed value
  - Dirichlet-parameterized CPD (for discrete variables)
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional, List, Tuple, Set
from copy import deepcopy
from itertools import product


class CausalMechanism:
    """
    A single causal mechanism: V_i := f_i(parents, noise).
    
    For discrete variables: parameterized as a Conditional Probability Table (CPT).
    For continuous variables: parameterized as an Additive Noise Model (ANM).
    """
    
    def __init__(
        self, name: str, n_values: int = 4,
        parents: Optional[List[str]] = None,
        noise_scale: float = 0.1,
        parent_cardinalities: Optional[List[int]] = None,
    ) -> None:
        self.name = name
        self.n_values = n_values  # Discrete cardinality
        self.parents = parents or []
        self.parent_cardinalities = list(parent_cardinalities or [n_values] * len(self.parents))
        
        if len(self.parent_cardinalities) != len(self.parents):
            raise ValueError("parent_cardinalities must match parents")
        
        self.noise_scale = noise_scale
        
        # Conditional Probability Table (Dirichlet-parameterized)
        # Shape: (n_values, n_parent_configs)
        n_parent_configs = 1
        
        for card in self.parent_cardinalities:
            n_parent_configs *= max(int(card), 1)
        
        # Initialize CPT as uniform Dirichlet
        self.cpt_params = np.ones((n_values, n_parent_configs))
        
        # Current observed value
        self.value: Optional[int] = None
        
        # Abduced noise (for counterfactual reasoning)
        self.abduced_noise: Optional[np.ndarray] = None
    
    @property
    def cpt(self) -> np.ndarray:
        """Normalized CPT: P(V | parents)"""
        return self.cpt_params / self.cpt_params.sum(axis=0, keepdims=True)
    
    def parent_config_index(self, parent_values: Dict[str, int]) -> int:
        """Convert parent values to a CPT column index."""
        if not self.parents:
            return 0
        
        idx = 0
        stride = 1
        
        for p, card in zip(self.parents, self.parent_cardinalities):
            value = int(parent_values.get(p, 0))
            idx += (value % max(int(card), 1)) * stride
            stride *= max(int(card), 1)
        
        return idx % self.cpt_params.shape[1]
    
    def sample(self, parent_values: Dict[str, int]) -> int:
        """Sample from P(V_i | parents = parent_values)."""
        config_idx = self.parent_config_index(parent_values)
        probs = self.cpt[:, config_idx]
        return int(np.random.choice(self.n_values, p=probs))
    
    def log_prob(self, value: int, parent_values: Dict[str, int]) -> float:
        """Compute log P(V_i = value | parents)."""
        value = int(value)

        if value < 0 or value >= self.n_values:
            return float(np.log(1e-16))

        config_idx = self.parent_config_index(parent_values)
        probs = self.cpt[:, config_idx]

        return float(np.log(max(probs[value], 1e-16)))
    
    def update(self, value: int, parent_values: Dict[str, int]):
        """Bayesian update: increment Dirichlet parameter."""
        value = int(value)
        
        if value < 0 or value >= self.n_values:
            return
        
        config_idx = self.parent_config_index(parent_values)
        self.cpt_params[value, config_idx] += 1.0
    
    def abduce(self, value: int, parent_values: Dict[str, int]) -> np.ndarray:
        """
        Step 1 of counterfactual: infer the noise that produced this observation.
        
        For discrete CPTs: the "noise" is the full conditional P(V | parents),
        conditioned on the observed value. We store the posterior over noise.
        """
        config_idx = self.parent_config_index(parent_values)
        probs = self.cpt[:, config_idx].copy()
        
        # The abduced noise is the conditional distribution concentrated on observed value
        noise = np.zeros(self.n_values)
        noise[value] = 1.0
        self.abduced_noise = noise
        
        return noise


class StructuralCausalModel:
    """
    Pearl's M = (U, V, F): a complete structural causal model.
    
    The DAG encodes causal structure. Each node carries a CausalMechanism.
    Supports all three rungs:
      - observe(): Association (Rung 1)
      - do(): Intervention (Rung 2)  
      - counterfactual(): Counterfactual (Rung 3)
    """
    
    def __init__(self, name: str = "SCM"):
        self.name = name
        self.graph = nx.DiGraph()
        self.mechanisms: Dict[str, CausalMechanism] = {}
        self._observed: Dict[str, int] = {}
    
    def add_variable(self, name: str, n_values: int = 4,
                     parents: Optional[List[str]] = None,
                     noise_scale: float = 0.1):
        """Add a variable with its causal mechanism."""
        parents = parents or []

        for parent in parents:
            if parent not in self.graph:
                # Auto-create parent as root node with the child's cardinality.
                self.add_variable(parent, n_values)

        parent_cardinalities = [self.mechanisms[p].n_values for p in parents]
        
        mechanism = CausalMechanism(
            name,
            n_values,
            parents,
            noise_scale,
            parent_cardinalities=parent_cardinalities,
        )

        self.mechanisms[name] = mechanism
        self.graph.add_node(name, mechanism=mechanism)
        
        for parent in parents:
            self.graph.add_edge(parent, name)
    
    def topological_order(self) -> List[str]:
        """Variables in causal order (parents before children)."""
        return list(nx.topological_sort(self.graph))
    
    def sample(self, n_samples: int = 1) -> List[Dict[str, int]]:
        """
        Forward sample from the SCM.
        
        Follows topological order: sample parents first, then children.
        """
        order = self.topological_order()
        samples = []
        
        for _ in range(n_samples):
            values = {}

            for var in order:
                mech = self.mechanisms[var]
                parent_vals = {p: values[p] for p in mech.parents if p in values}
                values[var] = mech.sample(parent_vals)

            samples.append(values)
        
        return samples
    
    def observe(self, evidence: Dict[str, int]) -> Dict[str, float]:
        """
        Rung 1 — Association: Compute P(query | evidence).
        
        Returns the exact finite-discrete log-likelihood of the evidence under
        the model by marginalizing unobserved variables. Unknown evidence keys
        are ignored to preserve the previous tolerant public behavior.
        """
        self._observed.update(evidence)
        
        # Set observed values on mechanisms
        for var, val in evidence.items():
            if var in self.mechanisms:
                self.mechanisms[var].value = val

        return {'log_likelihood': self.log_evidence([evidence])}
    
    def do(self, interventions: Dict[str, int]) -> 'StructuralCausalModel':
        """
        Rung 2 — Intervention: Apply do(X=x).
        
        Graph surgery: remove all incoming edges to intervened variables
        and fix their values. Returns a MUTILATED model.
        
        This is Pearl's truncated factorization:
        P(v | do(x)) = Π_{V_i ∉ X} P(v_i | pa_i) * I(X=x)
        """
        mutilated = deepcopy(self)
        mutilated.name = f"{self.name}_do({interventions})"
        
        for var, val in interventions.items():
            if var not in mutilated.graph:
                continue
            
            # Remove all incoming edges (graph surgery)
            parents = list(mutilated.graph.predecessors(var))

            for parent in parents:
                mutilated.graph.remove_edge(parent, var)
            
            # Fix the mechanism to produce the intervention value deterministically
            mech = mutilated.mechanisms[var]
            mech.parents = []
            mech.parent_cardinalities = []

            # Set CPT to delta function at intervention value
            if val < 0 or val >= mech.n_values:
                raise ValueError(f"Intervention value {val} out of range for {var}")

            mech.cpt_params = np.zeros((mech.n_values, 1))
            mech.cpt_params[val, 0] = 1000.0  # Strong prior = near-deterministic
            mech.value = val
        
        return mutilated
    
    def counterfactual(self, evidence: Dict[str, int],
                       interventions: Dict[str, int],
                       query: List[str]) -> Dict[str, np.ndarray]:
        """
        Rung 3 — Counterfactual: P(Y_{do(x)} | observed evidence).
        
        Pearl's 3-step procedure:
          1. ABDUCTION: Infer noise U given evidence
          2. INTERVENTION: Modify structural equations (graph surgery)
          3. PREDICTION: Forward-propagate with modified model + abduced noise
        
        Returns distribution over query variables in the counterfactual world.
        """
        for var, val in interventions.items():
            if var in self.mechanisms:
                n_values = self.mechanisms[var].n_values
                if val < 0 or val >= n_values:
                    raise ValueError(f"Intervention value {val} out of range for {var}")

        cf_results = {
            q: np.zeros(self.mechanisms[q].n_values, dtype=np.float64)
            for q in query
            if q in self.mechanisms
        }

        if not cf_results:
            return {}

        posterior_worlds = self._posterior_assignments(evidence)
        if not posterior_worlds:
            return cf_results

        order = self.topological_order()
        affected: Set[str] = set()
        for var in interventions:
            if var in self.graph:
                affected.add(var)
                affected.update(nx.descendants(self.graph, var))

        for posterior_assignment, posterior_weight in posterior_worlds:
            worlds: List[Tuple[Dict[str, int], float]] = [({}, posterior_weight)]

            for var in order:
                next_worlds: List[Tuple[Dict[str, int], float]] = []

                for values, weight in worlds:
                    if var in interventions:
                        v = int(interventions[var])
                        updated = dict(values)
                        updated[var] = v
                        next_worlds.append((updated, weight))
                        continue

                    if var not in affected:
                        # Abduced context outside the intervention's downstream
                        # cone is held fixed from the posterior factual world.
                        updated = dict(values)
                        updated[var] = int(posterior_assignment[var])
                        next_worlds.append((updated, weight))
                        continue

                    mech = self.mechanisms[var]
                    parent_vals = {p: values[p] for p in mech.parents}
                    probs = mech.cpt[:, mech.parent_config_index(parent_vals)]

                    for v, p_v in enumerate(probs):
                        if p_v <= 0:
                            continue
                        updated = dict(values)
                        updated[var] = int(v)
                        next_worlds.append((updated, weight * float(p_v)))

                worlds = next_worlds

            for values, weight in worlds:
                for q in cf_results:
                    cf_results[q][values[q]] += weight

        for var, dist in cf_results.items():
            total = float(dist.sum())
            if total > 0:
                cf_results[var] = dist / total

        return cf_results
    
    def log_evidence(self, data: List[Dict[str, int]]) -> float:
        """
        Compute log P(data | model) — marginal likelihood for model comparison.
        
        This is what the causal arena uses to rank competing models.
        """
        total_log_prob = 0.0
        
        for sample in data:
            likelihood = self._evidence_likelihood(sample)
            total_log_prob += float(np.log(max(likelihood, 1e-300)))
        
        return total_log_prob

    def _evidence_likelihood(self, evidence: Dict[str, int]) -> float:
        """Exact P(evidence) for this finite discrete SCM."""
        known = {
            var: int(val)
            for var, val in evidence.items()
            if var in self.mechanisms
        }

        for var, val in known.items():
            n_values = self.mechanisms[var].n_values
            if val < 0 or val >= n_values:
                return 0.0

        total = 0.0
        for assignment, prob in self._enumerate_joint_assignments(known):
            total += prob
        return float(total)

    def _joint_probability(self, assignment: Dict[str, int]) -> float:
        """P(assignment) under the SCM, assuming all variables are assigned."""
        prob = 1.0
        for var in self.topological_order():
            mech = self.mechanisms[var]
            parent_vals = {p: assignment[p] for p in mech.parents}
            prob *= float(np.exp(mech.log_prob(assignment[var], parent_vals)))
            if prob <= 0.0:
                return 0.0
        return float(prob)

    def _enumerate_joint_assignments(
        self,
        evidence: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[Dict[str, int], float]]:
        """Enumerate complete assignments consistent with evidence and their joint probabilities."""
        evidence = evidence or {}
        order = self.topological_order()
        missing = [var for var in order if var not in evidence]
        domains = [range(self.mechanisms[var].n_values) for var in missing]
        assignments: List[Tuple[Dict[str, int], float]] = []

        for values in product(*domains):
            assignment = dict(evidence)
            assignment.update({var: int(value) for var, value in zip(missing, values)})
            prob = self._joint_probability(assignment)
            if prob > 0.0:
                assignments.append((assignment, prob))

        return assignments

    def _posterior_assignments(self, evidence: Dict[str, int]) -> List[Tuple[Dict[str, int], float]]:
        """Enumerate complete factual worlds weighted by P(world | evidence)."""
        known = {
            var: int(val)
            for var, val in evidence.items()
            if var in self.mechanisms
        }
        assignments = self._enumerate_joint_assignments(known)
        total = sum(prob for _, prob in assignments)

        if total <= 0:
            return []

        return [(assignment, prob / total) for assignment, prob in assignments]
    
    def update_from_data(self, data: List[Dict[str, int]]):
        """
        Update all mechanisms from observed data (Bayesian counting).
        No gradient descent — pure Dirichlet updates.
        """
        for sample in data:
            for var in self.topological_order():
                if var in sample:
                    mech = self.mechanisms[var]
                    if any(p not in sample for p in mech.parents):
                        continue
                    parent_vals = {p: sample[p] for p in mech.parents}
                    mech.update(sample[var], parent_vals)
    
    def d_separation(self, x: str, y: str, z: Optional[Set[str]] = None) -> bool:
        """
        Test d-separation: X ⊥⊥ Y | Z in the DAG.
        
        Uses the Bayes-Ball algorithm (Shachter, 1998).
        """
        z = z or set()
        
        # NetworkX exposes this as ``is_d_separator`` in recent versions.
        return bool(nx.is_d_separator(self.graph, {x}, {y}, z))
    
    def find_adjustment_set(self, treatment: str, outcome: str) -> Optional[Set[str]]:
        """
        Find a valid backdoor adjustment set for P(Y | do(X)).
        
        The backdoor criterion (Pearl, 1993):
        Z satisfies the backdoor criterion if:
          1. No node in Z is a descendant of X
          2. Z blocks every path between X and Y that contains an arrow into X
        """
        # Get all non-descendants of X (candidates for adjustment)
        descendants_x = nx.descendants(self.graph, treatment)
        candidates = set(self.graph.nodes()) - descendants_x - {treatment, outcome}
        
        # Try the parents of X (simplest valid set if it works)
        parents_x = set(self.graph.predecessors(treatment))
        if parents_x <= candidates:
            # Check if parents block all backdoor paths
            # A simple sufficient condition: parents of X that aren't descendants
            valid_parents = parents_x - descendants_x
            if valid_parents:
                return valid_parents
        
        # Fall back to all non-descendants of X that are ancestors of Y or X
        ancestors_y = nx.ancestors(self.graph, outcome)
        ancestors_x = nx.ancestors(self.graph, treatment)
        potential = (ancestors_y | ancestors_x) & candidates
        
        return potential if potential else candidates if candidates else None
    
    @property
    def variables(self) -> List[str]:
        return list(self.graph.nodes())
    
    @property
    def edges(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges())
    
    def __repr__(self):
        return f"SCM({self.name}, vars={self.variables}, edges={self.edges})"
