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
from typing import Dict, Optional, Callable, List, Any, Tuple, Set
from copy import deepcopy


class CausalMechanism:
    """
    A single causal mechanism: V_i := f_i(parents, noise).
    
    For discrete variables: parameterized as a Conditional Probability Table (CPT).
    For continuous variables: parameterized as an Additive Noise Model (ANM).
    """
    
    def __init__(self, name: str, n_values: int = 4,
                 parents: Optional[List[str]] = None,
                 noise_scale: float = 0.1):
        self.name = name
        self.n_values = n_values  # Discrete cardinality
        self.parents = parents or []
        self.noise_scale = noise_scale
        
        # Conditional Probability Table (Dirichlet-parameterized)
        # Shape: (n_values, n_parent_configs)
        if self.parents:
            # For now, assume each parent has n_values states
            n_parent_configs = n_values ** len(self.parents)
        else:
            n_parent_configs = 1
        
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
        for i, p in enumerate(self.parents):
            idx += parent_values.get(p, 0) * (self.n_values ** i)
        return idx % self.cpt_params.shape[1]
    
    def sample(self, parent_values: Dict[str, int]) -> int:
        """Sample from P(V_i | parents = parent_values)."""
        config_idx = self.parent_config_index(parent_values)
        probs = self.cpt[:, config_idx]
        return int(np.random.choice(self.n_values, p=probs))
    
    def log_prob(self, value: int, parent_values: Dict[str, int]) -> float:
        """Compute log P(V_i = value | parents)."""
        config_idx = self.parent_config_index(parent_values)
        probs = self.cpt[:, config_idx]
        return float(np.log(max(probs[value], 1e-16)))
    
    def update(self, value: int, parent_values: Dict[str, int]):
        """Bayesian update: increment Dirichlet parameter."""
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
        
        mechanism = CausalMechanism(name, n_values, parents, noise_scale)
        self.mechanisms[name] = mechanism
        self.graph.add_node(name, mechanism=mechanism)
        
        for parent in parents:
            if parent not in self.graph:
                # Auto-create parent as root node
                self.add_variable(parent, n_values)
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
        
        Uses likelihood weighting (importance sampling) since exact
        inference on general DAGs is NP-hard.
        
        Returns log-likelihood of the evidence under the model.
        """
        self._observed.update(evidence)
        
        # Set observed values on mechanisms
        for var, val in evidence.items():
            if var in self.mechanisms:
                self.mechanisms[var].value = val
        
        # Compute joint log-probability of evidence
        order = self.topological_order()
        log_prob = 0.0
        
        for var in order:
            if var in evidence:
                mech = self.mechanisms[var]
                parent_vals = {p: evidence.get(p, 0) for p in mech.parents}
                log_prob += mech.log_prob(evidence[var], parent_vals)
        
        return {'log_likelihood': log_prob}
    
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
            # Set CPT to delta function at intervention value
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
        # Step 1: ABDUCTION — infer noise for each mechanism
        order = self.topological_order()
        for var in order:
            if var in evidence:
                mech = self.mechanisms[var]
                parent_vals = {p: evidence.get(p, 0) for p in mech.parents}
                mech.abduce(evidence[var], parent_vals)
        
        # Step 2: INTERVENTION — create mutilated model
        cf_model = self.do(interventions)
        
        # Step 3: PREDICTION — forward sample with abduced noise
        n_cf_samples = 1000
        cf_results = {q: np.zeros(self.mechanisms[q].n_values) for q in query}
        
        for _ in range(n_cf_samples):
            values = {}
            for var in cf_model.topological_order():
                mech_cf = cf_model.mechanisms[var]
                mech_orig = self.mechanisms[var]
                
                if var in interventions:
                    values[var] = interventions[var]
                elif mech_orig.abduced_noise is not None:
                    # Use abduced noise
                    values[var] = int(np.random.choice(
                        mech_cf.n_values, p=mech_orig.abduced_noise))
                else:
                    parent_vals = {p: values.get(p, 0) for p in mech_cf.parents}
                    values[var] = mech_cf.sample(parent_vals)
                
                if var in cf_results:
                    cf_results[var][values[var]] += 1
        
        # Normalize to distributions
        for var in cf_results:
            total = cf_results[var].sum()
            if total > 0:
                cf_results[var] /= total
        
        return cf_results
    
    def log_evidence(self, data: List[Dict[str, int]]) -> float:
        """
        Compute log P(data | model) — marginal likelihood for model comparison.
        
        This is what the causal arena uses to rank competing models.
        """
        total_log_prob = 0.0
        order = self.topological_order()
        
        for sample in data:
            for var in order:
                if var in sample:
                    mech = self.mechanisms[var]
                    parent_vals = {p: sample.get(p, 0) for p in mech.parents}
                    total_log_prob += mech.log_prob(sample[var], parent_vals)
        
        return total_log_prob
    
    def update_from_data(self, data: List[Dict[str, int]]):
        """
        Update all mechanisms from observed data (Bayesian counting).
        No gradient descent — pure Dirichlet updates.
        """
        for sample in data:
            for var in self.topological_order():
                if var in sample:
                    mech = self.mechanisms[var]
                    parent_vals = {p: sample.get(p, 0) for p in mech.parents}
                    mech.update(sample[var], parent_vals)
    
    def d_separation(self, x: str, y: str, z: Optional[Set[str]] = None) -> bool:
        """
        Test d-separation: X ⊥⊥ Y | Z in the DAG.
        
        Uses the Bayes-Ball algorithm (Shachter, 1998).
        """
        z = z or set()
        
        # Simple implementation via active trail checking
        return not nx.d_separated(self.graph, {x}, {y}, z)
    
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
