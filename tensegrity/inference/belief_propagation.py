"""
Belief Propagation Engine: Message passing on factor graphs.

Implements the sum-product algorithm (belief propagation) for inference
on graphical models. This is the computational backbone shared by:
  - State inference (perception)
  - Policy evaluation (planning)
  - Model comparison (causal arena)

All three are instances of the SAME algorithm (sum-product on a factor graph)
applied to different factor graphs:
  - Perception: factor graph of the POMDP generative model
  - Planning: factor graph extended with future time steps
  - Model comparison: meta-factor-graph over model indicators

No gradients. Only messages passed between nodes.

The sum-product algorithm:
  Variable→Factor: μ_{x→f}(x) = Π_{g ∈ N(x)\\f} μ_{g→x}(x)
  Factor→Variable: μ_{f→x}(x) = Σ_{x_f\\x} f(x_f) Π_{y ∈ N(f)\\x} μ_{y→f}(y)
  Marginal: P(x) ∝ Π_{f ∈ N(x)} μ_{f→x}(x)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FactorNode:
    """A factor in the factor graph: represents a potential function."""
    
    def __init__(self, name: str, variables: List[str],
                 potential: np.ndarray):
        """
        Args:
            name: Factor identifier
            variables: List of variable names this factor connects
            potential: The factor potential function as a tensor.
                      Shape: (|var_1| × |var_2| × ... × |var_k|)
        """
        self.name = name
        self.variables = variables
        self.potential = potential
        self.messages_in: Dict[str, np.ndarray] = {}  # var_name → incoming message
    
    def __repr__(self):
        return f"Factor({self.name}, vars={self.variables})"


class VariableNode:
    """A variable in the factor graph: represents a random variable."""
    
    def __init__(self, name: str, n_values: int):
        self.name = name
        self.n_values = n_values
        self.messages_in: Dict[str, np.ndarray] = {}  # factor_name → incoming message
        self.observed: Optional[int] = None  # If clamped (evidence)
    
    @property
    def belief(self) -> np.ndarray:
        """Current marginal belief = product of all incoming messages."""
        if self.observed is not None:
            b = np.zeros(self.n_values)
            b[self.observed] = 1.0
            return b
        
        if not self.messages_in:
            return np.ones(self.n_values) / self.n_values
        
        belief = np.ones(self.n_values)
        for msg in self.messages_in.values():
            belief *= msg
        
        total = belief.sum()
        if total > 0:
            belief /= total
        else:
            belief = np.ones(self.n_values) / self.n_values
        
        return belief
    
    def __repr__(self):
        return f"Var({self.name}, |dom|={self.n_values})"


class BeliefPropagator:
    """
    Loopy Belief Propagation on a factor graph.
    
    Implements the sum-product algorithm with:
      - Damped updates (for convergence on loopy graphs)
      - Max-product mode (for MAP inference)
      - Evidence clamping (for conditioning)
    
    This is the computational primitive that underlies ALL inference
    in the Tensegrity architecture:
      - State estimation → BP on POMDP factor graph
      - Policy selection → BP on planning factor graph  
      - Model comparison → BP on meta-model factor graph
    """
    
    def __init__(self, damping: float = 0.5, max_iterations: int = 50,
                 convergence_threshold: float = 1e-6,
                 mode: str = 'sum_product'):
        """
        Args:
            damping: α ∈ [0, 1). New message = α * old + (1-α) * computed.
                    Higher = more damping = slower but more stable convergence.
            max_iterations: Maximum BP iterations
            convergence_threshold: Stop when max message change < this
            mode: 'sum_product' (marginals) or 'max_product' (MAP)
        """
        self.damping = damping
        self.max_iter = max_iterations
        self.conv_threshold = convergence_threshold
        self.mode = mode
        
        self.variables: Dict[str, VariableNode] = {}
        self.factors: Dict[str, FactorNode] = {}
        
        # Adjacency
        self._var_to_factors: Dict[str, List[str]] = defaultdict(list)
        self._factor_to_vars: Dict[str, List[str]] = defaultdict(list)
        
        # Convergence tracking
        self.iteration_count = 0
        self.converged = False
        self.residuals: List[float] = []
    
    def add_variable(self, name: str, n_values: int) -> VariableNode:
        """Add a variable node to the factor graph."""
        var = VariableNode(name, n_values)
        self.variables[name] = var
        return var
    
    def add_factor(self, name: str, variables: List[str],
                   potential: np.ndarray) -> FactorNode:
        """
        Add a factor node connecting the given variables.
        
        The potential tensor has one axis per variable, with sizes
        matching each variable's cardinality.
        """
        factor = FactorNode(name, variables, potential)
        self.factors[name] = factor
        
        for var_name in variables:
            self._var_to_factors[var_name].append(name)
            self._factor_to_vars[name].append(var_name)
        
        return factor
    
    def set_evidence(self, variable: str, value: int):
        """Clamp a variable to an observed value."""
        if variable in self.variables:
            self.variables[variable].observed = value
    
    def clear_evidence(self):
        """Remove all evidence."""
        for var in self.variables.values():
            var.observed = None
    
    def propagate(self) -> Dict[str, np.ndarray]:
        """
        Run belief propagation until convergence.
        
        Returns:
            Dictionary of variable name → marginal belief
        """
        # Initialize messages
        self._initialize_messages()
        
        self.converged = False
        self.residuals = []
        
        for iteration in range(self.max_iter):
            max_residual = 0.0
            
            # Variable → Factor messages
            for var_name, factor_names in self._var_to_factors.items():
                for factor_name in factor_names:
                    residual = self._send_var_to_factor(var_name, factor_name)
                    max_residual = max(max_residual, residual)
            
            # Factor → Variable messages
            for factor_name, var_names in self._factor_to_vars.items():
                for var_name in var_names:
                    residual = self._send_factor_to_var(factor_name, var_name)
                    max_residual = max(max_residual, residual)
            
            self.residuals.append(max_residual)
            self.iteration_count = iteration + 1
            
            if max_residual < self.conv_threshold:
                self.converged = True
                break
        
        # Collect marginals
        marginals = {}
        for var_name, var_node in self.variables.items():
            marginals[var_name] = var_node.belief
        
        return marginals
    
    def _initialize_messages(self):
        """Initialize all messages to uniform."""
        for var_name, var_node in self.variables.items():
            for factor_name in self._var_to_factors[var_name]:
                # Variable → Factor: uniform
                self.factors[factor_name].messages_in[var_name] = \
                    np.ones(var_node.n_values) / var_node.n_values
        
        for factor_name, factor_node in self.factors.items():
            for var_name in factor_node.variables:
                var_node = self.variables[var_name]
                # Factor → Variable: uniform
                var_node.messages_in[factor_name] = \
                    np.ones(var_node.n_values) / var_node.n_values
    
    def _send_var_to_factor(self, var_name: str, factor_name: str) -> float:
        """
        Variable → Factor message:
        μ_{x→f}(x) = Π_{g ∈ N(x)\\f} μ_{g→x}(x)
        """
        var_node = self.variables[var_name]
        
        if var_node.observed is not None:
            # Clamped: send delta function
            msg = np.zeros(var_node.n_values)
            msg[var_node.observed] = 1.0
        else:
            # Product of all incoming factor messages except target
            msg = np.ones(var_node.n_values)
            for other_factor in self._var_to_factors[var_name]:
                if other_factor != factor_name:
                    if other_factor in var_node.messages_in:
                        msg *= var_node.messages_in[other_factor]
            
            total = msg.sum()
            if total > 0:
                msg /= total
        
        # Damping
        old_msg = self.factors[factor_name].messages_in.get(
            var_name, np.ones(var_node.n_values) / var_node.n_values)
        new_msg = self.damping * old_msg + (1 - self.damping) * msg
        
        residual = float(np.max(np.abs(new_msg - old_msg)))
        self.factors[factor_name].messages_in[var_name] = new_msg
        
        return residual
    
    def _send_factor_to_var(self, factor_name: str, var_name: str) -> float:
        """
        Factor → Variable message:
        μ_{f→x}(x) = Σ_{x_f\\x} f(x_f) Π_{y ∈ N(f)\\x} μ_{y→f}(y)
        
        For max-product: replace Σ with max.
        """
        factor_node = self.factors[factor_name]
        var_node = self.variables[var_name]
        target_idx = factor_node.variables.index(var_name)
        
        # Gather incoming messages from all other variables
        other_messages = []
        for i, other_var in enumerate(factor_node.variables):
            if other_var != var_name:
                msg = factor_node.messages_in.get(
                    other_var, np.ones(self.variables[other_var].n_values) / 
                    self.variables[other_var].n_values)
                other_messages.append((i, msg))
        
        if not other_messages:
            # Unary factor — just read off the potential
            if factor_node.potential.ndim == 1:
                msg = factor_node.potential.copy()
            else:
                # Sum/max over other dimensions
                msg = factor_node.potential.sum(axis=tuple(
                    i for i in range(factor_node.potential.ndim) if i != target_idx))
        else:
            # Contract the potential tensor with incoming messages
            potential = factor_node.potential.copy()
            
            # Build the message by marginalizing over all variables except target
            # For efficiency with small factors, use einsum-like contraction
            msg = self._contract_factor(potential, factor_node.variables,
                                        var_name, target_idx, other_messages)
        
        # Normalize
        total = msg.sum()
        if total > 0:
            msg /= total
        else:
            msg = np.ones(var_node.n_values) / var_node.n_values
        
        # Damping
        old_msg = var_node.messages_in.get(
            factor_name, np.ones(var_node.n_values) / var_node.n_values)
        new_msg = self.damping * old_msg + (1 - self.damping) * msg
        
        residual = float(np.max(np.abs(new_msg - old_msg)))
        var_node.messages_in[factor_name] = new_msg
        
        return residual
    
    def _contract_factor(self, potential: np.ndarray, var_names: List[str],
                        target: str, target_idx: int,
                        other_messages: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """
        Contract a factor tensor with incoming messages.
        
        For a factor f(x, y, z) with target x and messages μ_y, μ_z:
          result(x) = Σ_y Σ_z f(x, y, z) · μ_y(y) · μ_z(z)
        """
        result = potential.copy()
        
        # Weight by incoming messages
        for axis_idx, msg in sorted(other_messages, key=lambda x: x[0], reverse=True):
            # Reshape message for broadcasting
            shape = [1] * result.ndim
            shape[axis_idx] = len(msg)
            weighted = msg.reshape(shape)
            result = result * weighted
        
        # Marginalize: sum over all axes except target
        axes_to_sum = tuple(i for i in range(result.ndim) if i != target_idx)
        if axes_to_sum:
            if self.mode == 'max_product':
                for ax in sorted(axes_to_sum, reverse=True):
                    result = result.max(axis=ax)
            else:
                for ax in sorted(axes_to_sum, reverse=True):
                    result = result.sum(axis=ax)
        
        return result.flatten()[:self.variables[target].n_values]
    
    def log_partition(self) -> float:
        """
        Approximate log partition function (Bethe approximation).
        
        log Z ≈ Σ_f Σ_{x_f} q(x_f) [ln f(x_f) - ln q(x_f)]
               + Σ_i (d_i - 1) Σ_{x_i} q(x_i) ln q(x_i)
        
        Where d_i = degree of variable i (number of connected factors).
        """
        log_Z = 0.0
        
        # Factor contribution
        for factor_name, factor_node in self.factors.items():
            # Approximate factor marginal as product of variable beliefs
            beliefs = [self.variables[v].belief for v in factor_node.variables]
            
            if len(beliefs) == 1:
                q_f = beliefs[0]
                pot_flat = factor_node.potential.flatten()[:len(q_f)]
                log_pot = np.log(np.maximum(pot_flat, 1e-16))
                log_q = np.log(np.maximum(q_f, 1e-16))
                log_Z += np.sum(q_f * (log_pot - log_q))
            elif len(beliefs) == 2:
                q_f = np.outer(beliefs[0], beliefs[1])
                pot = factor_node.potential
                if pot.shape == q_f.shape:
                    log_pot = np.log(np.maximum(pot, 1e-16))
                    log_q = np.log(np.maximum(q_f, 1e-16))
                    log_Z += np.sum(q_f * (log_pot - log_q))
        
        # Variable correction
        for var_name, var_node in self.variables.items():
            degree = len(self._var_to_factors[var_name])
            if degree > 1:
                b = var_node.belief
                log_b = np.log(np.maximum(b, 1e-16))
                log_Z += (degree - 1) * np.sum(b * log_b)
        
        return log_Z
    
    def free_energy(self) -> float:
        """Bethe free energy = -log_partition (approximate)."""
        return -self.log_partition()
    
    @property
    def statistics(self) -> Dict[str, Any]:
        return {
            'n_variables': len(self.variables),
            'n_factors': len(self.factors),
            'iterations': self.iteration_count,
            'converged': self.converged,
            'final_residual': self.residuals[-1] if self.residuals else None,
            'bethe_free_energy': self.free_energy() if self.variables else None,
        }
    
    @staticmethod
    def from_bayesian_network(edges: List[Tuple[str, str]],
                              cpds: Dict[str, np.ndarray],
                              cardinalities: Dict[str, int]) -> 'BeliefPropagator':
        """
        Build a factor graph from a Bayesian network specification.
        
        Args:
            edges: List of (parent, child) edges
            cpds: Conditional probability tables {child: P(child | parents)}
            cardinalities: {var_name: n_values}
        """
        bp = BeliefPropagator()
        
        # Add variables
        all_vars = set()
        for parent, child in edges:
            all_vars.add(parent)
            all_vars.add(child)
        for var in cpds:
            all_vars.add(var)
        
        for var in all_vars:
            n_vals = cardinalities.get(var, 2)
            bp.add_variable(var, n_vals)
        
        # Build parent map
        parents = defaultdict(list)
        for parent, child in edges:
            parents[child].append(parent)
        
        # Add factors
        for var, cpd in cpds.items():
            parent_list = parents.get(var, [])
            factor_vars = parent_list + [var]
            bp.add_factor(f"f_{var}", factor_vars, cpd)
        
        return bp
