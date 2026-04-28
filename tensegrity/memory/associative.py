"""
Associative Memory: Content-addressable pattern completion via modern Hopfield network.

This is NOT a neural network in the backprop sense. A Hopfield network is a
Markov Random Field with energy function E(x) = -½ x^T W x + b^T x.
Pattern completion is energy minimization via iterative coordinate descent
(asynchronous updates). No gradients are computed — each neuron flips to
the state that locally minimizes energy, and the system converges to a
fixed point (stored pattern).

Modern Hopfield networks (Ramsauer et al., 2020) replace the quadratic
energy with an exponential interaction that gives exponential storage
capacity: ~exp(d/2) patterns in d dimensions, vs ~0.14d for classical.

Energy function (modern):
  E(ξ) = -log Σ_μ exp(ξ^T x_μ) + ½ ||ξ||² + const

Update rule (single step):
  ξ_new = softmax(β * X^T ξ) · X
  where X = [x_1, ..., x_M] is the matrix of stored patterns
  and β = inverse temperature

This is literally the attention mechanism, but used as an ENERGY-BASED
MEMORY system, not a differentiable layer. We use it for one-shot 
pattern storage and content-addressed retrieval.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any


class AssociativeMemory:
    """
    Modern Hopfield Network as associative memory.
    
    Stores patterns and retrieves the closest stored pattern given
    a partial/noisy query. No training, no gradients — patterns are
    stored by appending to the pattern matrix. Retrieval is energy
    minimization via the softmax update rule.
    
    The connection to Bayesian inference:
      P(pattern_μ | query ξ) ∝ exp(β * ξ^T x_μ)
    This is a Boltzmann distribution over stored patterns.
    Retrieval = MAP estimation under this posterior.
    """
    
    def __init__(self, pattern_dim: int, beta: float = 1.0,
                 max_patterns: int = 10000, convergence_steps: int = 5,
                 zipf_exponent: float = 1.0,
                 access_decay: float = 0.99,
                 decay_every_n_retrieves: int = 50):
        """
        Args:
            pattern_dim: Dimensionality of stored patterns
            beta: Inverse temperature. Higher = sharper retrieval (more certain).
                  Lower = softer retrieval (more associative blending).
            max_patterns: Maximum number of stored patterns
            convergence_steps: Number of iterative updates for retrieval
            zipf_exponent: Controls power-law weighting of pattern importance
            access_decay: Multiplicative decay applied to access counts periodically
            decay_every_n_retrieves: Invoke decay every N retrieve() calls (0 = never)
        """
        self.dim = pattern_dim
        self.beta = beta
        self.max_patterns = max_patterns
        self.convergence_steps = convergence_steps
        self.zipf_s = zipf_exponent
        self.access_decay = access_decay
        self.decay_every_n_retrieves = decay_every_n_retrieves
        
        # Pattern storage matrix X ∈ ℝ^(dim × n_patterns)
        self.patterns: List[np.ndarray] = []
        self._pattern_matrix: Optional[np.ndarray] = None  # Cached
        self._dirty = True  # Flag to rebuild cache
        
        # Pattern metadata (labels, timestamps, access counts)
        self._metadata: List[Dict[str, Any]] = []
        self._access_counts: List[float] = []
        self._retrieve_calls = 0
    
    def clear(self):
        """Drop all stored patterns (new episode)."""
        self.patterns.clear()
        self._metadata.clear()
        self._access_counts.clear()
        self._pattern_matrix = None
        self._dirty = True
        self._retrieve_calls = 0
    
    def store(self, pattern: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """
        Store a pattern in associative memory.
        
        No training — just append. The pattern matrix grows.
        Storage capacity is exponential in dimension (modern Hopfield).
        
        Args:
            pattern: Vector of shape (dim,)
            metadata: Optional labels/tags
        
        Returns:
            Index of stored pattern
        """
        pattern = np.asarray(pattern, dtype=np.float64).flatten()
        assert len(pattern) == self.dim, \
            f"Pattern dim {len(pattern)} != memory dim {self.dim}"
        
        # Normalize for stable energy computation
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm
        
        idx = len(self.patterns)
        self.patterns.append(pattern.copy())
        self._metadata.append(metadata or {})
        self._access_counts.append(0.0)
        self._dirty = True
        
        # Capacity management
        if len(self.patterns) > self.max_patterns:
            self._evict()
        
        return idx
    
    def _maybe_decay_on_retrieve(self) -> None:
        self._retrieve_calls += 1
        if (
            self.decay_every_n_retrieves > 0
            and self._retrieve_calls % self.decay_every_n_retrieves == 0
        ):
            self._decay_access_counts()
    
    def retrieve(self, query: np.ndarray, return_energy: bool = False,
                top_k: int = 1) -> Any:
        """
        Content-addressed retrieval via energy minimization.
        
        Runs the modern Hopfield update rule iteratively:
          ξ_{t+1} = X · softmax(β · X^T · ξ_t)
        
        Converges to the stored pattern closest to the query.
        
        Args:
            query: Partial/noisy pattern to complete
            return_energy: Also return the energy at convergence
            top_k: Return top-k closest patterns (if > 1)
        
        Returns:
            Retrieved pattern(s), optionally with energy
        """
        if not self.patterns:
            return (np.zeros(self.dim), float('inf')) if return_energy else np.zeros(self.dim)
        
        self._maybe_decay_on_retrieve()
        
        self._ensure_matrix()
        query = np.asarray(query, dtype=np.float64).flatten()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        # Iterative update: ξ → X · softmax(β · X^T · ξ)
        xi = query.copy()
        for step in range(self.convergence_steps):
            # Similarities: X^T · ξ  (each stored pattern's similarity to query)
            similarities = self._pattern_matrix.T @ xi  # shape: (n_patterns,)
            
            # Apply Zipf weighting: scale similarities by pattern importance
            zipf_weights = self._zipf_weights()
            similarities += np.log(zipf_weights + 1e-16)
            
            # Softmax: P(pattern_μ | query)
            similarities_scaled = self.beta * similarities
            similarities_scaled -= similarities_scaled.max()  # Numerical stability
            weights = np.exp(similarities_scaled)
            weights /= weights.sum()
            
            # Update: weighted combination of stored patterns
            xi_new = self._pattern_matrix @ weights
            
            # Normalize
            norm = np.linalg.norm(xi_new)
            if norm > 0:
                xi_new /= norm
            
            # Check convergence
            if np.allclose(xi, xi_new, atol=1e-8):
                break
            xi = xi_new
        
        # Compute energy at convergence
        energy = self._energy(xi)
        
        if top_k > 1:
            # Return top-k patterns by similarity
            final_sims = self._pattern_matrix.T @ xi
            top_indices = np.argsort(final_sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                self._access_counts[idx] += 1
                results.append((self.patterns[idx].copy(), 
                              float(final_sims[idx]),
                              self._metadata[idx]))
            return (results, energy) if return_energy else results
        
        # Find closest stored pattern
        final_sims = self._pattern_matrix.T @ xi
        best_idx = np.argmax(final_sims)
        self._access_counts[best_idx] += 1
        
        result = self.patterns[best_idx].copy()
        return (result, energy) if return_energy else result
    
    def retrieve_soft(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft retrieval: return the Boltzmann distribution over all stored patterns.
        
        This is the Bayesian posterior: P(pattern_μ | query) ∝ exp(β * query^T x_μ)
        
        Returns:
            (blended_pattern, weights): The soft mixture and the weights over patterns
        """
        if not self.patterns:
            return np.zeros(self.dim), np.array([])
        
        self._maybe_decay_on_retrieve()
        
        self._ensure_matrix()
        
        query = np.asarray(query, dtype=np.float64).flatten()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        similarities = self._pattern_matrix.T @ query
        zipf_weights = self._zipf_weights()
        similarities += np.log(zipf_weights + 1e-16)
        
        scaled = self.beta * similarities
        scaled -= scaled.max()
        weights = np.exp(scaled)
        weights /= weights.sum()
        
        blended = self._pattern_matrix @ weights
        return blended, weights
    
    def _energy(self, xi: np.ndarray) -> float:
        """
        Compute Hopfield energy: E(ξ) = -log Σ_μ exp(β · ξ^T x_μ) + ½||ξ||²
        """
        self._ensure_matrix()
        similarities = self._pattern_matrix.T @ xi
        log_sum_exp = np.log(np.sum(np.exp(self.beta * similarities - 
                                            self.beta * similarities.max()))) + \
                      self.beta * similarities.max()
        return float(-log_sum_exp / self.beta + 0.5 * np.dot(xi, xi))
    
    def _decay_access_counts(self):
        """Reduce access counts so stale dominance slowly evaporates."""
        if not self._access_counts:
            return
        self._access_counts = [float(c) * self.access_decay for c in self._access_counts]
    
    def _zipf_weights(self) -> np.ndarray:
        """
        Compute Zipf weights based on access frequency.
        
        Patterns accessed more frequently get higher weight in retrieval.
        This creates a self-reinforcing power law: popular patterns
        become more accessible, rare patterns fade.
        """
        counts = np.maximum(self._access_counts, 0.0) + 1.0
        # Rank by access count (descending)
        ranks = np.argsort(np.argsort(-counts)) + 1  # 1-indexed ranks
        # Zipf weight: 1/rank^s
        weights = 1.0 / (ranks ** self.zipf_s)
        return weights / weights.sum()
    
    def _ensure_matrix(self):
        """Rebuild pattern matrix cache if dirty."""
        if self._dirty and self.patterns:
            self._pattern_matrix = np.column_stack(self.patterns)
            self._dirty = False
    
    def _evict(self):
        """Evict lowest-value patterns when capacity exceeded."""
        if len(self.patterns) <= self.max_patterns:
            return
        
        # Score: access_count + recency bonus
        scores = np.array(self._access_counts, dtype=np.float64)
        # Keep most accessed
        keep = np.argsort(scores)[::-1][:self.max_patterns]
        keep = sorted(keep)
        
        self.patterns = [self.patterns[i] for i in keep]
        self._metadata = [self._metadata[i] for i in keep]
        self._access_counts = [self._access_counts[i] for i in keep]
        self._dirty = True
    
    def pattern_overlap(self, idx_a: int, idx_b: int) -> float:
        """Compute overlap (dot product) between two stored patterns."""
        return float(np.dot(self.patterns[idx_a], self.patterns[idx_b]))
    
    @property
    def n_patterns(self) -> int:
        return len(self.patterns)
    
    @property
    def statistics(self) -> Dict[str, Any]:
        if not self.patterns:
            return {'n_patterns': 0}
        
        accesses = np.array(self._access_counts)
        return {
            'n_patterns': len(self.patterns),
            'total_accesses': int(accesses.sum()),
            'mean_access': float(accesses.mean()),
            'max_access': float(accesses.max()),
            'beta': self.beta,
            'dimension': self.dim,
        }
