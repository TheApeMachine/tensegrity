"""
Epistemic Memory: What the system believes about the structure of the world.

This is NOT a lookup table. It's a Dirichlet-parameterized probability distribution
over causal structures. Each "belief" is a posterior distribution that gets updated
via Bayesian counting — no gradients, just evidence accumulation.

The epistemic memory stores:
  - A matrices: P(observation | hidden state) — likelihood beliefs
  - B matrices: P(next state | current state, action) — transition beliefs
  - D vectors: P(initial state) — prior beliefs about starting conditions
  - Model evidence: P(data | model) for each competing causal model

Zipf distribution governs retrieval priority: beliefs that have been accessed
most frequently are cheapest to retrieve, following power-law access patterns
observed in human semantic memory (Anderson & Schooler, 1991).
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


class EpistemicMemory:
    """
    Bayesian belief store with Dirichlet-parameterized distributions.
    
    Every belief is a Dirichlet distribution (conjugate prior for Categorical).
    Updates are pure counting: observe a transition (s, a, s') → increment
    the corresponding Dirichlet parameter. No optimizer, no learning rate.
    
    Zipf-weighted retrieval: access frequency follows power law.
    """
    
    def __init__(self, n_states: int, n_observations: int, n_actions: int,
                 dirichlet_prior: float = 1.0, zipf_exponent: float = 1.0):
        """
        Args:
            n_states: Number of hidden states in the generative model
            n_observations: Number of possible observations
            n_actions: Number of possible actions
            dirichlet_prior: Prior concentration parameter (uniform = 1.0)
            zipf_exponent: Controls the steepness of the access priority curve
        """
        self.n_states = n_states
        self.n_obs = n_observations
        self.n_actions = n_actions
        self.zipf_s = zipf_exponent
        
        # === LIKELIHOOD BELIEFS: A[o, s] = P(o | s) ===
        # Dirichlet parameters — each column is a Dirichlet over observations
        self.A_params = np.full((n_observations, n_states), dirichlet_prior)
        
        # === TRANSITION BELIEFS: B[s', s, a] = P(s' | s, a) ===
        # Dirichlet parameters — each slice B[:, s, a] is a Dirichlet over next states
        self.B_params = np.full((n_states, n_states, n_actions), dirichlet_prior)
        
        # === INITIAL STATE BELIEFS: D[s] = P(s_0) ===
        self.D_params = np.full(n_states, dirichlet_prior)
        
        # === PREFERENCE BELIEFS: C[o] = log P̃(o) ===
        # How much the agent "wants" each observation (Zipf-distributed by default)
        ranks = np.arange(1, n_observations + 1, dtype=np.float64)
        self.C = -np.log(ranks)  # Zipf prior: prefer low-rank (common) observations
        
        # === ACCESS TRACKING (for Zipf-weighted retrieval) ===
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._total_accesses = 0
        
        # === MODEL EVIDENCE LOG ===
        self.evidence_log: List[float] = []
    
    @property
    def A(self) -> np.ndarray:
        """
        Expected likelihood matrix E[A] under Dirichlet posterior.
        
        E[Cat(α)] = α / sum(α) for each column
        """
        self._access_counts['A'] += 1
        self._total_accesses += 1
        return self.A_params / self.A_params.sum(axis=0, keepdims=True)
    
    @property
    def B(self) -> np.ndarray:
        """Expected transition tensor E[B] under Dirichlet posterior."""
        self._access_counts['B'] += 1
        self._total_accesses += 1
        return self.B_params / self.B_params.sum(axis=0, keepdims=True)
    
    @property
    def D(self) -> np.ndarray:
        """Expected initial state distribution E[D] under Dirichlet posterior."""
        self._access_counts['D'] += 1
        self._total_accesses += 1
        return self.D_params / self.D_params.sum()
    
    @property
    def log_A(self) -> np.ndarray:
        """Expected log-likelihood E[ln A] under Dirichlet (digamma form)."""
        from scipy.special import digamma
        self._access_counts['log_A'] += 1
        self._total_accesses += 1
        return digamma(self.A_params) - digamma(self.A_params.sum(axis=0, keepdims=True))
    
    @property
    def log_B(self) -> np.ndarray:
        """Expected log-transition E[ln B] under Dirichlet."""
        from scipy.special import digamma
        self._access_counts['log_B'] += 1
        self._total_accesses += 1
        return digamma(self.B_params) - digamma(self.B_params.sum(axis=0, keepdims=True))
    
    def update_likelihood(self, observation_idx: int, state_posterior: np.ndarray):
        """
        Update A (likelihood) beliefs given an observation and state posterior.
        
        This is the Dirichlet counting update:
            A_params[o, :] += Q(s) * I(o_observed)
        
        No gradient. Pure Bayesian counting.
        """
        self.A_params[observation_idx, :] += state_posterior
    
    def update_transition(self, state_posterior_prev: np.ndarray,
                         state_posterior_curr: np.ndarray,
                         action: int):
        """
        Update B (transition) beliefs given consecutive state posteriors and action.
        
        B_params[:, :, a] += Q(s_t) ⊗ Q(s_{t-1})  (outer product)
        """
        outer = np.outer(state_posterior_curr, state_posterior_prev)
        self.B_params[:, :, action] += outer
    
    def update_initial(self, state_posterior: np.ndarray):
        """Update D (initial state) beliefs."""
        self.D_params += state_posterior
    
    def update_preferences(self, observation_idx: int, valence: float = 1.0):
        """
        Update C (preferences) based on experienced observation.
        
        Observations that co-occur with positive outcomes get boosted.
        This maintains the Zipf structure but shifts it based on experience.
        """
        self.C[observation_idx] += valence
    
    def zipf_retrieval_cost(self, key: str) -> float:
        """
        Compute the retrieval cost for a belief, following Zipf's law.
        
        Frequently accessed beliefs are cheaper (faster) to retrieve.
        Cost ∝ 1 / (rank ^ zipf_exponent)
        
        Returns a value in (0, 1] where lower = cheaper.
        """
        if self._total_accesses == 0:
            return 1.0
        
        # Sort beliefs by access frequency (descending)
        sorted_keys = sorted(self._access_counts.keys(), 
                           key=lambda k: self._access_counts[k], 
                           reverse=True)
        
        if key not in self._access_counts:
            # Never accessed — maximum cost
            return 1.0
        
        rank = sorted_keys.index(key) + 1  # 1-indexed rank
        # Zipf cost: 1 / rank^s, normalized
        cost = 1.0 / (rank ** self.zipf_s)
        max_cost = 1.0  # rank 1
        return cost / max_cost
    
    def get_access_distribution(self) -> Dict[str, float]:
        """Get the current Zipf-like access frequency distribution."""
        total = max(self._total_accesses, 1)
        return {k: v / total for k, v in 
                sorted(self._access_counts.items(), 
                      key=lambda x: x[1], reverse=True)}
    
    def model_evidence(self, observations: np.ndarray) -> float:
        """
        Compute log model evidence P(observations | model) for Bayesian model comparison.
        
        Uses the Dirichlet-Categorical conjugate relationship:
        log P(o₁:T) = Σ_t log E_Dir[P(o_t | s_t)]
        
        This is a lower bound — the true evidence requires marginalizing
        over hidden states, which the free energy engine does.
        """
        A = self.A
        # Simple observation likelihood under uniform state prior
        log_lik = 0.0
        for obs in observations:
            obs_idx = int(obs) % self.n_obs
            # Marginal: sum over states with uniform prior
            p_obs = A[obs_idx, :].mean()
            log_lik += np.log(max(p_obs, 1e-16))
        
        self.evidence_log.append(log_lik)
        return log_lik
    
    def entropy(self) -> Dict[str, float]:
        """Compute non-negative categorical entropy of expected beliefs.

        SciPy's Dirichlet differential entropy can be negative, which is
        mathematically valid for continuous densities but confusing as an
        uncertainty dashboard metric.  The agent usually wants entropy of the
        expected categorical distributions it will actually act on.
        """
        A = self.A_params / self.A_params.sum(axis=0, keepdims=True)
        D = self.D_params / self.D_params.sum()

        a_entropy_by_state = -np.sum(A * np.log(np.maximum(A, 1e-16)), axis=0)
        d_entropy = -np.sum(D * np.log(np.maximum(D, 1e-16)))
        
        return {
            'likelihood_entropy': float(np.mean(a_entropy_by_state)),
            'initial_state_entropy': float(d_entropy),
        }
    
    def snapshot(self) -> Dict[str, Any]:
        """Capture current belief state for episodic memory."""
        return {
            'A_params': self.A_params.copy(),
            'B_params': self.B_params.copy(),
            'D_params': self.D_params.copy(),
            'C': self.C.copy(),
            'evidence': self.evidence_log.copy(),
        }
    
    def restore(self, snapshot: Dict[str, Any]):
        """Restore beliefs from a snapshot."""
        self.A_params = snapshot['A_params'].copy()
        self.B_params = snapshot['B_params'].copy()
        self.D_params = snapshot['D_params'].copy()
        self.C = snapshot['C'].copy()



