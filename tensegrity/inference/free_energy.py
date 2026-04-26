"""
Free Energy Engine: Variational Free Energy Minimization without Gradients.

This is the core inference mechanism. It implements the discrete-state-space
active inference algorithm from Friston's POMDP formalism.

The variational free energy F is:
  F = E_q[ln q(s) - ln p(o, s)]
    = D_KL[q(s) || p(s)] - E_q[ln p(o | s)]
    = Complexity - Accuracy

Minimizing F is equivalent to maximizing the ELBO (Evidence Lower Bound).
But we do it WITHOUT gradient descent. Instead:

  1. PERCEPTION: Fixed-point iteration on q(s) until convergence
     ln q(s_t) ← ln(A^T · o_t) + ln(B_{a_{t-1}}^T · q(s_{t-1}))
     Then softmax-normalize. Repeat ~16 iterations.

  2. PLANNING: Compute Expected Free Energy G_π for each policy
     G_π = Σ_τ [ambiguity(τ) + risk(τ)]
     Where ambiguity = H[P(o|s)] weighted by q(s)  (epistemic drive)
           risk = D_KL[q(o|π) || p̃(o)]           (pragmatic drive)

  3. ACTION: Q(π) ∝ exp(-γ · G_π), then marginalize over policies

This is coordinate ascent on F — each step is guaranteed to decrease F.
The algorithm is the sum-product algorithm on the POMDP factor graph.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from scipy.special import softmax, digamma
import logging

logger = logging.getLogger(__name__)


class FreeEnergyEngine:
    """
    Discrete Active Inference engine.
    
    Implements the full perception-planning-action loop using only:
      - Matrix multiplication
      - Softmax normalization  
      - Fixed-point iteration
      - Dirichlet counting updates
    
    Zero gradient computation. Zero backpropagation.
    """
    
    def __init__(self, n_states: int, n_observations: int, n_actions: int,
                 planning_horizon: int = 3, n_policies: Optional[int] = None,
                 precision: float = 4.0, perception_iterations: int = 16,
                 policy_depth: int = 2):
        """
        Args:
            n_states: Number of hidden states
            n_observations: Number of observation categories
            n_actions: Number of actions
            planning_horizon: How far ahead to plan
            n_policies: Number of policies to evaluate. If None, enumerate all.
            precision: γ — inverse temperature for policy selection
            perception_iterations: Number of fixed-point iterations for state inference
            policy_depth: Action-sequence length per policy
        """
        self.n_states = n_states
        self.n_obs = n_observations
        self.n_actions = n_actions
        self.horizon = planning_horizon
        self.precision = precision
        self.n_iter = perception_iterations
        self.policy_depth = policy_depth
        
        # Generate policy space (enumerate action sequences)
        if n_policies is None:
            # Full enumeration (exponential — only feasible for small action spaces)
            self.policies = self._enumerate_policies()
        else:
            # Random sample of policies
            self.policies = self._sample_policies(n_policies)
        
        self.n_policies = len(self.policies)
        
        # State: current beliefs
        self.q_states = np.ones(n_states) / n_states  # Uniform prior
        self.q_policies = np.ones(self.n_policies) / self.n_policies
        self.G = np.zeros(self.n_policies)  # Expected free energies
        
        # Free energy tracking
        self.F_history: List[float] = []
        self.G_history: List[np.ndarray] = []
        
        # Previous action (for transition model)
        self.prev_action: Optional[int] = None
    
    def _enumerate_policies(self) -> np.ndarray:
        """
        Enumerate all possible action sequences of length policy_depth.
        
        For n_actions=4, depth=2: 4^2 = 16 policies.
        """
        depth = min(self.policy_depth, 3)  # Cap to prevent explosion
        n = self.n_actions ** depth
        
        policies = np.zeros((n, depth), dtype=np.int64)
        for i in range(n):
            for d in range(depth):
                policies[i, d] = (i // (self.n_actions ** d)) % self.n_actions
        
        return policies
    
    def _sample_policies(self, n: int) -> np.ndarray:
        """Sample random policies."""
        return np.random.randint(0, self.n_actions, size=(n, self.policy_depth))
    
    def infer_states(self, observation: int,
                     A: np.ndarray, B: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        PERCEPTION: Infer hidden states via fixed-point VFE minimization.
        
        The update equation:
          ln q(s_t) ← ln A[o_t, :] + ln(B[:, :, a_{t-1}]^T · q(s_{t-1}))
        
        Then softmax-normalize. Repeat until convergence.
        
        This is NOT gradient descent. It's coordinate ascent on the
        variational free energy, which is equivalent to belief propagation
        on the POMDP factor graph.
        
        Args:
            observation: Index of observed category (0 to n_obs-1)
            A: Likelihood matrix P(o|s), shape (n_obs, n_states)
            B: Transition tensor P(s'|s,a), shape (n_states, n_states, n_actions)
            D: Prior over initial states, shape (n_states,)
        
        Returns:
            q_states: Posterior belief over hidden states, shape (n_states,)
        """
        # Observation likelihood: ln P(o_t | s_t) for each state
        log_likelihood = np.log(np.maximum(A[observation, :], 1e-16))
        
        # Prior from transition model
        if self.prev_action is not None:
            # ln P(s_t | s_{t-1}, a_{t-1}) marginalized over s_{t-1}
            transition = B[:, :, self.prev_action]  # (n_states x n_states)
            predicted = transition @ self.q_states   # (n_states,)
            log_prior = np.log(np.maximum(predicted, 1e-16))
        else:
            log_prior = np.log(np.maximum(D, 1e-16))
        
        # Fixed-point iteration
        log_q = log_prior + log_likelihood
        
        for iteration in range(self.n_iter):
            # Softmax normalize
            q_new = softmax(log_q)
            
            # Recompute with updated beliefs
            if self.prev_action is not None:
                predicted = B[:, :, self.prev_action] @ q_new
                log_prior = np.log(np.maximum(predicted, 1e-16))
            
            log_q_new = log_prior + log_likelihood
            
            # Check convergence
            if np.allclose(softmax(log_q), softmax(log_q_new), atol=1e-8):
                break
            
            log_q = log_q_new
        
        self.q_states = softmax(log_q)
        return self.q_states
    
    def evaluate_policies(self, A: np.ndarray, B: np.ndarray,
                          C: np.ndarray, log_A: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PLANNING: Compute Expected Free Energy G_π for each policy.
        
        G_π = Σ_τ [ ambiguity(τ, π) + risk(τ, π) ]
        
        Where:
          ambiguity = E_{q(s_τ|π)}[H[P(o_τ|s_τ)]]    — epistemic drive
          risk      = D_KL[q(o_τ|π) || p̃(o_τ)]        — pragmatic drive
        
        In matrix form:
          ambiguity = Σ_τ q(s_τ|π)^T · H(A)   where H(A)_s = -Σ_o A[o,s] ln A[o,s]
          risk      = Σ_τ D_KL[A · q(s_τ|π) || σ(C)]
        
        Args:
            A: Likelihood matrix (n_obs, n_states)
            B: Transition tensor (n_states, n_states, n_actions)
            C: Log preference over observations (n_obs,)
            log_A: Expected log-likelihood (from Dirichlet) — uses digamma if provided
        
        Returns:
            (G, q_pi): Expected free energies and policy posterior
        """
        # Precompute entropy of each column of A: H[P(o|s)]
        if log_A is not None:
            H_A = -np.sum(A * log_A, axis=0)  # (n_states,)
        else:
            H_A = -np.sum(A * np.log(np.maximum(A, 1e-16)), axis=0)
        
        # Preference distribution (softmax of C)
        p_preferred = softmax(C)
        
        G = np.zeros(self.n_policies)
        
        for pi_idx, policy in enumerate(self.policies):
            G_pi = 0.0
            q_s = self.q_states.copy()  # Start from current belief
            
            for tau in range(min(len(policy), self.horizon)):
                action = policy[tau]
                
                # Predict next state: q(s_{τ+1} | π) = B_a · q(s_τ | π)
                q_s = B[:, :, action] @ q_s
                q_s = np.maximum(q_s, 1e-16)
                q_s /= q_s.sum()
                
                # Predict observation: q(o_τ | π) = A · q(s_τ | π)
                q_o = A @ q_s
                q_o = np.maximum(q_o, 1e-16)
                q_o /= q_o.sum()
                
                # AMBIGUITY (epistemic value): E[H[P(o|s)]]
                # How uncertain are the observations given the predicted states?
                ambiguity = np.dot(q_s, H_A)
                
                # RISK (pragmatic value): D_KL[q(o|π) || preferred]
                risk = np.sum(q_o * np.log(q_o / np.maximum(p_preferred, 1e-16)))
                
                G_pi += ambiguity + risk
            
            G[pi_idx] = G_pi
        
        self.G = G
        self.G_history.append(G.copy())
        
        # Policy posterior: Q(π) ∝ exp(-γ · G_π)
        log_q_pi = -self.precision * G
        self.q_policies = softmax(log_q_pi)
        
        return G, self.q_policies
    
    def select_action(self) -> Tuple[int, float]:
        """
        ACTION: Select action by marginalizing over policies.
        
        P(a_t) = Σ_π Q(π) · I(π_t = a_t)
        
        Returns:
            (action, confidence): Selected action and its probability
        """
        # Marginalize: P(a_t = a) = Σ_π Q(π) for policies where π[0] = a
        action_probs = np.zeros(self.n_actions)
        
        for pi_idx, policy in enumerate(self.policies):
            if len(policy) > 0:
                action_probs[policy[0]] += self.q_policies[pi_idx]
        
        # Normalize
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        else:
            action_probs = np.ones(self.n_actions) / self.n_actions
        
        # Sample (or argmax for exploitation)
        action = np.random.choice(self.n_actions, p=action_probs)
        confidence = float(action_probs[action])
        
        self.prev_action = action
        return int(action), confidence
    
    def compute_free_energy(self, observation: int,
                           A: np.ndarray, D: np.ndarray) -> float:
        """
        Compute the variational free energy F for the current belief state.
        
        F = E_q[ln q(s)] - E_q[ln p(o, s)]
          = D_KL[q(s) || p(s)] - E_q[ln p(o | s)]
          = Complexity - Accuracy
        """
        q = self.q_states
        
        # Accuracy: E_q[ln P(o | s)]
        log_A = np.log(np.maximum(A[observation, :], 1e-16))
        accuracy = np.dot(q, log_A)
        
        # Complexity: D_KL[q(s) || p(s)]  where p(s) = D
        log_q = np.log(np.maximum(q, 1e-16))
        log_p = np.log(np.maximum(D, 1e-16))
        complexity = np.sum(q * (log_q - log_p))
        
        F = complexity - accuracy
        self.F_history.append(float(F))
        
        return float(F)
    
    def step(self, observation: int, 
             A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
             log_A: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Full inference step: perceive → plan → act.
        
        This is one tick of the active inference loop.
        No gradients anywhere.
        """
        # 1. PERCEIVE: Update beliefs about hidden states
        q_states = self.infer_states(observation, A, B, D)
        
        # 2. EVALUATE: Compute free energy
        F = self.compute_free_energy(observation, A, D)
        
        # 3. PLAN: Evaluate policies via Expected Free Energy
        G, q_pi = self.evaluate_policies(A, B, C, log_A)
        
        # 4. ACT: Select action
        action, confidence = self.select_action()
        
        return {
            'belief_state': q_states.copy(),
            'free_energy': F,
            'expected_free_energies': G.copy(),
            'policy_posterior': q_pi.copy(),
            'action': action,
            'action_confidence': confidence,
            'observation': observation,
        }
    
    @property
    def epistemic_value(self) -> float:
        """How much the system wants to explore (reduce uncertainty)."""
        # Entropy of belief state — high entropy = high epistemic drive
        q = self.q_states
        return float(-np.sum(q * np.log(np.maximum(q, 1e-16))))
    
    @property
    def pragmatic_value(self) -> float:
        """How well the system is achieving its preferences."""
        if not self.F_history:
            return 0.0
        # Negative free energy trend — decreasing F = good pragmatic performance
        if len(self.F_history) < 2:
            return 0.0
        return float(self.F_history[-2] - self.F_history[-1])
    
    @property
    def statistics(self) -> Dict[str, Any]:
        return {
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'n_policies': self.n_policies,
            'precision': self.precision,
            'current_F': self.F_history[-1] if self.F_history else None,
            'epistemic_value': self.epistemic_value,
            'pragmatic_value': self.pragmatic_value,
            'belief_entropy': self.epistemic_value,
            'steps': len(self.F_history),
        }
