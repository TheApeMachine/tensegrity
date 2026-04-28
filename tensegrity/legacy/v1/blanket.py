"""
Markov Blanket: The computational boundary of the agent.

In Friston's formalism, the Markov blanket separates internal states (beliefs)
from external states (world). It consists of:
  - Sensory states (S): what flows IN from the world (observations)
  - Active states (A): what flows OUT to the world (actions)

The blanket enforces conditional independence:
  Internal ⊥ External | Blanket

This is not a metaphor. It's the literal statistical boundary that defines
where the agent ends and the world begins. The blanket nodes are the ONLY
points of contact between the agent's belief states and external reality.

Implementation: The blanket manages the flow of Morton-coded observations
in and action selections out. It also maintains the observation buffer
that feeds into the free energy engine.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from collections import deque

from tensegrity.legacy.v1.morton import MortonEncoder


class MarkovBlanket:
    """
    The agent's interface with the world.
    
    Sensory states receive Morton-coded observations.
    Active states emit discrete actions.
    
    The blanket enforces the Markov property: internal states
    are conditionally independent of external states given the blanket.
    """
    
    def __init__(self, 
                 encoder: MortonEncoder,
                 n_sensory_channels: int = 1,
                 n_active_channels: int = 1,
                 observation_buffer_size: int = 64):
        """
        Args:
            encoder: MortonEncoder for sensory preprocessing
            n_sensory_channels: Number of parallel sensory channels
            n_active_channels: Number of action dimensions
            observation_buffer_size: How many past observations to retain
        """
        self.encoder = encoder
        self.n_sensory = n_sensory_channels
        self.n_active = n_active_channels
        
        # Current blanket state
        self.sensory_state: Optional[np.ndarray] = None  # Morton codes
        self.active_state: Optional[np.ndarray] = None    # Action indices
        
        # Observation buffer — recent history for temporal inference
        self.observation_buffer: deque = deque(maxlen=observation_buffer_size)
        
        # Statistics for the blanket boundary (running means/vars for normalization)
        self._obs_count = 0
        self._obs_sum = None
        self._obs_sq_sum = None
        
        # Blanket surprise (how unexpected was the last observation?)
        self.surprise: float = 0.0
    
    def sense(self, raw_observation: np.ndarray) -> np.ndarray:
        """
        Process a raw observation through the sensory boundary.
        
        1. Morton-encode the raw data
        2. Update the observation buffer
        3. Compute surprise (deviation from running statistics)
        
        Args:
            raw_observation: Raw sensory data, shape depends on modality.
                           Will be reshaped to (n_points, n_dims) for Morton encoding.
        
        Returns:
            Morton-coded observation as integer array
        """
        # Ensure proper shape for Morton encoding
        if raw_observation.ndim == 1:
            if len(raw_observation) == self.encoder.n_dims:
                raw_observation = raw_observation.reshape(1, -1)
            else:
                # Treat as multiple single-dim observations
                raw_observation = raw_observation.reshape(-1, 1)
        
        # Morton encode
        morton_codes = self.encoder.encode_continuous(raw_observation)
        if isinstance(morton_codes, (int, np.integer)):
            morton_codes = np.array([morton_codes])
        
        # Update running statistics for surprise computation
        self._update_statistics(raw_observation)
        
        # Compute surprise: -log P(observation) under running model
        self.surprise = self._compute_surprise(raw_observation)
        
        # Store in buffer
        self.sensory_state = morton_codes
        self.observation_buffer.append({
            'morton': morton_codes.copy(),
            'raw': raw_observation.copy(),
            'surprise': self.surprise,
            'timestamp': self._obs_count
        })
        
        return morton_codes
    
    def act(self, action_distribution: np.ndarray) -> int:
        """
        Select an action through the active boundary.
        
        The action is sampled from the distribution provided by the
        inference engine (policy = softmax over expected free energies).
        
        Args:
            action_distribution: Probability distribution over actions.
        
        Returns:
            Selected action index.
        """
        # Ensure valid distribution
        action_distribution = np.asarray(action_distribution, dtype=np.float64)
        action_distribution = np.maximum(action_distribution, 1e-16)
        action_distribution /= action_distribution.sum()
        
        # Sample action
        action = np.random.choice(len(action_distribution), p=action_distribution)
        self.active_state = np.array([action])
        return int(action)
    
    def _update_statistics(self, observation: np.ndarray):
        """Update running statistics for surprise computation."""
        flat = observation.flatten()
        self._obs_count += 1
        
        if self._obs_sum is None:
            self._obs_sum = np.zeros_like(flat, dtype=np.float64)
            self._obs_sq_sum = np.zeros_like(flat, dtype=np.float64)
        
        # Pad or truncate to match
        n = min(len(flat), len(self._obs_sum))
        self._obs_sum[:n] += flat[:n]
        self._obs_sq_sum[:n] += flat[:n] ** 2
    
    def _compute_surprise(self, observation: np.ndarray) -> float:
        """
        Compute Bayesian surprise: -log P(o) under running Gaussian model.
        
        This is a simple proxy — the full surprise comes from the
        free energy engine. But this gives a fast heuristic at the boundary.
        """
        if self._obs_count < 2:
            return 0.0
        
        flat = observation.flatten()
        n = min(len(flat), len(self._obs_sum))
        
        mean = self._obs_sum[:n] / self._obs_count
        var = self._obs_sq_sum[:n] / self._obs_count - mean ** 2
        var = np.maximum(var, 1e-8)  # Prevent division by zero
        
        # Gaussian log-likelihood (negative = surprise)
        log_prob = -0.5 * np.sum(((flat[:n] - mean) ** 2) / var + np.log(2 * np.pi * var))
        return float(-log_prob)  # Higher = more surprising
    
    def get_observation_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the last n observations from the buffer."""
        if n is None:
            return list(self.observation_buffer)
        return list(self.observation_buffer)[-n:]
    
    def get_surprise_trajectory(self) -> np.ndarray:
        """Get the surprise values over time."""
        return np.array([obs['surprise'] for obs in self.observation_buffer])
    
    @property
    def state(self) -> Dict[str, Any]:
        """Current blanket state summary."""
        return {
            'sensory': self.sensory_state,
            'active': self.active_state,
            'surprise': self.surprise,
            'obs_count': self._obs_count,
            'buffer_size': len(self.observation_buffer)
        }


