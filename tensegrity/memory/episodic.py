"""
Episodic Memory: Temporal sequences of experiences.

Stores trajectories of (observation, belief_state, action, surprise, timestamp)
tuples. Retrieval is Zipf-weighted: frequently revisited episodes are cheaper
to access, and recent episodes have priority (recency bias).

The episodic memory enables:
  1. Temporal pattern detection (repeated sequences → expectations)
  2. Counterfactual reasoning ("what if I had acted differently at time t?")
  3. Experience replay for belief updating (re-process past observations)
  4. Context-dependent retrieval (current observation cues related episodes)

Mathematical basis:
  Temporal Context Model (Howard & Kahana, 2002):
  Context vector c_t evolves as: c_t = ρ * c_{t-1} + β * f(item_t)
  Where ρ = drift rate, β = encoding strength.
  Retrieval: P(recall item_i) ∝ cos(c_retrieval, c_encoding_i)
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from collections import deque
import heapq


class Episode:
    """A single episode: a snapshot of the agent's experience at one moment."""
    
    __slots__ = ['timestamp', 'observation', 'morton_code', 'belief_state',
                 'action', 'surprise', 'free_energy', 'context_vector',
                 'access_count', 'metadata']
    
    def __init__(self, timestamp: int, observation: np.ndarray,
                 morton_code: np.ndarray, belief_state: np.ndarray,
                 action: Optional[int], surprise: float,
                 free_energy: float, context_vector: np.ndarray,
                 metadata: Optional[Dict] = None):
        self.timestamp = timestamp
        self.observation = observation
        self.morton_code = morton_code
        self.belief_state = belief_state
        self.action = action
        self.surprise = surprise
        self.free_energy = free_energy
        self.context_vector = context_vector
        self.access_count = 0
        self.metadata = metadata or {}
    
    def __repr__(self):
        return (f"Episode(t={self.timestamp}, surprise={self.surprise:.3f}, "
                f"F={self.free_energy:.3f}, accesses={self.access_count})")


class EpisodicMemory:
    """
    Temporal experience store with context-dependent retrieval.
    
    Uses the Temporal Context Model: each episode is tagged with a
    context vector that drifts over time. Retrieval similarity is
    computed between the current context and stored contexts.
    
    Zipf law: access counts follow power-law distribution. Frequently
    accessed episodes become "consolidated" (cheaper to retrieve).
    """
    
    def __init__(self, context_dim: int = 64, capacity: int = 10000,
                 drift_rate: float = 0.95, encoding_strength: float = 0.3,
                 zipf_exponent: float = 1.0):
        """
        Args:
            context_dim: Dimensionality of the context vector
            capacity: Maximum number of episodes to store
            drift_rate: ρ — how much context persists between timesteps
            encoding_strength: β — how strongly new items update context
            zipf_exponent: Controls power-law retrieval priority
        """
        self.context_dim = context_dim
        self.capacity = capacity
        self.drift_rate = drift_rate
        self.encoding_strength = encoding_strength
        self.zipf_s = zipf_exponent
        
        # Current context vector (evolves over time)
        self.context = np.random.randn(context_dim)
        self.context /= np.linalg.norm(self.context)
        
        # Episode store
        self.episodes: List[Episode] = []
        
        # Index: morton_code → list of episode indices (for content-addressed retrieval)
        self._morton_index: Dict[int, List[int]] = {}
        
        # Surprise-sorted index (for retrieving most surprising episodes)
        self._surprise_heap: List[Tuple[float, int]] = []  # (neg_surprise, idx)
        
        self._timestep = 0
    
    def _compute_item_representation(self, observation: np.ndarray,
                                     belief_state: np.ndarray) -> np.ndarray:
        """
        Project observation + belief into context space.
        
        Uses a deterministic hash-like projection (no learned weights):
        f(item) = normalize(W_obs @ obs + W_belief @ belief)
        where W is a fixed random projection matrix (Johnson-Lindenstrauss).
        """
        # Deterministic random projection (seeded by dimension sizes)
        rng_obs = np.random.RandomState(42)
        rng_bel = np.random.RandomState(43)
        
        obs_flat = observation.flatten()
        bel_flat = belief_state.flatten()
        
        # Random projection to context_dim
        W_obs = rng_obs.randn(self.context_dim, len(obs_flat)) / np.sqrt(len(obs_flat))
        W_bel = rng_bel.randn(self.context_dim, len(bel_flat)) / np.sqrt(len(bel_flat))
        
        item_rep = W_obs @ obs_flat + W_bel @ bel_flat
        norm = np.linalg.norm(item_rep)
        if norm > 0:
            item_rep /= norm
        return item_rep
    
    def encode(self, observation: np.ndarray, morton_code: np.ndarray,
               belief_state: np.ndarray, action: Optional[int],
               surprise: float, free_energy: float,
               metadata: Optional[Dict] = None) -> Episode:
        """
        Encode a new episode into memory.
        
        1. Compute item representation
        2. Update temporal context: c_t = ρ * c_{t-1} + β * f(item)
        3. Store episode with current context
        4. Update indices
        """
        # Compute item representation
        item_rep = self._compute_item_representation(observation, belief_state)
        
        # Update context (temporal drift + new encoding)
        self.context = (self.drift_rate * self.context + 
                       self.encoding_strength * item_rep)
        norm = np.linalg.norm(self.context)
        if norm > 0:
            self.context /= norm
        
        # Create episode
        episode = Episode(
            timestamp=self._timestep,
            observation=observation.copy(),
            morton_code=morton_code.copy(),
            belief_state=belief_state.copy(),
            action=action,
            surprise=surprise,
            free_energy=free_energy,
            context_vector=self.context.copy(),
            metadata=metadata
        )
        
        # Store
        idx = len(self.episodes)
        self.episodes.append(episode)
        
        # Update morton index
        for code in (morton_code if morton_code.ndim > 0 else [morton_code]):
            code_int = int(code)
            if code_int not in self._morton_index:
                self._morton_index[code_int] = []
            self._morton_index[code_int].append(idx)
        
        # Update surprise heap
        heapq.heappush(self._surprise_heap, (-surprise, idx))
        
        # Capacity management: remove oldest low-access episodes
        if len(self.episodes) > self.capacity:
            self._consolidate()
        
        self._timestep += 1
        return episode
    
    def retrieve_by_context(self, query_context: Optional[np.ndarray] = None,
                           k: int = 5) -> List[Episode]:
        """
        Retrieve episodes most similar to the query context.
        
        If no query provided, uses current temporal context.
        Similarity = cosine(query, episode_context) * zipf_weight
        """
        if query_context is None:
            query_context = self.context
        
        if not self.episodes:
            return []
        
        # Compute similarities with Zipf weighting
        scored = []
        for i, ep in enumerate(self.episodes):
            # Cosine similarity
            sim = np.dot(query_context, ep.context_vector)
            
            # Zipf weight based on access rank
            # More accessed = higher priority (but diminishing returns)
            zipf_weight = 1.0 + np.log1p(ep.access_count)
            
            # Recency bonus (exponential decay)
            recency = np.exp(-0.01 * (self._timestep - ep.timestamp))
            
            score = sim * zipf_weight * recency
            scored.append((score, i))
        
        # Top-k
        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:k]:
            ep = self.episodes[idx]
            ep.access_count += 1
            results.append(ep)
        
        return results
    
    def retrieve_by_morton(self, morton_code: int, k: int = 5) -> List[Episode]:
        """
        Content-addressed retrieval: find episodes with similar Morton codes.
        
        Exact match first, then neighbors in Morton space.
        """
        indices = set()
        
        # Exact match
        if morton_code in self._morton_index:
            indices.update(self._morton_index[morton_code])
        
        # If not enough, look for nearby codes (bit-flip neighbors)
        if len(indices) < k:
            for stored_code in self._morton_index:
                # XOR proximity
                xor_dist = bin(morton_code ^ stored_code).count('1')
                if xor_dist <= 4:  # Within 4 bit-flips
                    indices.update(self._morton_index[stored_code])
                if len(indices) >= k * 3:
                    break
        
        # Score by recency and access count
        episodes = [(self.episodes[i], i) for i in indices if i < len(self.episodes)]
        episodes.sort(key=lambda x: x[0].timestamp, reverse=True)
        
        results = []
        for ep, idx in episodes[:k]:
            ep.access_count += 1
            results.append(ep)
        
        return results
    
    def retrieve_most_surprising(self, k: int = 5) -> List[Episode]:
        """Retrieve the k most surprising episodes ever encountered."""
        # Rebuild if needed
        sorted_eps = sorted(self.episodes, key=lambda e: e.surprise, reverse=True)
        results = sorted_eps[:k]
        for ep in results:
            ep.access_count += 1
        return results
    
    def replay(self, n: int = 10) -> List[Episode]:
        """
        Experience replay: sample episodes weighted by surprise * recency.
        
        This provides training signal for the epistemic memory's Dirichlet updates.
        High-surprise episodes are more informative (more Bayesian update).
        """
        if not self.episodes:
            return []
        
        weights = np.array([
            ep.surprise * np.exp(-0.005 * (self._timestep - ep.timestamp))
            for ep in self.episodes
        ])
        weights = np.maximum(weights, 1e-16)
        weights /= weights.sum()
        
        indices = np.random.choice(len(self.episodes), size=min(n, len(self.episodes)),
                                   replace=False, p=weights)
        
        results = [self.episodes[i] for i in indices]
        for ep in results:
            ep.access_count += 1
        return results
    
    def _consolidate(self):
        """
        Remove low-value episodes when capacity is exceeded.
        
        Value = access_count * surprise * recency_weight
        Keeps high-value episodes (Zipf: frequently accessed survive).
        """
        if len(self.episodes) <= self.capacity:
            return
        
        # Score each episode
        scores = []
        for i, ep in enumerate(self.episodes):
            recency = np.exp(-0.01 * (self._timestep - ep.timestamp))
            value = (1 + ep.access_count) * ep.surprise * recency
            scores.append((value, i))
        
        # Keep top capacity episodes
        scores.sort(reverse=True)
        keep_indices = set(idx for _, idx in scores[:self.capacity])
        
        # Rebuild
        new_episodes = [ep for i, ep in enumerate(self.episodes) if i in keep_indices]
        self.episodes = new_episodes
        
        # Rebuild indices
        self._morton_index.clear()
        self._surprise_heap.clear()
        for i, ep in enumerate(self.episodes):
            for code in (ep.morton_code if ep.morton_code.ndim > 0 else [ep.morton_code]):
                code_int = int(code)
                if code_int not in self._morton_index:
                    self._morton_index[code_int] = []
                self._morton_index[code_int].append(i)
            heapq.heappush(self._surprise_heap, (-ep.surprise, i))
    
    def get_sequence(self, start_t: int, end_t: int) -> List[Episode]:
        """Get a temporal sequence of episodes."""
        return [ep for ep in self.episodes 
                if start_t <= ep.timestamp <= end_t]
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Memory statistics."""
        if not self.episodes:
            return {'count': 0}
        
        accesses = [ep.access_count for ep in self.episodes]
        surprises = [ep.surprise for ep in self.episodes]
        
        return {
            'count': len(self.episodes),
            'mean_access_count': np.mean(accesses),
            'max_access_count': np.max(accesses),
            'mean_surprise': np.mean(surprises),
            'max_surprise': np.max(surprises),
            'zipf_exponent': self.zipf_s,
            'context_drift': self.drift_rate,
        }
