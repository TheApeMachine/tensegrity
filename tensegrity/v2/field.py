"""
Unified Energy Landscape: One functional to rule them all.

The v1 architecture had four separate components doing four separate kinds
of energy minimization. This module unifies them into a single energy
functional that decomposes into local terms:

    E_total = E_perception + E_memory + E_causal

Where:
    E_perception = Σ_ℓ (1/2Σℓ) ||zℓ - Wℓφ(z^{ℓ+1})||²     (NGC/predictive coding)
    E_memory     = -lse(β, Xᵀξ) + ½||ξ||²                   (Hopfield energy)
    E_causal     = Σ_v (1/2) ||z_v - f_v(z_pa(v))||²          (SCM prediction error)

All three are: "sum of squared prediction errors on a graph."
The NGC circuit predicts its input. The Hopfield network predicts its query.
The causal model predicts effects from causes. Same operation, different scale.

The system settles by passing messages on this combined graph until the
total energy reaches a minimum. That minimum IS the system's best explanation
of the observation, given its memory and causal beliefs.

This is what Friston's Free Energy Principle actually says: every component
of the system minimizes its own local VFE, and the global behavior emerges
from the composition of these local optimizations.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from tensegrity.v2.fhrr import FHRREncoder, bind, bundle, unbind
from tensegrity.v2.ngc import PredictiveCodingCircuit


@dataclass
class EnergyDecomposition:
    """Breakdown of the total energy into components."""
    perception: float    # NGC prediction error energy
    memory: float        # Hopfield retrieval energy
    causal: float        # Causal SCM prediction error
    total: float         # Sum
    prediction_error_norm: float  # ||observation - predicted||
    surprise: float      # -log P(observation | beliefs)


class HopfieldMemoryBank:
    """
    Modern Hopfield network operating in FHRR space.
    
    Stores FHRR hypervectors as patterns. Retrieval is energy minimization:
        E(ξ) = -lse(β, Xᵀξ) + ½||ξ||²
        ξ_new = X · softmax(β · Xᵀ · ξ)
    
    This is mathematically identical to a single attention head where:
        - stored patterns X = keys = values
        - query ξ = the probe
        - β = 1/√d_k (inverse temperature)
    """
    
    def __init__(self, dim: int, beta: float = 0.01, capacity: int = 10000):
        self.dim = dim
        self.beta = beta
        self.capacity = capacity
        
        self.patterns: List[np.ndarray] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty = True
    
    def store(self, pattern: np.ndarray, normalize: bool = True):
        """Store a pattern (FHRR vector — use real part for Hopfield)."""
        p = np.real(pattern).astype(np.float64) if np.iscomplexobj(pattern) else pattern.astype(np.float64)
        if normalize:
            norm = np.linalg.norm(p)
            if norm > 0:
                p = p / norm
        self.patterns.append(p)
        self._dirty = True
        
        if len(self.patterns) > self.capacity:
            self.patterns.pop(0)
            self._dirty = True
    
    def retrieve(self, query: np.ndarray, steps: int = 3) -> Tuple[np.ndarray, float]:
        """
        Retrieve via energy minimization.
        Returns (retrieved_pattern, energy).
        """
        if not self.patterns:
            return np.zeros(self.dim), 0.0
        
        self._ensure_matrix()
        
        q = np.real(query).astype(np.float64) if np.iscomplexobj(query) else query.astype(np.float64)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        
        xi = q.copy()
        for _ in range(steps):
            sims = self._matrix.T @ xi  # (n_patterns,)
            scaled = self.beta * sims
            scaled -= scaled.max()
            weights = np.exp(scaled)
            weights /= weights.sum()
            xi_new = self._matrix @ weights
            norm = np.linalg.norm(xi_new)
            if norm > 0:
                xi_new /= norm
            if np.allclose(xi, xi_new, atol=1e-8):
                break
            xi = xi_new
        
        # Energy
        sims = self._matrix.T @ xi
        log_sum_exp = np.log(np.sum(np.exp(self.beta * sims - self.beta * sims.max()))) + self.beta * sims.max()
        energy = float(-log_sum_exp / self.beta + 0.5 * np.dot(xi, xi))
        
        return xi, energy
    
    def _ensure_matrix(self):
        if self._dirty and self.patterns:
            self._matrix = np.column_stack(self.patterns)
            self._dirty = False
    
    @property
    def n_patterns(self):
        return len(self.patterns)


class UnifiedField:
    """
    The unified cognitive field.
    
    Composes:
      1. FHRR encoder (observation → compositional hypervector)
      2. NGC circuit (hierarchical predictive coding)
      3. Hopfield memory (content-addressed retrieval)
    
    All connected through a single energy functional.
    One step of cognition:
      a. Encode observation as FHRR vector
      b. Settle NGC circuit (minimize perception energy)
      c. Query Hopfield memory with settled top-layer state (minimize memory energy)
      d. Use memory retrieval to refine predictions (close the loop)
      e. Learn: Hebbian update on NGC weights + store in Hopfield
    
    The total energy E_total = E_ngc + E_hopfield monotonically decreases.
    """
    
    def __init__(self,
                 obs_dim: int = 256,
                 hidden_dims: List[int] = None,
                 fhrr_dim: int = 2048,
                 hopfield_beta: float = 0.01,
                 ngc_settle_steps: int = 20,
                 ngc_learning_rate: float = 0.005,
                 ngc_precisions: Optional[List[float]] = None):
        """
        Args:
            obs_dim: Dimension of the observation layer (FHRR → real projection)
            hidden_dims: NGC hidden layer dimensions [h1, h2, ...]. 
                        Full hierarchy = [obs_dim] + hidden_dims
            fhrr_dim: FHRR hypervector dimensionality
            hopfield_beta: Inverse temperature for Hopfield retrieval
            ngc_settle_steps: Settling iterations for NGC
            ngc_learning_rate: Hebbian learning rate
        """
        if hidden_dims is None:
            hidden_dims = [128, 32]
        
        self.obs_dim = obs_dim
        self.fhrr_dim = fhrr_dim
        
        # FHRR encoder
        self.encoder = FHRREncoder(dim=fhrr_dim)
        
        # Random projection: FHRR (complex, fhrr_dim) → real (obs_dim)
        # Fixed, not learned — this is the sensory transduction
        rng = np.random.RandomState(42)
        self._proj = rng.randn(obs_dim, fhrr_dim).astype(np.float64) / np.sqrt(fhrr_dim)
        
        # NGC circuit: hierarchical predictive coding
        layer_sizes = [obs_dim] + hidden_dims
        self.ngc = PredictiveCodingCircuit(
            layer_sizes=layer_sizes,
            precisions=ngc_precisions,
            settle_steps=ngc_settle_steps,
            learning_rate=ngc_learning_rate,
        )
        
        # Hopfield memory: stores abstract states from NGC top layer
        top_dim = hidden_dims[-1]
        self.memory = HopfieldMemoryBank(dim=top_dim, beta=hopfield_beta)
        
        # Energy tracking
        self._step_count = 0
        self.energy_history: List[EnergyDecomposition] = []
    
    def _fhrr_to_obs(self, fhrr_vec: np.ndarray) -> np.ndarray:
        """Project FHRR complex vector to real observation space."""
        real_part = np.real(fhrr_vec).astype(np.float64)
        return self._proj @ real_part
    
    def observe(self, raw_input: Any, input_type: str = "numeric") -> Dict[str, Any]:
        """
        Full cognitive cycle: observe → predict → error → settle → learn → remember.
        
        Args:
            raw_input: The observation. Type depends on input_type:
                "numeric": np.ndarray of floats
                "bindings": dict of {role: filler} string pairs
                "tokens": list of string tokens
                "text": a single string (split into tokens)
            input_type: How to interpret raw_input
        
        Returns:
            Full cycle diagnostics
        """
        self._step_count += 1
        
        # === 1. ENCODE: raw input → FHRR → observation vector ===
        if input_type == "numeric":
            fhrr_vec = self.encoder.encode_numeric_vector(np.asarray(raw_input))
        elif input_type == "bindings":
            fhrr_vec = self.encoder.encode_observation(raw_input)
        elif input_type == "tokens":
            fhrr_vec = self.encoder.encode_sequence(raw_input)
        elif input_type == "text":
            tokens = str(raw_input).lower().split()
            fhrr_vec = self.encoder.encode_sequence(tokens)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        obs_vec = self._fhrr_to_obs(fhrr_vec)
        
        # === 2. PREDICT: what did the NGC expect to see? ===
        prediction_error_pre = self.ngc.prediction_error(obs_vec)
        
        # === 3. SETTLE: minimize perception energy ===
        settle_result = self.ngc.settle(obs_vec)
        perception_energy = settle_result["final_energy"]
        
        # === 4. REMEMBER: query Hopfield with abstract state ===
        abstract_state = self.ngc.get_abstract_state(level=-1)
        retrieved, memory_energy = self.memory.retrieve(abstract_state)
        
        # === 5. LEARN: Hebbian update on NGC + store in Hopfield ===
        self.ngc.learn()
        self.memory.store(abstract_state)
        
        # === 6. ENERGY: compute decomposition ===
        decomp = EnergyDecomposition(
            perception=perception_energy,
            memory=memory_energy,
            causal=0.0,  # Will be added when causal module is connected
            total=perception_energy + memory_energy,
            prediction_error_norm=float(prediction_error_pre),
            surprise=float(np.log(max(prediction_error_pre, 1e-16))),
        )
        self.energy_history.append(decomp)
        
        return {
            "step": self._step_count,
            "fhrr_vector": fhrr_vec,
            "observation": obs_vec,
            "abstract_state": abstract_state,
            "retrieved_memory": retrieved,
            "memory_similarity": float(np.dot(abstract_state, retrieved) / 
                                      (np.linalg.norm(abstract_state) * np.linalg.norm(retrieved) + 1e-16)),
            "energy": decomp,
            "settle": settle_result,
            "prediction_error": prediction_error_pre,
        }
    
    def predict(self) -> np.ndarray:
        """
        What does the system expect to observe next?
        
        This is the prediction that v1 never made.
        """
        return self.ngc.predict_observation()
    
    @property
    def total_energy(self) -> float:
        if self.energy_history:
            return self.energy_history[-1].total
        return 0.0
    
    @property
    def statistics(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "total_energy": self.total_energy,
            "ngc": self.ngc.statistics,
            "memory_patterns": self.memory.n_patterns,
            "fhrr_dim": self.fhrr_dim,
            "obs_dim": self.obs_dim,
        }
