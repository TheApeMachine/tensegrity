"""
Hierarchical Predictive Coding Circuit (NGC).

This is what was missing. Each layer maintains a belief about its hidden
state and PREDICTS the layer below. The difference between prediction and
actual input is the prediction error. Errors propagate upward. Predictions
propagate downward. The system settles when errors are minimized.

This is Friston's Free Energy Principle implemented as it was meant to be:
not as a flat POMDP solver, but as a hierarchical generative model where
each level explains away the residuals of the level below.

The energy functional (from Ororbia & Kelly, arXiv:2310.15177):

    ℱ(Θ) = Σ_ℓ (1 / 2Σℓ) Σᵢ (zℓᵢ(t) - z̄ℓᵢ)²

Where:
    zℓ      = actual state at layer ℓ  
    z̄ℓ      = W^ℓ · φ(z^{ℓ+1})  = prediction from layer above
    eℓ      = (1/Σℓ)(zℓ - z̄ℓ)    = precision-weighted prediction error
    Σℓ      = precision (confidence) at layer ℓ

State dynamics (settling toward VFE minimum):
    τ · ∂zℓ/∂t = -γ·zℓ + dℓ ⊙ fD(zℓ) - eℓ
    where dℓ = Eℓ · e^{ℓ-1}  (feedback error from below)

Synaptic update (Hebbian, local, no backprop):
    ΔWℓ = eℓ⁻¹ · (zℓ)ᵀ     (prediction errors × pre-synaptic activity)

This is ALL local computation. No gradient chain. Each layer only needs
its own state, the prediction from above, and the error from below.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class LayerState:
    """State of a single predictive coding layer."""
    z: np.ndarray           # State neurons (current belief about this level)
    z_bar: np.ndarray       # Top-down prediction (from layer above)
    error: np.ndarray       # Prediction error: precision * (z - z_bar)
    precision: float        # 1/Σ — how much to trust this layer's errors
    energy: float = 0.0     # Local contribution to VFE
    

class PredictiveCodingCircuit:
    """
    A hierarchical NGC circuit with L layers.
    
    Layer 0 = sensory (clamped to observation)
    Layer L = highest abstraction (prior beliefs)
    
    Information flow:
      Top-down:  z̄ℓ = Wℓ · φ(z^{ℓ+1})     (predictions flow down)
      Bottom-up: eℓ = (1/Σℓ)(zℓ - z̄ℓ)       (errors flow up)
      Lateral:   dℓ = Eℓ · e^{ℓ-1}           (feedback corrections)
    
    All computation is local. No backpropagation.
    Settling minimizes the variational free energy.
    """
    
    def __init__(self, 
                 layer_sizes: List[int],
                 precisions: Optional[List[float]] = None,
                 tau: float = 1.0,
                 gamma: float = 0.01,
                 settle_steps: int = 20,
                 learning_rate: float = 0.01,
                 activation: str = "tanh"):
        """
        Args:
            layer_sizes: [dim_sensory, dim_hidden1, ..., dim_top]
                        e.g., [2048, 512, 128, 32] for a 4-layer hierarchy
            precisions: Per-layer precision (1/variance). Higher = more trusted.
                       If None, defaults to 1.0 everywhere.
            tau: Membrane time constant (settling speed)
            gamma: State decay rate (leaky integration)
            settle_steps: How many steps to run before declaring convergence
            learning_rate: Hebbian learning rate for synaptic updates
            activation: Nonlinearity: "tanh", "relu", "sigmoid", or "linear"
        """
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.tau = tau
        self.gamma = gamma
        self.settle_steps = settle_steps
        self.lr = learning_rate
        
        # Activation function
        self._phi, self._phi_deriv = self._get_activation(activation)
        
        # Precisions (per layer)
        if precisions is None:
            self.precisions = [1.0] * self.n_layers
        else:
            self.precisions = precisions
        
        # Generative weights W[ℓ]: maps layer ℓ+1 → prediction of layer ℓ
        # W[ℓ] has shape (layer_sizes[ℓ], layer_sizes[ℓ+1])
        self.W: List[np.ndarray] = []
        for ell in range(self.n_layers - 1):
            fan_in = layer_sizes[ell + 1]
            fan_out = layer_sizes[ell]
            # Xavier initialization
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            W_ell = np.random.randn(fan_out, fan_in).astype(np.float64) * scale
            self.W.append(W_ell)
        
        # Feedback weights E[ℓ]: maps error at ℓ-1 → correction at ℓ
        # E[ℓ] has shape (layer_sizes[ℓ], layer_sizes[ℓ-1])
        # Initialize as transpose of W (symmetric initialization)
        self.E: List[np.ndarray] = []
        for ell in range(self.n_layers - 1):
            self.E.append(self.W[ell].T.copy())
        
        # Layer states (initialized lazily on first observation)
        self.layers: List[LayerState] = []
        self._initialized = False
        
        # Energy tracking
        self.energy_history: List[float] = []
        self.error_history: List[List[float]] = []  # Per-layer error norms
    
    def _get_activation(self, name: str):
        """Get activation function and its derivative."""
        if name == "tanh":
            return np.tanh, lambda x: 1.0 - np.tanh(x) ** 2
        elif name == "relu":
            return (lambda x: np.maximum(0, x),
                    lambda x: (x > 0).astype(np.float64))
        elif name == "sigmoid":
            sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
            return sig, lambda x: sig(x) * (1 - sig(x))
        elif name == "linear":
            return (lambda x: x, lambda x: np.ones_like(x))
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def _init_layers(self, observation: Optional[np.ndarray] = None):
        """Initialize layer states."""
        self.layers = []
        for ell in range(self.n_layers):
            z = np.zeros(self.layer_sizes[ell], dtype=np.float64)
            if ell == 0 and observation is not None:
                z = observation.copy()
            self.layers.append(LayerState(
                z=z,
                z_bar=np.zeros_like(z),
                error=np.zeros_like(z),
                precision=self.precisions[ell],
            ))
        self._initialized = True
    
    def _predict(self, ell: int) -> np.ndarray:
        """
        Top-down prediction: layer ℓ+1 predicts layer ℓ.
        z̄ℓ = Wℓ · φ(z^{ℓ+1})
        """
        if ell >= self.n_layers - 1:
            # Top layer has no prediction from above — use prior (zero)
            return np.zeros(self.layer_sizes[ell], dtype=np.float64)
        
        z_above = self.layers[ell + 1].z
        return self.W[ell] @ self._phi(z_above)
    
    def _compute_error(self, ell: int) -> np.ndarray:
        """
        Prediction error: eℓ = (1/Σℓ)(zℓ - z̄ℓ)
        Precision-weighted mismatch between actual and predicted state.
        """
        z = self.layers[ell].z
        z_bar = self.layers[ell].z_bar
        return self.precisions[ell] * (z - z_bar)
    
    def _compute_feedback(self, ell: int) -> np.ndarray:
        """
        Feedback from error below: dℓ = Eℓ · e^{ℓ-1}
        How should this layer adjust to reduce errors in the layer below?
        """
        if ell == 0:
            return np.zeros(self.layer_sizes[0], dtype=np.float64)
        
        error_below = self.layers[ell - 1].error
        return self.E[ell - 1] @ error_below
    
    def settle(self, observation: np.ndarray, 
               steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the predictive coding settling dynamics.
        
        1. Clamp layer 0 to observation
        2. For each settling step:
           a. Compute predictions (top-down): z̄ℓ = Wℓ · φ(z^{ℓ+1})
           b. Compute errors (bottom-up): eℓ = precision * (zℓ - z̄ℓ)
           c. Compute feedback: dℓ = Eℓ · e^{ℓ-1}
           d. Update states: τ · Δzℓ = -γ·zℓ + dℓ·φ'(zℓ) - eℓ
        3. Return when energy converges
        
        Args:
            observation: Sensory input (real-valued vector, e.g., from FHRR magnitude)
            steps: Override settling steps
        
        Returns:
            Settling diagnostics
        """
        n_steps = steps or self.settle_steps
        
        if not self._initialized:
            self._init_layers(observation)
        
        # Clamp sensory layer
        obs = np.asarray(observation, dtype=np.float64)
        if len(obs) != self.layer_sizes[0]:
            # Project to sensory dimension
            if len(obs) > self.layer_sizes[0]:
                obs = obs[:self.layer_sizes[0]]
            else:
                padded = np.zeros(self.layer_sizes[0], dtype=np.float64)
                padded[:len(obs)] = obs
                obs = padded
        
        self.layers[0].z = obs.copy()
        
        energy_trace = []
        error_norms = []
        
        for step in range(n_steps):
            # === TOP-DOWN: Compute predictions ===
            for ell in range(self.n_layers - 1):
                self.layers[ell].z_bar = self._predict(ell)
            # Top layer predicts itself (prior)
            self.layers[-1].z_bar = np.zeros_like(self.layers[-1].z)
            
            # === BOTTOM-UP: Compute prediction errors ===
            for ell in range(self.n_layers):
                self.layers[ell].error = self._compute_error(ell)
            
            # === LATERAL: Compute feedback corrections ===
            # === UPDATE: State dynamics (skip layer 0 — it's clamped) ===
            for ell in range(1, self.n_layers):
                feedback = self._compute_feedback(ell)
                
                # τ · ∂z/∂t = -γ·z + d·φ'(z) - e
                phi_deriv = self._phi_deriv(self.layers[ell].z)
                dz = (-self.gamma * self.layers[ell].z 
                      + feedback * phi_deriv 
                      - self.layers[ell].error)
                
                self.layers[ell].z += (1.0 / self.tau) * dz
                # Clamp states to prevent runaway dynamics
                np.clip(self.layers[ell].z, -10.0, 10.0, out=self.layers[ell].z)
            
            # === ENERGY: Compute VFE ===
            total_energy = 0.0
            step_error_norms = []
            for ell in range(self.n_layers):
                e = self.layers[ell].error
                layer_energy = 0.5 * np.dot(e, e) / max(self.precisions[ell], 1e-8)
                self.layers[ell].energy = layer_energy
                total_energy += layer_energy
                step_error_norms.append(float(np.linalg.norm(e)))
            
            energy_trace.append(total_energy)
            error_norms.append(step_error_norms)
        
        self.energy_history.append(energy_trace[-1])
        self.error_history.append(error_norms[-1])
        
        return {
            "final_energy": energy_trace[-1],
            "energy_trace": energy_trace,
            "error_norms": error_norms[-1],
            "converged": len(energy_trace) > 1 and abs(energy_trace[-1] - energy_trace[-2]) < 1e-6,
            "settle_steps": n_steps,
            "layer_states": [l.z.copy() for l in self.layers],
        }
    
    def learn(self):
        """
        Hebbian synaptic update after settling.
        
        ΔWℓ = lr * (e^{ℓ-1} · (φ(z^ℓ))ᵀ)    — error × pre-synaptic activity
        ΔEℓ = lr * (z^ℓ · (e^{ℓ-1})ᵀ)         — state × error (feedback path)
        
        Includes weight decay (γ_w) and spectral normalization to prevent
        weight explosion. This is fully local: each synapse only needs the 
        pre- and post-synaptic signals available at its endpoints.
        """
        for ell in range(self.n_layers - 1):
            error_below = self.layers[ell].error
            z_above = self._phi(self.layers[ell + 1].z)
            
            # Generative weight update: Hebbian + decay
            dW = np.outer(error_below, z_above)
            self.W[ell] += self.lr * dW - self.lr * self.gamma * self.W[ell]
            
            # Feedback weight update
            dE = np.outer(self.layers[ell + 1].z, error_below)
            self.E[ell] += self.lr * dE - self.lr * self.gamma * self.E[ell]
            
            # Spectral normalization: cap the largest singular value at 1.0
            # This prevents weight explosion while preserving learned structure
            w_norm = np.linalg.norm(self.W[ell], ord=2)
            if w_norm > 1.0:
                self.W[ell] /= w_norm
            e_norm = np.linalg.norm(self.E[ell], ord=2)
            if e_norm > 1.0:
                self.E[ell] /= e_norm
    
    def predict_observation(self) -> np.ndarray:
        """
        Generate a prediction of what layer 0 should look like,
        given current higher-level beliefs.
        
        This is THE missing piece from v1: the system actually predicts
        its sensory input and can be measured on how wrong it is.
        """
        if not self._initialized or self.n_layers < 2:
            return np.zeros(self.layer_sizes[0])
        
        return self._predict(0)
    
    def prediction_error(self, observation: np.ndarray) -> float:
        """
        Compute prediction error: how surprised is the system?
        
        PE = ||observation - predicted_observation||²
        """
        predicted = self.predict_observation()
        obs = np.asarray(observation, dtype=np.float64)
        if len(obs) != len(predicted):
            obs = obs[:len(predicted)] if len(obs) > len(predicted) else np.pad(obs, (0, len(predicted) - len(obs)))
        
        return float(np.sum((obs - predicted) ** 2))
    
    def get_abstract_state(self, level: int = -1) -> np.ndarray:
        """Get the belief state at a given level (-1 = top)."""
        if not self._initialized:
            return np.zeros(self.layer_sizes[level])
        return self.layers[level].z.copy()
    
    @property
    def total_energy(self) -> float:
        """Current variational free energy."""
        return sum(l.energy for l in self.layers) if self.layers else 0.0
    
    @property
    def statistics(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "layer_sizes": self.layer_sizes,
            "total_energy": self.total_energy,
            "energy_history_len": len(self.energy_history),
            "last_error_norms": self.error_history[-1] if self.error_history else [],
        }
