"""
Hierarchical Predictive Coding Circuit (NGC).

This is what was missing. Each layer maintains a belief about its hidden
state and PREDICTS the layer below. The difference between prediction and
actual input is the prediction error. Errors propagate upward. Predictions
propagate downward. The system settles when errors are minimized.

This is Friston's Free Energy Principle implemented as it was meant to be:
not as a flat POMDP solver, but as a hierarchical generative model where
each level explains away the residuals of the level below.

The energy functional (from Ororbia & Kelly, arXiv:2310.15177), integrating
the precision-weighted residuals with correct scaling:

    ℱ(Θ) ≈ Σ_ℓ (1 / 2 ρℓ²) · ||eℓ||²   with   eℓ = ρℓ · (zℓ − z̄ℓ)

Where:
    zℓ      = actual state at layer ℓ  
    z̄ℓ      = W^ℓ · φ(z^{ℓ+1})  = prediction from layer above
    ρℓ      = precision = 1 / σℓ²  (inverse variance; higher = trust layer more)
    eℓ      = ρℓ · (zℓ − z̄ℓ)        = precision-weighted prediction error

State dynamics (settling toward VFE minimum):
    τ · ∂zℓ/∂t = -γ·zℓ + dℓ ⊙ fD(zℓ) - eℓ
    where dℓ = Eℓ · e^{ℓ-1}  (feedback error from below)

Synaptic update (Hebbian, local, no backprop):
    ΔWℓ = eℓ⁻¹ · (zℓ)ᵀ     (prediction errors × pre-synaptic activity)

This is ALL local computation. No gradient chain. Each layer only needs
its own state, the prediction from above, and the error from below.
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


def _spectral_norm_power_iteration(A: np.ndarray, n_iters: int = 4) -> float:
    """Approximate largest singular value via power iteration."""
    m, n = A.shape
    rng = np.random.RandomState(424242)
    v = rng.randn(n).astype(np.float64)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(max(1, n_iters)):
        u = A @ v
        u /= np.linalg.norm(u) + 1e-12
        v = A.T @ u
        v /= np.linalg.norm(v) + 1e-12
    return float(np.linalg.norm(A @ v))
@dataclass
class LayerState:
    """State of a single predictive coding layer."""
    z: np.ndarray           # State neurons (current belief about this level)
    z_bar: np.ndarray       # Top-down prediction (from layer above)
    error: np.ndarray       # Precision-weighted mismatch: ρ · (z − z̄)
    precision: float        # ρ = 1 / σ² — inverse variance for this layer (higher = sharper trust)
    energy: float = 0.0     # Local contribution to VFE
    

class PredictiveCodingCircuit:
    """
    A hierarchical NGC circuit with L layers.
    
    Layer 0 = sensory (clamped to observation)
    Layer L = highest abstraction (prior beliefs)
    
    Information flow:
      Top-down:  z̄ℓ = Wℓ · φ(z^{ℓ+1})     (predictions flow down)
      Bottom-up: eℓ = ρℓ · (zℓ − z̄ℓ)   (ρℓ = 1/σ²)
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
                 settle_steps_warm: int = 5,
                 obs_change_threshold: float = 1e-2,
                 learning_rate: float = 0.01,
                 activation: str = "tanh",
                 adaptive_precision: bool = True,
                 precision_momentum: float = 0.9,
                 precision_min: float = 0.1,
                 precision_max: float = 100.0,
                 max_history_length: int = 2000):
        """
        Args:
            layer_sizes: [dim_sensory, dim_hidden1, ..., dim_top]
                        e.g., [2048, 512, 128, 32] for a 4-layer hierarchy
            precisions: Per-layer precision ρ = 1/σ² (inverse variance). Higher = more trust.
                       If None, defaults to 1.0 everywhere. Length must equal ``n_layers`` when given.
            tau: Membrane time constant (settling speed)
            gamma: State decay rate (leaky integration)
            settle_steps: How many steps to run before declaring convergence
            settle_steps_warm: Steps when the observation is nearly unchanged (warm-started z)
            obs_change_threshold: L2 change above this triggers full settle_steps
            learning_rate: Hebbian learning rate for synaptic updates
            activation: Nonlinearity: "tanh", "relu", "sigmoid", or "linear"
            adaptive_precision: If True, update precisions from prediction-error variance in learn()
            precision_momentum: EMA factor for precision updates (higher = slower change)
            precision_min / precision_max: Clamp learned precisions
            max_history_length: Max entries retained in energy / error history (ring buffer)
        """
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.tau = tau
        self.gamma = gamma
        self.settle_steps = settle_steps
        self.settle_steps_warm = max(1, int(settle_steps_warm))
        self.obs_change_threshold = obs_change_threshold
        self.lr = learning_rate
        self.adaptive_precision = adaptive_precision
        self.precision_momentum = precision_momentum
        self.precision_min = precision_min
        self.precision_max = precision_max
        self.max_history_length = max(1, int(max_history_length))
        
        # Activation function
        self._phi, self._phi_deriv = self._get_activation(activation)
        
        # Precisions (per layer): ρ = 1/σ²
        if precisions is None:
            self.precisions = [1.0] * self.n_layers
        else:
            if len(precisions) != self.n_layers:
                raise ValueError(
                    f"precisions must have length n_layers={self.n_layers}, got {len(precisions)}"
                )
            self.precisions = list(precisions)
        
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
        
        # Energy tracking (bounded)
        self.energy_history: deque = deque(maxlen=self.max_history_length)
        self.error_history: deque = deque(maxlen=self.max_history_length)
        
        # Warm-start: last observation for change detection
        self._last_obs: Optional[np.ndarray] = None
    
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
        obs = np.asarray(observation, dtype=np.float64).ravel()
        need = self.layer_sizes[0]
        if obs.size != need:
            logger.warning(
                "PredictiveCodingCircuit.settle: len(obs)=%s != expected sensory width %s",
                obs.size,
                need,
            )
            raise ValueError(
                f"observation length {obs.size} must match layer_sizes[0]={need}"
            )
        obs_changed = True
        if self._last_obs is not None and self._last_obs.shape == obs.shape:
            if float(np.linalg.norm(obs - self._last_obs)) <= self.obs_change_threshold:
                obs_changed = False
        self._last_obs = obs.copy()
        
        if steps is not None:
            n_steps = steps
        elif not self._initialized:
            n_steps = self.settle_steps
        elif obs_changed:
            n_steps = self.settle_steps
        else:
            n_steps = self.settle_steps_warm
        
        if not self._initialized:
            self._init_layers(obs)
        
        # Clamp sensory layer
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
                prec = max(self.precisions[ell], 1e-8)
                layer_energy = 0.5 * np.dot(e, e) / (prec ** 2)
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
    
    def learn(self, modulation: float = 1.0):
        """
        Hebbian synaptic update after settling.
        
        ΔWℓ = modulation * lr * (e^{ℓ-1} · (φ(z^ℓ))ᵀ)
        
        The modulation parameter gates learning: when the current observation
        is inconsistent with established beliefs (high prediction error +
        low memory similarity), modulation should be low, preventing the
        system from learning from contradictory evidence.
        
        This is precision-weighted Hebbian learning: the effective learning
        rate is lr * modulation, where modulation encodes the system's
        confidence that this observation is trustworthy.
        """
        effective_lr = self.lr * modulation
        
        if self.adaptive_precision and self.layers:
            for ell in range(self.n_layers):
                residual = self.layers[ell].z - self.layers[ell].z_bar
                sq_error = float(np.mean(residual ** 2))
                target_precision = 1.0 / max(sq_error, 1e-6)
                mom = self.precision_momentum
                self.precisions[ell] = mom * self.precisions[ell] + (1.0 - mom) * target_precision
                self.precisions[ell] = float(
                    np.clip(self.precisions[ell], self.precision_min, self.precision_max)
                )
                self.layers[ell].precision = self.precisions[ell]
        
        for ell in range(self.n_layers - 1):
            error_below = self.layers[ell].error
            z_above = self._phi(self.layers[ell + 1].z)
            
            # Generative weight update: Hebbian + decay
            dW = np.outer(error_below, z_above)
            self.W[ell] += effective_lr * dW - effective_lr * self.gamma * self.W[ell]
            
            # Feedback weight update
            dE = np.outer(self.layers[ell + 1].z, error_below)
            self.E[ell] += effective_lr * dE - effective_lr * self.gamma * self.E[ell]
            
            # Spectral normalization (power iteration — cheaper than full SVD)
            w_norm = _spectral_norm_power_iteration(self.W[ell])
            if w_norm > 1.0:
                self.W[ell] /= w_norm
            e_norm = _spectral_norm_power_iteration(self.E[ell])
            if e_norm > 1.0:
                self.E[ell] /= e_norm
    
    def clear_history(self) -> None:
        """Drop recorded energy / error traces."""
        self.energy_history.clear()
        self.error_history.clear()
    
    def reinitialize(self, weight_seed: int = 12345) -> None:
        """Reset layer states and resample W/E."""
        rng = np.random.RandomState(weight_seed)
        self.layers = []
        self._initialized = False
        self._last_obs = None
        self.clear_history()
        for ell in range(self.n_layers - 1):
            fan_in = self.layer_sizes[ell + 1]
            fan_out = self.layer_sizes[ell]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W[ell] = rng.randn(fan_out, fan_in).astype(np.float64) * scale
            self.E[ell] = self.W[ell].T.copy()
    
    def save_state(self) -> Dict[str, Any]:
        """Snapshot weights and layer activations."""
        layer_snap = [
            (
                l.z.copy(),
                l.z_bar.copy(),
                l.error.copy(),
                float(l.precision),
                float(l.energy),
            )
            for l in self.layers
        ]
        return {
            "layers": layer_snap,
            "W": [w.copy() for w in self.W],
            "E": [e.copy() for e in self.E],
            "precisions": list(self.precisions),
            "_initialized": self._initialized,
            "_last_obs": None if self._last_obs is None else self._last_obs.copy(),
        }
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore from ``save_state()``."""
        self.precisions = list(state["precisions"])
        self.W = [w.copy() for w in state["W"]]
        self.E = [e.copy() for e in state["E"]]
        self._initialized = bool(state["_initialized"])
        lo = state["_last_obs"]
        self._last_obs = None if lo is None else np.asarray(lo, dtype=np.float64).copy()
        self.layers = []
        for z, zb, er, prec, ener in state["layers"]:
            self.layers.append(
                LayerState(
                    z=z, z_bar=zb, error=er, precision=prec, energy=ener,
                )
            )
    
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
