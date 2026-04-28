"""
Causal Energy: Pearl's SCMs as energy terms in the unified landscape.

Each SCM contributes a prediction error to the total energy:
    E_causal(M_k) = Σ_v ||z_v - f_v(z_pa(v))||²

Where:
    z_v = observed value of variable v
    f_v(z_pa(v)) = structural equation's prediction from parents
    pa(v) = parents of v in the causal DAG

Multiple SCMs compete. The model with lowest causal energy provides
the best explanation. This complements the log-likelihood CausalArena in ``tensegrity.causal.arena``
when an energy-based readout of SCM fit is required.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from tensegrity.causal.scm import StructuralCausalModel


class CausalEnergyTerm:
    """
    Computes causal prediction error energy for an SCM.
    
    Given observations of some variables, computes how well
    the SCM's structural equations predict them.
    """
    
    def __init__(self, scm: StructuralCausalModel, precision: float = 1.0):
        self.scm = scm
        self.precision = precision
    
    def energy(self, observations: Dict[str, int]) -> float:
        """
        Compute causal prediction error energy.
        
        E = Σ_v (1/2σ²) ||obs_v - predicted_v||²
        
        Where predicted_v = E[V | parents of V observed]
        """
        total_energy = 0.0
        order = self.scm.topological_order()
        
        for var in order:
            if var not in observations:
                continue
            
            mech = self.scm.mechanisms[var]
            parent_vals = {p: observations.get(p, 0) for p in mech.parents}
            
            # Expected value under the CPT
            cpt = mech.cpt
            config_idx = mech.parent_config_index(parent_vals)
            probs = cpt[:, config_idx]
            
            # Prediction = expected value index
            expected = np.sum(probs * np.arange(len(probs)))
            observed = float(observations[var])
            
            # Squared prediction error
            error = (observed - expected) ** 2
            total_energy += 0.5 * self.precision * error
        
        return total_energy
    
    def prediction(self, observations: Dict[str, int], 
                   target: str) -> np.ndarray:
        """Predict distribution over target given observed parents."""
        mech = self.scm.mechanisms.get(target)
        if mech is None:
            return np.array([1.0])
        
        parent_vals = {p: observations.get(p, 0) for p in mech.parents}
        config_idx = mech.parent_config_index(parent_vals)
        return mech.cpt[:, config_idx]


class EnergyCausalArena:
    """
    SCMs compete via prediction-error energy. Lowest energy wins.
    """
    
    def __init__(self, precision: float = 1.0, beta: float = 1.0):
        """
        Args:
            precision: Causal prediction error precision
            beta: Inverse temperature for model selection softmax
        """
        self.models: Dict[str, CausalEnergyTerm] = {}
        self.beta = beta
        self.precision = precision
        self._history: List[Dict[str, float]] = []
    
    def register(self, scm: StructuralCausalModel):
        """Add a competing causal model."""
        self.models[scm.name] = CausalEnergyTerm(scm, self.precision)
    
    def compete(self, observations: Dict[str, int]) -> Dict[str, Any]:
        """
        All models compute their causal energy on the observation.
        Returns energies, posteriors, and tension.
        """
        energies = {}
        for name, term in self.models.items():
            energies[name] = term.energy(observations)
        
        if not energies:
            return {"winner": None, "tension": 1.0, "energies": {}}
        
        # Softmax over negative energies (lower energy = higher weight)
        vals = np.array(list(energies.values()))
        neg_e = -self.beta * vals
        neg_e -= neg_e.max()
        weights = np.exp(neg_e)
        weights /= weights.sum()
        
        posteriors = dict(zip(energies.keys(), weights.tolist()))
        
        # Tension = normalized entropy
        probs = weights[weights > 0]
        if len(probs) > 1:
            entropy = -np.sum(probs * np.log(probs))
            tension = float(entropy / np.log(len(probs)))
        else:
            tension = 0.0
        
        winner = min(energies, key=energies.get)
        best_energy = energies[winner]
        
        result = {
            "winner": winner,
            "tension": tension,
            "posteriors": posteriors,
            "energies": energies,
            "best_energy": best_energy,
        }
        self._history.append(energies)
        
        return result
    
    def best_energy(self, observations: Dict[str, int]) -> float:
        """Get the energy of the best-fitting model."""
        result = self.compete(observations)
        return result.get("best_energy", 0.0)
    
    def update_models(self, observations: Dict[str, int]):
        """Update all models' parameters from observation (Dirichlet counting)."""
        for name, term in self.models.items():
            term.scm.update_from_data([observations])
    
    @property
    def tension(self) -> float:
        """Current tension (from last competition)."""
        if not self._history:
            return 1.0
        last = self._history[-1]
        vals = np.array(list(last.values()))
        neg_e = -self.beta * vals
        neg_e -= neg_e.max()
        w = np.exp(neg_e)
        w /= w.sum()
        w = w[w > 0]
        if len(w) > 1:
            return float(-np.sum(w * np.log(w)) / np.log(len(w)))
        return 0.0
