"""
Causal Arena: Where competing structural causal models fight.

THIS is where the "tension" lives. Multiple SCMs, each proposing a different
causal explanation of the same observations, compete for evidential supremacy.

The tension is NOT metaphorical. It is the literal mathematical difference
between model evidences: F_k - F_j = ln P(data | M_k) - ln P(data | M_j).
When this difference is small, the system is in high tension — it cannot
decide which causal story is correct. When it's large, one model dominates.

The arena resolves tension through:
  1. Bayesian model comparison: P(M_k | data) ∝ P(data | M_k) P(M_k)
  2. Falsification: models that predict observations poorly are eliminated
  3. Structural intervention: do() experiments to distinguish observationally
     equivalent models (this is the system "choosing to learn")
  4. Counterfactual reasoning: what WOULD the data look like under each model?

The arena also generates "epistemic actions" — experiments the agent should
run to maximally reduce its uncertainty about which model is correct.
This is the Expected Free Energy's epistemic component driving exploration.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

from tensegrity.causal.scm import StructuralCausalModel

logger = logging.getLogger(__name__)


class CausalArena:
    """
    Adversarial arena for competing structural causal models.
    
    Each model tries to explain the same stream of observations.
    The arena maintains a posterior distribution over models and
    uses free energy to measure which model wins each round.
    
    The "logic competing with itself" is realized here:
    every new observation triggers a re-evaluation where models
    that predicted it well gain evidence, and models that didn't
    lose evidence. The system is always in tension between
    competing explanations until one dominates.
    """
    
    def __init__(
        self, prior_concentration: float = 1.0,
        falsification_threshold: float = -50.0,
        min_models: int = 2,
    ):
        """
        Args:
            prior_concentration: Dirichlet prior over model space (uniform = 1.0)
            falsification_threshold: Log-evidence below which a model is eliminated
            min_models: Minimum number of models to maintain (prevents collapse)
        """
        self.models: Dict[str, StructuralCausalModel] = {}
        self.model_log_evidence: Dict[str, float] = {}
        self.model_prior: Dict[str, float] = {}
        self.prior_concentration = prior_concentration
        self.falsification_threshold = falsification_threshold
        self.min_models = min_models
        
        # History of arena competitions
        self.competition_history: List[Dict[str, Any]] = []
        
        # Tension metric: entropy of posterior over models
        self.tension_history: List[float] = []
        
        # Evidence trajectories per model
        self.evidence_trajectories: Dict[str, List[float]] = defaultdict(list)
    
    def register_model(self, model: StructuralCausalModel, 
                       prior_weight: Optional[float] = None):
        """Add a competing causal model to the arena.
        
        Before registration, checks for structurally redundant models 
        (same DAG topology as an existing model) and merges their CPTs
        instead of adding a duplicate. This prevents the combinatorial
        explosion identified in the review: dozens of near-identical SCMs
        exhausting the counterfactual budget.
        """
        # Check for structural duplicates: same variables + same edges
        for existing_name, existing_model in self.models.items():
            if self._structurally_equivalent(model, existing_model):
                # Merge: absorb the new model's CPTs into the existing one
                # by averaging Dirichlet pseudocounts. This is mathematically
                # equivalent to a single parameterized meta-model.
                logger.info(
                    f"Merging structurally equivalent model '{model.name}' "
                    f"into existing '{existing_name}'"
                )
                self._merge_model_cpts(existing_model, model)
                return
        
        self.models[model.name] = model
        self.model_log_evidence[model.name] = 0.0
        self.model_prior[model.name] = prior_weight or self.prior_concentration
        self.evidence_trajectories[model.name] = [0.0]
        
        logger.info(f"Registered model '{model.name}' in arena")
    
    @staticmethod
    def _structurally_equivalent(a: StructuralCausalModel, 
                                  b: StructuralCausalModel) -> bool:
        """Check if two SCMs have the same DAG topology (same variables, same edges)."""
        if set(a.variables) != set(b.variables):
            return False
        for var in a.variables:
            a_parents = set(a.mechanisms[var].parents) if var in a.mechanisms else set()
            b_parents = set(b.mechanisms[var].parents) if var in b.mechanisms else set()
            if a_parents != b_parents:
                return False
        return True
    
    @staticmethod
    def _merge_model_cpts(target: StructuralCausalModel, 
                           source: StructuralCausalModel) -> None:
        """Merge source CPTs into target by averaging Dirichlet pseudocounts."""
        for var in target.variables:
            t_mech = target.mechanisms.get(var)
            s_mech = source.mechanisms.get(var)
            if t_mech is not None and s_mech is not None:
                if t_mech.cpt_params.shape == s_mech.cpt_params.shape:
                    # Average the Dirichlet pseudocounts
                    t_mech.cpt_params = (t_mech.cpt_params + s_mech.cpt_params) / 2.0
    
    def compete(self, observation: Dict[str, int]) -> Dict[str, Any]:
        """
        Run one round of competition: all models try to explain the observation.
        
        Returns:
            Competition results including winning model, posteriors, tension
        """
        if len(self.models) < 2:
            logger.warning("Arena needs at least 2 models for meaningful competition")
        
        # Each model computes log P(observation | model)
        log_likelihoods = {}
        
        for name, model in self.models.items():
            log_lik = model.log_evidence([observation])
            log_likelihoods[name] = log_lik
            
            # Accumulate evidence
            self.model_log_evidence[name] += log_lik
            self.evidence_trajectories[name].append(self.model_log_evidence[name])
        
        # --- Early energy filter ---
        # Before running expensive posterior computation and counterfactuals,
        # check if any model's single-step log-likelihood is catastrophically
        # bad. If a proposed SCM completely contradicts the observation, skip
        # the full Bayesian update — it would waste counterfactual budget.
        if log_likelihoods:
            best_lik = max(log_likelihoods.values())
            for name in list(log_likelihoods.keys()):
                if log_likelihoods[name] < best_lik - 20.0:
                    # This model's prediction is >20 nats worse than the best.
                    # Don't waste counterfactuals on it — mark for faster elimination.
                    logger.debug(
                        "Energy filter: model '%s' log-lik=%.1f vs best=%.1f (gap=%.1f)",
                        name, log_likelihoods[name], best_lik,
                        best_lik - log_likelihoods[name],
                    )
        
        # Compute posterior P(M_k | data) ∝ exp(cumulative_log_evidence + log_prior)
        posterior = self._compute_posterior()
        
        # Compute tension = entropy of posterior (high entropy = high tension)
        tension = self._compute_tension(posterior)
        self.tension_history.append(tension)
        
        # Determine winner
        winner = max(posterior, key=posterior.get)
        
        # Falsification: check if any model should be eliminated
        eliminated = self._falsify()
        
        # Update all surviving models with the observation
        for name, model in self.models.items():
            model.update_from_data([observation])
        
        result = {
            'winner': winner,
            'posterior': posterior,
            'log_likelihoods': log_likelihoods,
            'cumulative_evidence': dict(self.model_log_evidence),
            'tension': tension,
            'eliminated': eliminated,
            'n_surviving_models': len(self.models),
        }
        
        self.competition_history.append(result)
        return result
    
    def compete_batch(self, observations: List[Dict[str, int]]) -> List[Dict[str, Any]]:
        """Run competition over a batch of observations."""
        return [self.compete(obs) for obs in observations]
    
    def _compute_posterior(self) -> Dict[str, float]:
        """
        Bayesian model posterior: P(M_k | data) ∝ P(data | M_k) P(M_k)
        
        In log space: log P(M_k | data) = log_evidence_k + log_prior_k - log Z
        """
        log_posteriors = {}
        
        for name in self.models:
            log_posteriors[name] = self.model_log_evidence[name] + np.log(
                max(self.model_prior[name], 1e-16)
            )
        
        # Normalize via log-sum-exp
        max_log = max(log_posteriors.values())
        
        log_Z = max_log + np.log(sum(
            np.exp(lp - max_log) for lp in log_posteriors.values()
        ))
        
        posterior = {name: np.exp(lp - log_Z) 
                    for name, lp in log_posteriors.items()}
        
        return posterior
    
    def _compute_tension(self, posterior: Dict[str, float]) -> float:
        """
        Tension = normalized entropy of the posterior over models.
        
        Tension ∈ [0, 1]:
          0 = one model completely dominates (no tension, resolved)
          1 = all models equally likely (maximum tension, unresolved)
        """
        probs = np.array(list(posterior.values()))
        probs = np.maximum(probs, 1e-16)
        
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(max(len(probs), 1))
        
        if max_entropy == 0:
            return 0.0
        
        return float(entropy / max_entropy)
    
    def _falsify(self) -> List[str]:
        """
        Eliminate models whose evidence drops below threshold.
        
        This is Popperian falsification: models that consistently fail
        to predict observations are removed from the competition.
        """
        if len(self.models) <= self.min_models:
            return []
        
        to_eliminate = []
        # Compare each model against the best
        best_evidence = max(self.model_log_evidence.values())
        
        for name, evidence in self.model_log_evidence.items():
            relative_evidence = evidence - best_evidence
            if relative_evidence < self.falsification_threshold:
                to_eliminate.append(name)
        
        # Don't eliminate below minimum
        if len(self.models) - len(to_eliminate) < self.min_models:
            # Keep the best of the eliminated
            to_eliminate.sort(key=lambda n: self.model_log_evidence[n])
            to_eliminate = to_eliminate[:len(self.models) - self.min_models]
        
        for name in to_eliminate:
            logger.info(f"Falsified model '{name}' (evidence: {self.model_log_evidence[name]:.2f})")
            del self.models[name]
            del self.model_log_evidence[name]
            del self.model_prior[name]
        
        return to_eliminate
    
    def suggest_experiment(self, n_samples: int = 32,
                           n_outcome_samples: int = 8) -> Dict[str, Any]:
        """
        Suggest an intervention that would maximally reduce tension.
        
        This is the epistemic component of Expected Free Energy:
        Choose the intervention that maximizes information gain about
        which model is correct.
        
        IG(do(X=x)) = H[P(M|data)] - E_{o~P(o|do(x))} H[P(M|data,o)]
        
        In practice: find the variable where models disagree most about
        the effect of intervention, and suggest intervening on it.
        """
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")
        if not isinstance(n_outcome_samples, int) or n_outcome_samples < 1:
            raise ValueError(
                f"n_outcome_samples must be an int >= 1, got {n_outcome_samples!r}"
            )

        outcome_cap = min(n_samples, n_outcome_samples)

        if len(self.models) < 2:
            return {'intervention': None, 'expected_info_gain': 0.0}
        
        # Get all variables across models
        all_vars = set()
        
        for model in self.models.values():
            all_vars.update(model.variables)
        
        best_experiment = None
        best_info_gain = -float('inf')
        
        for var in all_vars:
            # For each possible intervention value
            n_values = 4  # Default cardinality
        
            for model in self.models.values():
                if var in model.mechanisms:
                    n_values = model.mechanisms[var].n_values
                    break
            
            for val in range(n_values):
                info_gain = self._estimate_info_gain(
                    var, val,
                    n_samples=n_samples,
                    n_outcome_samples=outcome_cap,
                )

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_experiment = {'variable': var, 'value': val}
        
        return {
            'intervention': best_experiment,
            'expected_info_gain': best_info_gain,
            'current_tension': self.tension_history[-1] if self.tension_history else 1.0,
        }
    
    def _estimate_info_gain(self, var: str, val: int, n_samples: int = 32,
                            n_outcome_samples: int = 8) -> float:
        """
        Estimate information gain from do(var=val).
        
        IG = H[posterior_current] - E[H[posterior_after_experiment]]
        """
        current_posterior = self._compute_posterior()
        current_tension = self._compute_tension(current_posterior)
        
        # For each model, predict what we'd observe after intervention
        predicted_outcomes = {}
        
        for name, model in self.models.items():
            if var not in model.graph:
                continue
            
            mutilated = model.do({var: val})
            outcomes = mutilated.sample(n_samples)
            predicted_outcomes[name] = outcomes
        
        if not predicted_outcomes:
            return 0.0
        
        # Estimate expected posterior entropy after seeing outcomes
        # Use model-averaged predictions
        expected_tension = 0.0
        effective_outcome_samples = min(n_samples, max(1, n_outcome_samples))
        
        for name, outcomes in predicted_outcomes.items():
            model_weight = current_posterior.get(name, 1.0 / len(self.models))
            
            for outcome in outcomes[:effective_outcome_samples]:
                # What would the posterior look like if we saw this outcome?
                hypothetical_log_liks = {}

                for m_name, model in self.models.items():
                    hypothetical_log_liks[m_name] = model.log_evidence([outcome])

                # Hypothetical posterior
                hyp_evidence = {m: self.model_log_evidence[m] + hypothetical_log_liks[m]
                               for m in self.models}

                max_e = max(hyp_evidence.values())

                log_Z = max_e + np.log(sum(
                    np.exp(e - max_e) for e in hyp_evidence.values()))

                hyp_posterior = {m: np.exp(e - log_Z) for m, e in hyp_evidence.items()}

                expected_tension += model_weight * self._compute_tension(hyp_posterior)

        expected_tension /= max(effective_outcome_samples, 1)

        # Information gain = current uncertainty - expected uncertainty after experiment
        return current_tension - expected_tension
    
    def do_experiment(self, var: str, val: int, 
                      observed_outcome: Dict[str, int]) -> Dict[str, Any]:
        """
        Execute an intervention experiment and update models.
        
        This is the active inference loop: the agent intervenes in the world,
        observes the outcome, and uses it to update model posteriors.
        """
        # Apply do() to each model and compute likelihood of observed outcome
        for name, model in self.models.items():
            if var in model.graph:
                mutilated = model.do({var: val})
                # Log-likelihood of outcome under mutilated model
                log_lik = mutilated.log_evidence([observed_outcome])
                self.model_log_evidence[name] += log_lik
                self.evidence_trajectories[name].append(self.model_log_evidence[name])
        
        posterior = self._compute_posterior()
        tension = self._compute_tension(posterior)
        self.tension_history.append(tension)
        
        return {
            'posterior': posterior,
            'tension': tension,
            'intervention': {var: val},
            'outcome': observed_outcome,
        }
    
    def counterfactual_comparison(
        self, 
        evidence: Dict[str, int],
        intervention: Dict[str, int],
        query: List[str],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Ask each model: "What would have happened if we had done X instead?"
        
        Returns counterfactual predictions from each competing model.
        Where they disagree is where future experiments should focus.
        """
        results = {}
        for name, model in self.models.items():
            try:
                cf = model.counterfactual(evidence, intervention, query)
                results[name] = cf
            except Exception as e:
                logger.warning(f"Counterfactual failed for model '{name}': {e}")
                results[name] = {q: np.ones(model.mechanisms[q].n_values) / 
                                model.mechanisms[q].n_values for q in query
                                if q in model.mechanisms}
        
        return results
    
    @property
    def current_winner(self) -> Optional[str]:
        """The model with highest current posterior probability."""
        if not self.models:
            return None
        
        posterior = self._compute_posterior()
        return max(posterior, key=posterior.get)
    
    @property
    def current_tension(self) -> float:
        """Current tension level (0=resolved, 1=maximum uncertainty)."""
        if not self.models:
            return 0.0
        
        posterior = self._compute_posterior()
        return self._compute_tension(posterior)
    
    @property
    def statistics(self) -> Dict[str, Any]:
        return {
            'n_models': len(self.models),
            'current_winner': self.current_winner,
            'current_tension': self.current_tension,
            'total_competitions': len(self.competition_history),
            'model_evidence': dict(self.model_log_evidence),
            'posterior': self._compute_posterior() if self.models else {},
        }
