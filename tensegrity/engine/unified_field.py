"""
Unified Energy Landscape — operating in SBERT embedding space.

V3: The cognitive stack operates DIRECTLY in the sentence-transformer
embedding space. No random projection, no FHRR→obs conversion for the
cognitive path.

    Layer 0 of NGC = SBERT embedding (384-dim for MiniLM-L6-v2)
    Hopfield memory stores SBERT embeddings
    Falsification compares SBERT embeddings through learned W matrices

Why: The NGC needs to learn the structure of how questions map to answers.
Random projections destroy the semantic structure that SBERT provides.
With 100+ benchmark items, the NGC sees enough data to learn real
generative models of question→answer mappings in SBERT space.

The FHRR encoder is preserved for compositional binding operations
(role-filler pairs, sequence encoding) but is NOT in the NGC's
observation path. FHRR lives alongside SBERT, not in front of it.

Energy decomposition remains:
    E_total = E_perception (NGC) + E_memory (Hopfield)
Both now operate on semantically meaningful vectors.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Deque
from dataclasses import dataclass
from collections import deque

_logger = logging.getLogger(__name__)

from .fhrr import FHRREncoder, bind, bundle, unbind
from .ngc import PredictiveCodingCircuit


@dataclass
class EnergyDecomposition:
    """Breakdown of the total energy into components."""
    perception: float    # NGC prediction error energy
    memory: float        # Hopfield retrieval energy
    causal: float        # Causal SCM prediction error
    total: float         # Sum
    prediction_error_norm: float  # ||obs − predicted||² after settling
    surprise: float      # -log P(observation | beliefs)


class HopfieldMemoryBank:
    """
    Modern Hopfield network operating in SBERT embedding space.

    Stores sentence embeddings as patterns. Retrieval is energy minimization:
        E(ξ) = -lse(β, Xᵀξ) + ½||ξ||²
        ξ_new = X · softmax(β · Xᵀ · ξ)

    Now stores 384-dim SBERT embeddings (not 8-dim NGC top states).
    This gives the memory enough information to distinguish semantically
    different inputs and retrieve genuinely relevant past experiences.
    """

    def __init__(self, dim: int, beta: float = 0.05, capacity: int = 10000):
        self.dim = dim
        self.beta = beta
        self.capacity = capacity

        self.patterns: deque = deque(maxlen=capacity)
        self._matrix: Optional[np.ndarray] = None
        self._dirty = True

    def clear(self) -> None:
        self.patterns.clear()
        self._matrix = None
        self._dirty = True

    def store(self, pattern: np.ndarray, normalize: bool = True):
        p = np.real(pattern).astype(np.float64) if np.iscomplexobj(pattern) else pattern.astype(np.float64)
        if normalize:
            norm = np.linalg.norm(p)
            if norm > 0:
                p = p / norm
        self.patterns.append(p)
        self._dirty = True

    def retrieve(self, query: np.ndarray, steps: int = 3) -> Tuple[np.ndarray, float]:
        if not self.patterns:
            return np.zeros(self.dim), 0.0

        self._ensure_matrix()

        q = np.real(query).astype(np.float64) if np.iscomplexobj(query) else query.astype(np.float64)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        xi = q.copy()
        for _ in range(steps):
            sims = self._matrix.T @ xi
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

        sims = self._matrix.T @ xi
        if self.beta <= 1e-12:
            energy = float(0.5 * np.dot(xi, xi) - np.mean(sims))
        else:
            log_sum_exp = np.log(np.sum(np.exp(self.beta * sims - self.beta * sims.max()))) + self.beta * sims.max()
            energy = float(-log_sum_exp / self.beta + 0.5 * np.dot(xi, xi))

        return xi, energy

    def _ensure_matrix(self):
        if self._dirty and self.patterns:
            self._matrix = np.column_stack(list(self.patterns))
            self._dirty = False

    @property
    def n_patterns(self):
        return len(self.patterns)


class UnifiedField:
    """
    The unified cognitive field — operating in SBERT embedding space.

    V3 architecture:
      1. SBERT encoder provides the observation vector (no projection needed)
      2. NGC circuit operates on SBERT embeddings: layer 0 = sbert_dim
      3. Hopfield memory stores SBERT embeddings directly
      4. FHRR encoder preserved for compositional binding (parallel path)

    The NGC learns a generative model of how text maps to embeddings.
    After 100+ items, the W matrices encode real structure: "prompts in
    this domain tend to predict answers with these embedding patterns."
    This makes falsification genuine: "does settling on this answer's
    embedding produce a good prediction of the prompt's embedding?"
    """

    # Default SBERT dim for all-MiniLM-L6-v2
    DEFAULT_SBERT_DIM = 384

    def __init__(self,
                 obs_dim: int = 256,
                 hidden_dims: List[int] = None,
                 fhrr_dim: int = 2048,
                 hopfield_beta: float = 0.05,
                 ngc_settle_steps: int = 20,
                 ngc_learning_rate: float = 0.005,
                 ngc_precisions: Optional[List[float]] = None,
                 energy_history_maxlen: int = 500,
                 sbert_dim: Optional[int] = None):
        """
        Args:
            obs_dim: Legacy parameter. If sbert_dim is set, NGC uses sbert_dim
                     for layer 0 instead. Kept for backward compatibility with
                     code that constructs UnifiedField with obs_dim.
            hidden_dims: NGC hidden layer dimensions. Full hierarchy =
                        [sbert_dim or obs_dim] + hidden_dims
            fhrr_dim: FHRR dimensionality (for compositional binding path)
            hopfield_beta: Inverse temperature for Hopfield retrieval
            ngc_settle_steps: Settling iterations for NGC
            ngc_learning_rate: Hebbian learning rate
            sbert_dim: If set, NGC layer 0 uses this dimension (SBERT space).
                      Detected automatically when SBERT is available.
        """
        if hidden_dims is None:
            hidden_dims = [128, 32]

        self.fhrr_dim = fhrr_dim

        # FHRR encoder (for compositional binding — parallel path)
        self.encoder = FHRREncoder(dim=fhrr_dim)

        # Detect SBERT dimension from the encoder
        self._sbert_dim = sbert_dim
        if self._sbert_dim is None:
            # Try to detect from the semantic codebook
            features = self.encoder.features
            if hasattr(features, '_sbert_dim') and features._sbert_dim is not None:
                self._sbert_dim = features._sbert_dim
            elif hasattr(features, '_ensure_sbert'):
                features._ensure_sbert()
                if hasattr(features, '_sbert_dim') and features._sbert_dim is not None:
                    self._sbert_dim = features._sbert_dim

        # NGC operates in SBERT space when available, else falls back to obs_dim
        if self._sbert_dim is not None and self._sbert_dim > 0:
            self.obs_dim = self._sbert_dim
            _logger.info(
                "UnifiedField: NGC operating in SBERT space (dim=%d)", self._sbert_dim
            )
        else:
            self.obs_dim = obs_dim
            _logger.info(
                "UnifiedField: SBERT unavailable, NGC using obs_dim=%d", obs_dim
            )

        # NGC circuit: layer 0 = SBERT dim (or obs_dim fallback)
        layer_sizes = [self.obs_dim] + hidden_dims
        self.ngc = PredictiveCodingCircuit(
            layer_sizes=layer_sizes,
            precisions=ngc_precisions,
            settle_steps=ngc_settle_steps,
            learning_rate=ngc_learning_rate,
        )

        # Hopfield memory: stores SBERT embeddings directly
        # NOT the tiny NGC top-layer states — full SBERT embeddings so
        # retrieval can distinguish semantically different inputs.
        self.memory = HopfieldMemoryBank(
            dim=self.obs_dim, beta=hopfield_beta
        )

        # Legacy: keep _fhrr_to_obs working for callers that still use it
        self._proj_mode = "identity_or_sbert"
        self._proj_block_size = max(1, fhrr_dim // self.obs_dim)

        # Energy tracking
        self._step_count = 0
        self.energy_history: Deque[EnergyDecomposition] = deque(
            maxlen=max(1, int(energy_history_maxlen))
        )

    def get_sbert_embedding(self, text: str) -> Optional[np.ndarray]:
        """Encode text directly to SBERT embedding, bypassing FHRR.

        Returns None if SBERT is not available.
        """
        features = self.encoder.features
        getter = getattr(features, "get_sbert_model", None)
        sbert = getter() if callable(getter) else None
        if sbert is None:
            return None
        try:
            emb = sbert.encode([text], show_progress_bar=False)[0]
            return np.asarray(emb, dtype=np.float64)
        except Exception as e:
            _logger.debug("SBERT encoding failed: %s", e)
            return None

    def text_to_obs(self, text: str) -> np.ndarray:
        """Convert text to observation vector for NGC.

        Prefers SBERT embedding (semantically rich, right dimensionality).
        Falls back to FHRR→block-average if SBERT is unavailable.
        """
        emb = self.get_sbert_embedding(text)
        if emb is not None:
            # Ensure correct dimension
            if len(emb) == self.obs_dim:
                return emb
            elif len(emb) > self.obs_dim:
                return emb[:self.obs_dim]
            else:
                return np.pad(emb, (0, self.obs_dim - len(emb)))

        # Fallback: FHRR path
        import re
        tokens = re.findall(
            r"[a-zA-Z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?", text.lower()
        )[-64:]
        fhrr_vec = self.encoder.encode_sequence(tokens) if tokens else \
            np.ones(self.fhrr_dim, dtype=np.complex64)
        return self._fhrr_to_obs(fhrr_vec)

    def _fhrr_to_obs(self, fhrr_vec: np.ndarray) -> np.ndarray:
        """Legacy: project FHRR to obs space via block averaging."""
        real_part = np.real(fhrr_vec).astype(np.float64)
        bs = self._proj_block_size
        obs = np.zeros(self.obs_dim, dtype=np.float64)
        for i in range(self.obs_dim):
            start = i * bs
            end = min(start + bs, len(real_part))
            if start < len(real_part):
                obs[i] = np.mean(real_part[start:end])
        return obs

    def observe(self, raw_input: Any, input_type: str = "numeric") -> Dict[str, Any]:
        """
        Full cognitive cycle in SBERT space.

        For text inputs: encodes via SBERT directly (no FHRR intermediary).
        For legacy inputs (numeric, bindings, tokens): uses FHRR→obs fallback.
        """
        self._step_count += 1

        # === 1. ENCODE ===
        if input_type == "text":
            obs_vec = self.text_to_obs(str(raw_input))
            fhrr_vec = self.encoder.encode_sequence(
                str(raw_input).lower().split()
            )
        elif input_type == "tokens":
            # Try SBERT on joined text, fall back to FHRR
            text = " ".join(raw_input) if isinstance(raw_input, list) else str(raw_input)
            obs_vec = self.text_to_obs(text)
            fhrr_vec = self.encoder.encode_sequence(raw_input)
        elif input_type == "bindings":
            fhrr_vec = self.encoder.encode_observation(raw_input)
            # Try to get SBERT embedding from the binding values
            text = " ".join(f"{k} {v}" for k, v in raw_input.items())
            obs_vec = self.text_to_obs(text)
        elif input_type == "numeric":
            fhrr_vec = self.encoder.encode_numeric_vector(np.asarray(raw_input))
            obs_vec = self._fhrr_to_obs(fhrr_vec)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        # === 2. PREDICT ===
        prediction_error_pre_settle = self.ngc.prediction_error(obs_vec)

        # === 3. SETTLE ===
        settle_result = self.ngc.settle(obs_vec)
        perception_energy = settle_result["final_energy"]

        # === 4. MEMORY: query and re-settle ===
        abstract_state = self.ngc.get_abstract_state(level=-1)

        # Store the SBERT embedding in Hopfield (not the tiny abstract state)
        # but query with the abstract state projected back to obs_dim
        # Actually: store obs_vec (SBERT embedding) and query with obs_vec
        # The memory operates in the same space as NGC layer 0.
        retrieved, memory_energy = self.memory.retrieve(obs_vec)

        obs_norm = np.linalg.norm(obs_vec)
        ret_norm = np.linalg.norm(retrieved)
        if obs_norm > 1e-8 and ret_norm > 1e-8:
            memory_similarity = float(
                np.dot(obs_vec / obs_norm, retrieved / ret_norm)
            )
        else:
            memory_similarity = 0.0

        # Memory-guided re-settle
        if self.memory.n_patterns > 2 and ret_norm > 1e-8:
            blend = float(1.0 / (1.0 + np.exp(-3.0 * memory_similarity)))
            blend = min(blend, 0.5)
            top_layer = self.ngc.layers[-1]
            # Project retrieved memory (obs_dim) to top-layer dim
            # Use the NGC's own prediction to do this: retrieve → settle layer 0
            # → the top layer state IS the "memory's view" of the abstract state
            # Simpler: just blend at layer 0 and re-settle
            self.ngc.layers[0].z = (1.0 - blend) * obs_vec + blend * retrieved
            re_settle = self.ngc.settle(
                self.ngc.layers[0].z,
                steps=max(3, self.ngc.settle_steps // 3)
            )
            perception_energy = re_settle["final_energy"]
            abstract_state = self.ngc.get_abstract_state(level=-1)

        prediction_error_post_settle = self.ngc.prediction_error(obs_vec)

        # === 5. LEARN ===
        if self.memory.n_patterns <= 2:
            learning_modulation = 1.0
        else:
            learning_modulation = float(
                1.0 / (1.0 + np.exp(-3.0 * memory_similarity))
            )

        self.ngc.learn(modulation=learning_modulation)
        self.memory.store(obs_vec)  # Store SBERT embedding, not abstract state

        # === 6. ENERGY ===
        decomp = EnergyDecomposition(
            perception=perception_energy,
            memory=memory_energy,
            causal=0.0,
            total=perception_energy + memory_energy,
            prediction_error_norm=float(prediction_error_post_settle),
            surprise=float(np.log1p(max(prediction_error_post_settle, 0.0))),
        )
        self.energy_history.append(decomp)

        return {
            "step": self._step_count,
            "fhrr_vector": fhrr_vec,
            "observation": obs_vec,
            "abstract_state": abstract_state,
            "retrieved_memory": retrieved,
            "memory_similarity": memory_similarity,
            "learning_modulation": learning_modulation,
            "energy": decomp,
            "settle": settle_result,
            "prediction_error": prediction_error_pre_settle,
            "prediction_error_pre_settle": prediction_error_pre_settle,
            "prediction_error_post_settle": prediction_error_post_settle,
        }

    def predict(self) -> np.ndarray:
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
            "sbert_dim": self._sbert_dim,
        }
