"""
FHRR-RNS Encoder: Compositional observation encoding via Vector Symbolic Architecture.

Replaces Morton codes. Instead of an opaque integer, observations become
high-dimensional complex phasor vectors that support:

  BINDING:    a ⊙ b  = element-wise multiply   → "a WITH b"
  BUNDLING:   a + b  = element-wise add         → "a OR b"  
  UNBINDING:  z ⊙ a⁻¹ ≈ b  (if z = a ⊙ b)     → "what was bound with a?"
  SIMILARITY: cos(a, b) → 0 if unrelated, → 1 if same

The Fractional Holographic Reduced Representation (FHRR) uses complex
phasors: each element is e^(iθ) on the unit circle. Binding is element-wise
multiplication of phasors (angle addition). Unbinding is conjugation.

When semantic=True, the features codebook is grounded in sentence-transformer
embeddings projected into phasor space: similar words → similar phasors.
This is what gives the cognitive layer real semantic knowledge.
"""

import hashlib
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


def _stable_label_rng_seed(label: str) -> int:
    """Deterministic 31-bit seed from UTF-8 label (reproducible across processes)."""
    h = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False) % (2**31)


class FHRRCodebook:
    """Random FHRR phasor codebook. Each entry is a D-dim complex vector on the unit circle."""
    
    def __init__(self, n_symbols: int, dim: int, seed: int = 0):
        self.n_symbols = n_symbols
        self.dim = dim
        rng = np.random.RandomState(seed)
        phases = rng.uniform(0, 2 * np.pi, size=(n_symbols, dim))
        self.vectors = np.exp(1j * phases).astype(np.complex64)
        self._labels: Dict[str, int] = {}
    
    def register(self, label: str) -> int:
        if label not in self._labels:
            idx = len(self._labels)

            if idx >= self.n_symbols:
                rng = np.random.RandomState(_stable_label_rng_seed(label))
                new_phases = rng.uniform(0, 2 * np.pi, size=(256, self.dim))
                new_vecs = np.exp(1j * new_phases).astype(np.complex64)
                self.vectors = np.concatenate([self.vectors, new_vecs], axis=0)
                self.n_symbols += 256

            self._labels[label] = idx

        return self._labels[label]
    
    def get(self, label_or_idx: Union[str, int]) -> np.ndarray:
        if isinstance(label_or_idx, str):
            idx = self._labels.get(label_or_idx)

            if idx is None:
                idx = self.register(label_or_idx)

            return self.vectors[idx]

        return self.vectors[label_or_idx]
    
    def inverse(self, label_or_idx: Union[str, int]) -> np.ndarray:
        return np.conj(self.get(label_or_idx))
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.real(np.dot(a, np.conj(b))) / self.dim)
    
    def query(self, probe: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        idx_to_label = {v: k for k, v in self._labels.items()}

        if idx_to_label:
            active_idx = np.array(sorted(idx_to_label), dtype=np.int64)
            sims_active = np.real(self.vectors[active_idx] @ np.conj(probe)) / self.dim
            order = np.argsort(sims_active)[::-1][:top_k]

            return [
                (idx_to_label[int(active_idx[i])], float(sims_active[i]))
                for i in order
            ]

        sims = np.real(self.vectors @ np.conj(probe)) / self.dim
        top_idx = np.argsort(sims)[::-1][:top_k]

        return [(f"#{int(i)}", float(sims[i])) for i in top_idx]


class SemanticFHRRCodebook(FHRRCodebook):
    """
    FHRR codebook grounded in sentence-transformer embeddings.
    
    θ_token = π * tanh(α * P @ sbert_embed(token))
    phasor_token = exp(i * θ_token)
    
    Semantically similar tokens → similar phasor vectors.
    No training required — one projection matrix, one sbert forward pass.
    """
    
    def __init__(self, dim: int, sbert_model: str = "all-MiniLM-L6-v2", seed: int = 42):
        self.dim = dim
        self.n_symbols = 0
        self._capacity = 0
        self._size = 0
        self._buf = np.zeros((0, dim), dtype=np.complex64)
        self._labels: Dict[str, int] = {}
        self._sbert_model_name = sbert_model
        self._sbert = None
        self._sbert_dim = None
        self._proj_seed = seed
        self._proj: Optional[np.ndarray] = None

    @property
    def vectors(self) -> np.ndarray:
        """Rows of active codebook entries (shape ``(n_symbols, dim)``)."""
        return self._buf[: self._size] if self._size else self._buf[:0]

    def _ensure_vector_capacity(self, need: int = 1) -> None:
        if self._size + need <= self._capacity:
            return

        new_cap = max(8, self._capacity * 2 if self._capacity else 8, self._size + need)
        nb = np.zeros((new_cap, self.dim), dtype=np.complex64)
        
        if self._size > 0:
            nb[: self._size] = self._buf[: self._size]

        self._buf = nb
        self._capacity = new_cap
    
    def _ensure_sbert(self):
        if self._sbert is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self._sbert_model_name)
            test = self._sbert.encode(["test"], show_progress_bar=False)
            self._sbert_dim = test.shape[1]
            rng = np.random.RandomState(self._proj_seed)
            self._proj = rng.randn(self.dim, self._sbert_dim).astype(np.float32)
            self._proj /= np.sqrt(self._sbert_dim)
            logger.info(f"SemanticFHRR: loaded {self._sbert_model_name} "
                       f"(dim={self._sbert_dim}) → FHRR(dim={self.dim})")
        except Exception as exc:
            logger.warning("SemanticFHRR: falling back to deterministic random vectors (%s)", exc)
            self._sbert = "FALLBACK"
            self._proj = None
    
    def _embed_to_phasor(self, embedding: np.ndarray) -> np.ndarray:
        projected = self._proj @ embedding.astype(np.float32)
        proj_std = np.std(projected)
        if proj_std > 1e-8:
            projected = projected * (1.5 / proj_std)
        return np.exp(1j * np.pi * np.tanh(projected)).astype(np.complex64)
    
    def register(self, label: str) -> int:
        if label in self._labels:
            return self._labels[label]

        self._ensure_sbert()

        if self._sbert == "FALLBACK" or self._proj is None:
            rng = np.random.RandomState(_stable_label_rng_seed(label))
            new_vec = np.exp(1j * rng.uniform(0, 2 * np.pi, size=self.dim)).astype(np.complex64)
        else:
            embedding = self._sbert.encode([label], show_progress_bar=False)[0]
            new_vec = self._embed_to_phasor(embedding)

        self._ensure_vector_capacity(1)
        self._buf[self._size] = new_vec.reshape(self.dim)
        self._labels[label] = self._size
        self._size += 1
        self.n_symbols = self._size

        return self._labels[label]
    
    def register_batch(self, labels: List[str]) -> List[int]:
        new_labels = [l for l in labels if l not in self._labels]

        if not new_labels:
            return [self._labels[l] for l in labels]

        self._ensure_sbert()

        if self._sbert == "FALLBACK" or self._proj is None:
            return [self.register(l) for l in labels]

        embeddings = self._sbert.encode(new_labels, show_progress_bar=False)
        
        new_matrix = np.stack(
            [self._embed_to_phasor(embeddings[i]) for i in range(len(new_labels))],
            axis=0,
        )
        
        self._ensure_vector_capacity(len(new_labels))
        start = self._size
        
        for i, label in enumerate(new_labels):
            self._labels[label] = start + i
            self._buf[start + i] = new_matrix[i]
        
        self._size += len(new_labels)
        self.n_symbols = self._size
        
        return [self._labels[l] for l in labels]


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind: element-wise complex multiplication."""
    return a * b

def bundle(*vectors: np.ndarray) -> np.ndarray:
    """Bundle: element-wise addition + normalize to unit circle."""
    if not vectors:
        return np.array([], dtype=np.complex64)
    stacked = np.stack([np.asarray(v, dtype=np.complex128) for v in vectors], axis=0)
    result = np.sum(stacked, axis=0).astype(np.complex128)
    magnitude = np.maximum(np.abs(result), 1e-8)
    return (result / magnitude).astype(np.complex64)

def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Unbind: multiply by conjugate."""
    return bound * np.conj(key)

def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """Permute: circular shift (encodes sequence order)."""
    return np.roll(v, shift)


class FHRREncoder:
    """
    FHRR-RNS encoder: modality-agnostic compositional observation encoding.
    
    When semantic=True (default), the features codebook uses sentence-transformer
    embeddings projected into phasor space. Semantically similar tokens get
    similar phasor vectors, giving the cognitive layer real semantic knowledge.
    """
    
    def __init__(self, dim: int = 2048, n_position_moduli: int = 3,
                 position_range: int = 100000, n_features: int = 4096,
                 n_roles: int = 32, semantic: bool = True,
                 sbert_model: str = "all-MiniLM-L6-v2"):
        self.dim = dim
        self.semantic = semantic
        self.moduli = self._select_coprimes(n_position_moduli, position_range)
        
        self._pos_bases = []

        for i, m in enumerate(self.moduli):
            rng = np.random.RandomState(1000 + i)
            self._pos_bases.append(np.exp(1j * rng.uniform(0, 2*np.pi, size=dim)).astype(np.complex64))
        
        self.roles = FHRRCodebook(n_roles, dim, seed=2000)
        self.features = SemanticFHRRCodebook(dim=dim, sbert_model=sbert_model) if semantic \
            else FHRRCodebook(n_features, dim, seed=3000)
        
        self._position_cache: Dict[int, np.ndarray] = {}
        self._position_cache_max = 4096
        
        for role in ["position", "value", "type", "attribute", "relation",
                     "subject", "object", "time", "channel"]:
            self.roles.register(role)
    
    def _select_coprimes(self, n: int, min_product: int) -> List[int]:
        primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        selected = primes[:n]
        product = 1
        
        for p in selected:
            product *= p
        
        while product < min_product and len(selected) < len(primes):
            selected.append(primes[len(selected)])
            product *= selected[-1]
        
        return selected
    
    def encode_position(self, x: int) -> np.ndarray:
        x = int(x)
        cached = self._position_cache.get(x)
        
        if cached is not None:
            return cached.copy()
        
        result = np.ones(self.dim, dtype=np.complex64)
        
        for base, m in zip(self._pos_bases, self.moduli):
            result = result * (base ** (x % m))
        
        if len(self._position_cache) < self._position_cache_max:
            self._position_cache[x] = result.copy()
        
        return result
    
    def encode_value(self, value: float, precision: int = 100) -> np.ndarray:
        return self.encode_position(int(round(value * precision)))
    
    def encode_token(self, token: str) -> np.ndarray:
        return self.features.get(token)
    
    def encode_binding(self, role: str, filler: str) -> np.ndarray:
        return bind(self.roles.get(role), self.features.get(filler))
    
    def encode_observation(self, bindings: Dict[str, str]) -> np.ndarray:
        bound_pairs = [self.encode_binding(r, f) for r, f in bindings.items()]
        return bundle(*bound_pairs) if bound_pairs else np.ones(self.dim, dtype=np.complex64)
    
    def encode_sequence(self, tokens: List[str]) -> np.ndarray:
        elements = [permute(self.features.get(t), shift=i) for i, t in enumerate(tokens)]
        return bundle(*elements) if elements else np.ones(self.dim, dtype=np.complex64)
    
    def encode_numeric_vector(self, values: np.ndarray) -> np.ndarray:
        bound = [bind(self.encode_position(i), self.encode_value(float(v))) for i, v in enumerate(values)]
        return bundle(*bound) if bound else np.ones(self.dim, dtype=np.complex64)
    
    def decode_role(self, observation: np.ndarray, role: str) -> List[Tuple[str, float]]:
        return self.features.query(unbind(observation, self.roles.get(role)), top_k=5)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.real(np.dot(a, np.conj(b))) / self.dim)
