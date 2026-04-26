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

Position encoding uses the Residue Number System (RNS):
  For co-prime moduli m₁, m₂, ..., mₖ:
    encode(x) = g₁^(x mod m₁) ⊙ g₂^(x mod m₂) ⊙ ... ⊙ gₖ^(x mod mₖ)
  
  where gᵢ are fixed random phasor codebooks.

Properties (proven in Frady et al. 2021, arXiv:2406.18808):
  - Exponential coding range: M = ∏ mᵢ
  - Near-orthogonal: E[cos(encode(x), encode(y))] ≈ 0 for x ≠ y
  - Similarity-preserving: similar inputs → correlated vectors
  - Path integration: encode(x) ⊙ encode(y) = encode(x + y)
  - Invertible: unbind to recover components
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Union


class FHRRCodebook:
    """
    A codebook of random FHRR phasor vectors for one semantic domain.
    
    Each entry is a D-dimensional complex vector on the unit circle.
    Used for: role vectors, filler vectors, feature vectors.
    """
    
    def __init__(self, n_symbols: int, dim: int, seed: int = 0):
        """
        Args:
            n_symbols: Number of distinct symbols in this codebook
            dim: Hypervector dimensionality (typically 1000-10000)
            seed: Random seed for reproducibility
        """
        self.n_symbols = n_symbols
        self.dim = dim
        rng = np.random.RandomState(seed)
        
        # Generate random phases on [0, 2π) for each symbol
        phases = rng.uniform(0, 2 * np.pi, size=(n_symbols, dim))
        self.vectors = np.exp(1j * phases).astype(np.complex64)
        
        # Label map
        self._labels: Dict[str, int] = {}
    
    def register(self, label: str) -> int:
        """Register a named symbol, return its index."""
        if label not in self._labels:
            idx = len(self._labels)
            if idx >= self.n_symbols:
                raise ValueError(f"Codebook full ({self.n_symbols} symbols)")
            self._labels[label] = idx
        return self._labels[label]
    
    def get(self, label_or_idx: Union[str, int]) -> np.ndarray:
        """Get the hypervector for a symbol."""
        if isinstance(label_or_idx, str):
            idx = self._labels.get(label_or_idx)
            if idx is None:
                idx = self.register(label_or_idx)
            return self.vectors[idx]
        return self.vectors[label_or_idx]
    
    def inverse(self, label_or_idx: Union[str, int]) -> np.ndarray:
        """Get the inverse (conjugate) for unbinding."""
        return np.conj(self.get(label_or_idx))
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two hypervectors."""
        return float(np.real(np.dot(a, np.conj(b))) / self.dim)
    
    def query(self, probe: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the closest codebook entries to a probe vector."""
        sims = np.real(self.vectors @ np.conj(probe)) / self.dim
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        idx_to_label = {v: k for k, v in self._labels.items()}
        for idx in top_idx:
            label = idx_to_label.get(int(idx), f"#{idx}")
            results.append((label, float(sims[idx])))
        return results


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two FHRR vectors: element-wise complex multiplication."""
    return a * b


def bundle(*vectors: np.ndarray) -> np.ndarray:
    """Bundle multiple FHRR vectors: element-wise addition + normalize."""
    result = sum(vectors)
    # Project back to unit circle (normalize magnitude, keep phase)
    magnitude = np.abs(result)
    magnitude = np.maximum(magnitude, 1e-8)
    return result / magnitude


def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Unbind: retrieve the vector that was bound with key."""
    return bound * np.conj(key)


def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """Permute: circular shift (encodes sequence/order)."""
    return np.roll(v, shift)


class FHRREncoder:
    """
    FHRR-RNS encoder: modality-agnostic compositional observation encoding.
    
    Replaces Morton codes with compositionally structured hypervectors.
    
    An observation like "red ball on table" becomes:
      z = bind(role_color, filler_red) + bind(role_object, filler_ball) 
        + bind(role_relation, filler_on) + bind(role_location, filler_table)
    
    This can be decomposed back:
      unbind(z, role_color) ≈ filler_red
      unbind(z, role_object) ≈ filler_ball
    
    Positions are encoded via RNS:
      encode_position(x) = g₁^(x%m₁) ⊙ g₂^(x%m₂) ⊙ ...
    """
    
    def __init__(self, dim: int = 2048,
                 n_position_moduli: int = 3,
                 position_range: int = 100000,
                 n_features: int = 256,
                 n_roles: int = 32):
        """
        Args:
            dim: Hypervector dimensionality
            n_position_moduli: Number of co-prime moduli for RNS
            position_range: Maximum position value to encode
            n_features: Size of feature codebook
            n_roles: Size of role codebook
        """
        self.dim = dim
        
        # RNS moduli: choose co-primes whose product > position_range
        self.moduli = self._select_coprimes(n_position_moduli, position_range)
        
        # Position basis vectors (one per modulus)
        self._pos_bases = []
        for i, m in enumerate(self.moduli):
            rng = np.random.RandomState(1000 + i)
            phases = rng.uniform(0, 2 * np.pi, size=dim)
            base = np.exp(1j * phases).astype(np.complex64)
            self._pos_bases.append(base)
        
        # Codebooks
        self.roles = FHRRCodebook(n_roles, dim, seed=2000)
        self.features = FHRRCodebook(n_features, dim, seed=3000)
        
        # Pre-register common roles
        for role in ["position", "value", "type", "attribute", "relation",
                     "subject", "object", "time", "channel"]:
            self.roles.register(role)
    
    def _select_coprimes(self, n: int, min_product: int) -> List[int]:
        """Select n co-prime numbers whose product exceeds min_product."""
        # Use small primes
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
        """
        Encode an integer position via RNS-FHRR.
        
        encode(x) = ⊙ᵢ gᵢ^(x mod mᵢ)
        
        Properties:
          encode(x) ⊙ encode(y) ≈ encode(x + y)  (path integration)
          sim(encode(x), encode(y)) → 0 for |x-y| > threshold
        """
        result = np.ones(self.dim, dtype=np.complex64)
        for base, m in zip(self._pos_bases, self.moduli):
            residue = x % m
            result = result * (base ** residue)
        return result
    
    def encode_value(self, value: float, precision: int = 100) -> np.ndarray:
        """Encode a continuous value by quantizing and position-encoding."""
        quantized = int(round(value * precision))
        return self.encode_position(quantized)
    
    def encode_token(self, token: str) -> np.ndarray:
        """Encode a text token as a feature hypervector."""
        return self.features.get(token)
    
    def encode_binding(self, role: str, filler: str) -> np.ndarray:
        """Encode a role-filler pair: bind(role_vector, filler_vector)."""
        return bind(self.roles.get(role), self.features.get(filler))
    
    def encode_observation(self, bindings: Dict[str, str]) -> np.ndarray:
        """
        Encode a structured observation as a bundled set of role-filler bindings.
        
        Args:
            bindings: {role: filler} dictionary
                e.g., {"object": "ball", "color": "red", "location": "table"}
        
        Returns:
            Bundled hypervector representing the full observation
        """
        bound_pairs = []
        for role, filler in bindings.items():
            bound_pairs.append(self.encode_binding(role, filler))
        
        if not bound_pairs:
            return np.ones(self.dim, dtype=np.complex64)
        
        return bundle(*bound_pairs)
    
    def encode_sequence(self, tokens: List[str]) -> np.ndarray:
        """
        Encode an ordered sequence of tokens.
        Uses permutation to encode position within the sequence.
        
        seq_vec = Σᵢ permute(encode(tokenᵢ), i)
        """
        elements = []
        for i, token in enumerate(tokens):
            vec = self.features.get(token)
            elements.append(permute(vec, shift=i))
        
        if not elements:
            return np.ones(self.dim, dtype=np.complex64)
        
        return bundle(*elements)
    
    def encode_numeric_vector(self, values: np.ndarray) -> np.ndarray:
        """
        Encode a numeric vector (any modality) as bound position-value pairs.
        
        For a vector [v₀, v₁, v₂, ...]:
          z = Σᵢ bind(encode_position(i), encode_value(vᵢ))
        """
        bound = []
        for i, v in enumerate(values):
            pos_vec = self.encode_position(i)
            val_vec = self.encode_value(float(v))
            bound.append(bind(pos_vec, val_vec))
        
        if not bound:
            return np.ones(self.dim, dtype=np.complex64)
        
        return bundle(*bound)
    
    def decode_role(self, observation: np.ndarray, role: str) -> List[Tuple[str, float]]:
        """
        Unbind a role from an observation and query the feature codebook.
        
        "What fills the role 'color' in this observation?"
        """
        unbound = unbind(observation, self.roles.get(role))
        return self.features.query(unbound, top_k=5)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two hypervectors."""
        return float(np.real(np.dot(a, np.conj(b))) / self.dim)
