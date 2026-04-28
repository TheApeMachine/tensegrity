"""
Morton (Z-order) Encoder: Modality-Agnostic Sensory Frontend

Morton codes interleave bits from multiple dimensions into a single integer,
preserving spatial locality: points close in N-dimensional space map to
nearby Morton codes. This gives us a UNIVERSAL encoding for any modality.

The key insight: every sensory modality is ultimately a set of measurements
across dimensions. An image is (x, y, channel). Audio is (time, frequency).
Text is (position, embedding_dim). Sensor data is (sensor_id, time, value).

By Morton-encoding any of these, we get a single integer that:
  1. Preserves neighborhood structure (similar inputs → similar codes)
  2. Is modality-agnostic (the system doesn't "know" what modality it is)
  3. Enables efficient range queries via bit-prefix matching
  4. Maps naturally to Bayesian state spaces (discretize → categorize)

Mathematical basis:
  For k dimensions with coordinates (c₁, c₂, ..., cₖ):
  Morton(c₁, c₂, ..., cₖ) = interleave_bits(c₁, c₂, ..., cₖ)

  Where interleave_bits takes bit i of dimension j and places it at
  position i*k + j in the output. This creates a Z-order space-filling curve.
"""

import numpy as np
from itertools import product
from typing import Union, List, Tuple, Optional


# Guard against exponential neighborhood enumeration when radius × dims is large.
MAX_NEIGHBORHOOD_COMBINATIONS = 50_000


class MortonEncoder:
    """
    Encodes arbitrary-dimensional data into Morton codes (Z-order curve indices).
    
    This is the Markov blanket's sensory interface — it transforms raw modality
    data into a unified discrete state space that the inference engine operates on.
    
    The encoding is information-preserving (invertible) and locality-preserving
    (nearby points in input space → nearby Morton codes).
    """
    
    def __init__(self, n_dims: int, bits_per_dim: int = 10, 
                 ranges: Optional[List[Tuple[float, float]]] = None):
        """
        Args:
            n_dims: Number of input dimensions (e.g., 2 for image patches, 
                    3 for volumetric, N for embeddings)
            bits_per_dim: Resolution per dimension. 10 bits = 1024 levels per dim.
                         Total Morton code space = 2^(n_dims * bits_per_dim)
                         Must satisfy n_dims * bits_per_dim <= 63 so codes fit np.int64.
            ranges: Min/max per dimension for quantization. If None, auto-calibrated.
        """
        self.n_dims = n_dims
        self.bits_per_dim = bits_per_dim
        total_bits = n_dims * bits_per_dim
        if total_bits > 63:
            raise ValueError(
                f"total_bits (n_dims * bits_per_dim) must be <= 63 to fit in np.int64; "
                f"got total_bits={total_bits}"
            )
        self.total_bits = total_bits
        self.levels = 2 ** bits_per_dim
        
        # Quantization ranges per dimension
        if ranges is not None:
            self.ranges = np.asarray(ranges, dtype=np.float64)
            if self.ranges.ndim != 2 or self.ranges.shape[1] != 2:
                raise ValueError("ranges must be a sequence of (min, max) tuples per dimension.")
            spans = self.ranges[:, 1] - self.ranges[:, 0]
            flat_spans = np.asarray(spans).flatten()
            bad = np.where(np.abs(flat_spans) < 1e-15)[0]
            if len(bad):
                dims_list = [int(i) for i in bad.tolist()]
                raise ValueError(
                    "Quantization ranges have zero span on dimension index(es) "
                    f"{dims_list}; ensure max > min for each dimension "
                    "(or omit ranges to auto-calibrate from data)."
                )
            if int(self.ranges.shape[0]) != int(n_dims):
                raise ValueError(
                    f"ranges must have length n_dims ({n_dims}), got shape {self.ranges.shape}."
                )
        else:
            self.ranges = None  # Will be set on first encode (auto-calibrate)
        
        # Precompute bit interleaving masks for fast encoding
        # For k dims, bit i of dim j goes to position i*k + j
        self._build_interleave_tables()
    
    def _build_interleave_tables(self):
        """Precompute lookup tables for fast bit interleaving."""
        # For each dimension, build a mask that spreads its bits
        # across the interleaved positions
        self._spread_masks = []
        for dim in range(self.n_dims):
            # For dimension `dim`, bit position `b` in the input
            # maps to bit position `b * n_dims + dim` in the output
            mask_positions = [b * self.n_dims + dim for b in range(self.bits_per_dim)]
            self._spread_masks.append(mask_positions)
    
    def _spread_bits(self, value: int, dim: int) -> int:
        """Spread bits of a single value according to its dimension's interleave pattern."""
        result = 0
        for b in range(self.bits_per_dim):
            if value & (1 << b):
                result |= (1 << self._spread_masks[dim][b])
        return result
    
    def _compact_bits(self, morton: int, dim: int) -> int:
        """Extract and compact bits for a single dimension from a Morton code."""
        result = 0
        for b in range(self.bits_per_dim):
            if morton & (1 << self._spread_masks[dim][b]):
                result |= (1 << b)
        return result
    
    def quantize(self, values: np.ndarray) -> np.ndarray:
        """
        Quantize continuous values to discrete levels.
        
        Maps each dimension's range to [0, 2^bits_per_dim - 1] uniformly.
        This is the analog-to-digital conversion at the sensory boundary.
        """
        if self.ranges is None:
            # Auto-calibrate from data
            if values.ndim == 1:
                values = values.reshape(1, -1)
            self.ranges = np.stack([values.min(axis=0), values.max(axis=0)], axis=1)
            # Prevent zero-range dimensions
            zero_range = self.ranges[:, 0] == self.ranges[:, 1]
            self.ranges[zero_range, 1] = self.ranges[zero_range, 0] + 1.0
        
        if values.ndim == 1:
            values = values.reshape(1, -1)
        
        # Normalize to [0, 1] then scale to [0, levels-1]
        mins = self.ranges[:, 0]
        maxs = self.ranges[:, 1]
        spans = np.maximum(maxs - mins, 1e-15)
        normalized = (values - mins) / spans
        normalized = np.clip(normalized, 0.0, 1.0)
        quantized = (normalized * (self.levels - 1)).astype(np.int64)
        return quantized
    
    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Inverse of quantize — reconstruct continuous approximation."""
        if self.ranges is None:
            raise ValueError(
                "ranges not initialized: call encode (or compute_ranges) "
                "before MortonEncoder.dequantize"
            )
        mins = self.ranges[:, 0]
        maxs = self.ranges[:, 1]
        spans = np.maximum(maxs - mins, 1e-15)
        normalized = quantized.astype(np.float64) / (self.levels - 1)
        return normalized * spans + mins
    
    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        Encode N-dimensional data points into Morton codes.
        
        Args:
            values: Shape (n_points, n_dims) or (n_dims,) for single point.
                   Can be continuous (will be quantized) or already integer.
        
        Returns:
            Morton codes as integer array of shape (n_points,)
        """
        single = values.ndim == 1
        if single:
            values = values.reshape(1, -1)
        
        assert values.shape[1] == self.n_dims, \
            f"Expected {self.n_dims} dims, got {values.shape[1]}"
        
        # Quantize if continuous
        if values.dtype in (np.float32, np.float64):
            quantized = self.quantize(values)
        else:
            quantized = np.asarray(values, dtype=np.int64)
            qmin = int(np.min(quantized))
            qmax = int(np.max(quantized))
            lo = 0
            hi = int(self.levels - 1)
            if qmin < lo or qmax > hi:
                raise ValueError(
                    f"MortonEncoder.encode expects integer coords in [{lo}, {hi}] "
                    f"(levels={self.levels}); got range [{qmin}, {qmax}]"
                )
        
        # Interleave bits for each point
        n_points = quantized.shape[0]
        codes = np.zeros(n_points, dtype=np.int64)
        
        for i in range(n_points):
            morton = 0
            for d in range(self.n_dims):
                morton |= self._spread_bits(int(quantized[i, d]), d)
            codes[i] = morton
        
        return codes[0] if single else codes
    
    def decode(self, codes: Union[int, np.ndarray]) -> np.ndarray:
        """
        Decode Morton codes back to N-dimensional coordinates.
        
        Args:
            codes: Morton code(s) as int or array.
        
        Returns:
            Quantized coordinates of shape (n_points, n_dims) or (n_dims,)
        """
        single = isinstance(codes, (int, np.integer))
        if single:
            codes = np.array([codes], dtype=np.int64)
        
        n_points = len(codes)
        coords = np.zeros((n_points, self.n_dims), dtype=np.int64)
        
        for i in range(n_points):
            for d in range(self.n_dims):
                coords[i, d] = self._compact_bits(int(codes[i]), d)
        
        return coords[0] if single else coords
    
    def encode_continuous(self, values: np.ndarray) -> np.ndarray:
        """Encode continuous data — quantizes automatically."""
        return self.encode(values.astype(np.float64))
    
    def decode_continuous(self, codes: Union[int, np.ndarray]) -> np.ndarray:
        """Decode Morton codes back to continuous approximations."""
        quantized = self.decode(codes)
        if quantized.ndim == 1:
            quantized = quantized.reshape(1, -1)
        return self.dequantize(quantized).squeeze()
    
    def proximity(self, code_a: int, code_b: int) -> float:
        """
        Compute proximity between two Morton codes.
        
        Uses the XOR distance: codes that differ only in low-order bits
        (fine-grained spatial difference) are closer than those differing
        in high-order bits (coarse spatial difference).
        
        Returns value in [0, 1] where 1 = identical.
        """
        xor = code_a ^ code_b
        if xor == 0:
            return 1.0
        # Count the position of the highest differing bit
        highest_diff = int(xor).bit_length()
        return 1.0 - (highest_diff / self.total_bits)
    
    def neighborhood(self, code: int, radius: int = 1) -> List[int]:
        """
        Find Morton codes within a given radius (in quantized coordinates).

        Uses ``decode`` → offset enumeration → ``encode`` within ``[0, levels)``.
        """
        decoded = self.decode(code)
        center = (
            decoded.reshape(-1).astype(np.int64)
            if isinstance(decoded, np.ndarray)
            else np.asarray([decoded], dtype=np.int64)
        )
        n_combo = int((2 * radius + 1) ** self.n_dims)
        if n_combo > MAX_NEIGHBORHOOD_COMBINATIONS:
            raise ValueError(
                f"MortonEncoder.neighborhood would enumerate {n_combo} quantized offset "
                f"combinations (n_dims={self.n_dims}, radius={radius}, levels={self.levels}), "
                f"which exceeds MAX_NEIGHBORHOOD_COMBINATIONS={MAX_NEIGHBORHOOD_COMBINATIONS}; "
                "reduce radius or n_dims."
            )

        offsets = range(-radius, radius + 1)
        neighbors: List[int] = []
        for tup in product(offsets, repeat=self.n_dims):
            offset = np.array(tup, dtype=np.int64)
            point = center + offset
            if np.all(point >= 0) and np.all(point < self.levels):
                neighbors.append(int(self.encode(point.reshape(1, -1))))
        return sorted(set(neighbors))
    
    @staticmethod
    def from_modality(modality: str, **kwargs) -> 'MortonEncoder':
        """
        Factory for common modality configurations.
        
        Args:
            modality: One of 'image', 'audio', 'text', 'timeseries', 'generic'
        """
        configs = {
            'image': {'n_dims': 3, 'bits_per_dim': 8},      # x, y, channel
            'audio': {'n_dims': 2, 'bits_per_dim': 12},      # time, frequency
            'text': {'n_dims': 2, 'bits_per_dim': 10},       # position, feature
            'timeseries': {'n_dims': 2, 'bits_per_dim': 14}, # time, value
            'generic': {'n_dims': kwargs.get('n_dims', 4), 'bits_per_dim': 8},
        }
        config = configs.get(modality, configs['generic'])
        config.update(kwargs)
        return MortonEncoder(**config)



