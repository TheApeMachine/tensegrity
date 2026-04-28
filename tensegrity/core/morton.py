"""Deprecated V1 Morton encoder shim. Import from ``tensegrity.legacy.v1.morton``."""

import warnings

warnings.warn(
    "tensegrity.core.morton is legacy V1; use tensegrity.legacy.v1.morton "
    "for the old Morton-coded frontend.",
    DeprecationWarning,
    stacklevel=2,
)

from tensegrity.legacy.v1.morton import MortonEncoder

__all__ = ("MortonEncoder",)
