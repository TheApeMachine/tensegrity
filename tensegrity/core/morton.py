"""Deprecated V1 Morton encoder shim. Import from ``tensegrity.legacy.v1.morton``."""

import warnings

warnings.warn(
    "tensegrity.core.morton is legacy V1; import from tensegrity.legacy.v1.morton "
    "for the Morton-coded frontend (same API — re-export only). There is no "
    "alternative module beyond legacy.v1 for this shim.",
    DeprecationWarning,
    stacklevel=2,
)

from tensegrity.legacy.v1.morton import MortonEncoder

__all__ = ("MortonEncoder",)
