"""Deprecated V1 Markov blanket shim. Import from ``tensegrity.legacy.v1.blanket``."""

import warnings

warnings.warn(
    "tensegrity.core.blanket is legacy V1; use tensegrity.legacy.v1.blanket "
    "for the old Morton-coded frontend.",
    DeprecationWarning,
    stacklevel=2,
)

from tensegrity.legacy.v1.blanket import MarkovBlanket

__all__ = ("MarkovBlanket",)
