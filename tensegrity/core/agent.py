"""Deprecated V1 agent shim. Import from ``tensegrity.legacy.v1.agent``."""

import warnings

warnings.warn(
    "tensegrity.core.agent is legacy V1; use tensegrity.legacy.v1.agent for "
    "the Morton/POMDP agent or tensegrity.core.UnifiedField for the V2 engine.",
    DeprecationWarning,
    stacklevel=2,
)

from tensegrity.legacy.v1.agent import DEFAULT_MEDIATED_SCM_NAME, TensegrityAgent

__all__ = ("DEFAULT_MEDIATED_SCM_NAME", "TensegrityAgent")
