"""
Legacy V1 substrate.

This package contains the Morton-coded Markov blanket, flat POMDP active
inference loop, and compatibility ``TensegrityAgent`` facade. New integrations
should prefer ``tensegrity.core`` / ``tensegrity.engine`` and the
``UnifiedField`` energy landscape.
"""

from tensegrity.legacy.v1.agent import DEFAULT_MEDIATED_SCM_NAME, TensegrityAgent
from tensegrity.legacy.v1.blanket import MarkovBlanket
from tensegrity.legacy.v1.morton import MortonEncoder

__all__ = (
    "DEFAULT_MEDIATED_SCM_NAME",
    "TensegrityAgent",
    "MarkovBlanket",
    "MortonEncoder",
)
