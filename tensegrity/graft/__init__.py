

"""LLM graft interfaces for token-level and semantic vocabulary grounding."""

from importlib import import_module
from typing import Any

from tensegrity.graft.vocabulary import SemanticProjectionLayer, VocabularyGrounding
from tensegrity.graft.logit_bias import (
    GraftState,
    StaticLogitBiasBuilder,
    TensegrityLogitsProcessor,
)

__all__ = (
    "SemanticProjectionLayer",
    "VocabularyGrounding",
    "GraftState",
    "StaticLogitBiasBuilder",
    "TensegrityLogitsProcessor",
    "HybridPipeline",
)


def __dir__():
    merged = set(globals().keys()) | set(__all__)
    return sorted(merged)


def __getattr__(name: str) -> Any:
    if name == "HybridPipeline":
        value = getattr(import_module("tensegrity.graft.pipeline"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tensegrity.graft' has no attribute {name!r}")
