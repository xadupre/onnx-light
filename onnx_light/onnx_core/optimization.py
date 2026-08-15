"""Graph-pattern optimization with standard and Python-defined patterns."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]
from ..onnx_py import _onnxpypatterns as _patterns  # type: ignore[attr-defined]
from .graph_builder import GraphBuilder, _default_schema_lookup

PatternOptimization = _C.PatternOptimization
MatchResult = _C.MatchResult
GraphGraph = _C.GraphGraph
LocalRewriting = _C.LocalRewriting
OptimizationReport = _C.OptimizationReport

CastPattern = _patterns.CastPattern
CastCastPattern = _patterns.CastCastPattern
CastCastBinaryPattern = _patterns.CastCastBinaryPattern
CastOpCastPattern = _patterns.CastOpCastPattern


def registered_pattern_names() -> list[str]:
    """Returns the registered standard ONNX pattern names."""
    return _patterns.registered_pattern_names()


def standard_patterns(names: Iterable[str] | None = None) -> list[PatternOptimization]:
    """Creates the selected standard ONNX patterns."""
    selected = registered_pattern_names() if names is None else list(names)
    return [_patterns.create_pattern(name) for name in selected]


def replay(model: Any, rewrites: Iterable[LocalRewriting], schema_lookup=_default_schema_lookup):
    """Replays captured rewrites and returns the resulting graph."""
    return _C.replay(model, list(rewrites), schema_lookup)


__all__ = [
    "CastCastBinaryPattern",
    "CastCastPattern",
    "CastOpCastPattern",
    "CastPattern",
    "GraphBuilder",
    "GraphGraph",
    "LocalRewriting",
    "MatchResult",
    "OptimizationReport",
    "PatternOptimization",
    "registered_pattern_names",
    "replay",
    "standard_patterns",
]
