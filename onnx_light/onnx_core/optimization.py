"""Graph-pattern optimization with standard and Python-defined patterns."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from ..onnx_lib import GraphProto, ModelProto
from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]
from ..onnx_py import _onnxpypatterns as _patterns  # type: ignore[attr-defined]
from .graph_builder import GraphBuilder, SchemaLookup, _default_schema_lookup

PatternOptimization: TypeAlias = _C.PatternOptimization
MatchResult: TypeAlias = _C.MatchResult
LocalRewriting: TypeAlias = _C.LocalRewriting
OptimizationReport: TypeAlias = _C.OptimizationReport

CastPattern: TypeAlias = _patterns.CastPattern
CastCastPattern: TypeAlias = _patterns.CastCastPattern
CastCastBinaryPattern: TypeAlias = _patterns.CastCastBinaryPattern
CastOpCastPattern: TypeAlias = _patterns.CastOpCastPattern
ClipClipPattern: TypeAlias = _patterns.ClipClipPattern


def standard_pattern_names() -> list[str]:
    """Returns the standard ONNX pattern names."""
    return _patterns.registered_pattern_names()


def standard_patterns(names: Iterable[str] | None = None) -> list[PatternOptimization]:
    """Creates the selected standard ONNX patterns."""
    selected = standard_pattern_names() if names is None else list(names)
    return [_patterns.create_pattern(name) for name in selected]


_GLOBAL_PATTERNS: dict[str, PatternOptimization] = {
    pattern.name: pattern for pattern in standard_patterns()
}


def register_pattern(pattern: PatternOptimization) -> None:
    """Registers or replaces a process-global pattern."""
    name = str(pattern.name)
    if not name:
        raise ValueError("A registered pattern must have a non-empty name.")
    _GLOBAL_PATTERNS[name] = pattern


def unregister_pattern(name: str) -> bool:
    """Removes a global pattern and returns whether it existed."""
    return _GLOBAL_PATTERNS.pop(name, None) is not None


def clear_registered_patterns() -> None:
    """Removes every globally registered pattern, including standard patterns."""
    _GLOBAL_PATTERNS.clear()


def reset_registered_patterns() -> None:
    """Restores the global registry to the standard ONNX patterns."""
    _GLOBAL_PATTERNS.clear()
    _GLOBAL_PATTERNS.update((pattern.name, pattern) for pattern in standard_patterns())


def registered_patterns() -> tuple[PatternOptimization, ...]:
    """Returns global patterns in registration order."""
    return tuple(_GLOBAL_PATTERNS.values())


def registered_pattern_names() -> tuple[str, ...]:
    """Returns global pattern names in registration order."""
    return tuple(_GLOBAL_PATTERNS)


def render_rst_standard_patterns_table() -> str:
    """Renders the standard ONNX patterns as a reST ``list-table``.

    The table is generated from the patterns returned by
    :func:`standard_patterns`, so it stays in sync with the registered
    patterns without any manual maintenance.

    Returns:
        The ``list-table`` directive as a reST string.
    """
    lines = [
        ".. list-table::",
        "    :header-rows: 1",
        "    :widths: 30 10 30 40",
        "",
        "    * - Class / registered name",
        "      - Priority",
        "      - Candidate roots",
        "      - Transformation",
    ]
    for pattern in sorted(standard_patterns(), key=lambda p: str(p.name)):
        class_name = type(pattern).__name__
        roots = ", ".join(f"``{op}``" for op in sorted(pattern.fast_op_type()))
        doc = (type(pattern).__doc__ or "").strip()
        summary = " ".join(doc.split("\n\n", 1)[0].split()) if doc else ""
        lines.extend(
            [
                f"    * - :class:`{class_name}` / ``{pattern.name}``",
                f"      - {pattern.priority}",
                f"      - {roots}",
                f"      - {summary}",
            ]
        )
    return "\n".join(lines) + "\n"


class GraphGraph(_C.GraphGraph):
    """Indexes a builder and runs globally or locally registered patterns.

    Global patterns are applied first. Patterns registered on ``builder`` then
    replace global patterns sharing their name, and ``patterns`` passed here
    have the highest precedence. Set ``use_global_patterns=False`` to start
    from an empty registry.
    """

    def __init__(
        self,
        builder: GraphBuilder,
        patterns: Iterable[str | PatternOptimization] | None = None,
        *,
        use_global_patterns: bool = True,
    ) -> None:
        selected = (
            {pattern.name: pattern for pattern in registered_patterns()}
            if use_global_patterns
            else {}
        )
        if hasattr(builder, "registered_patterns"):
            selected.update((pattern.name, pattern) for pattern in builder.registered_patterns())
        if patterns is not None:
            for pattern in patterns:
                resolved = (
                    _patterns.create_pattern(pattern) if isinstance(pattern, str) else pattern
                )
                selected[resolved.name] = resolved
        super().__init__(builder, list(selected.values()))


def replay(
    model: ModelProto,
    rewrites: Iterable[LocalRewriting],
    schema_lookup: SchemaLookup | None = _default_schema_lookup,
) -> GraphProto:
    """Replays captured rewrites and returns the resulting graph."""
    return _C.replay(model, list(rewrites), schema_lookup)


__all__ = [
    "CastCastBinaryPattern",
    "CastCastPattern",
    "CastOpCastPattern",
    "CastPattern",
    "ClipClipPattern",
    "GraphBuilder",
    "GraphGraph",
    "LocalRewriting",
    "MatchResult",
    "OptimizationReport",
    "PatternOptimization",
    "clear_registered_patterns",
    "register_pattern",
    "registered_pattern_names",
    "registered_patterns",
    "render_rst_standard_patterns_table",
    "replay",
    "reset_registered_patterns",
    "standard_pattern_names",
    "standard_patterns",
    "unregister_pattern",
]
