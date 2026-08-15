"""Incremental ONNX graph builder backed by a C++ library.

This module exposes :class:`GraphBuilder`, an incremental builder for ONNX
graphs, models and functions. A builder starts empty, holds a compute context
and records every value name it hands out so a name can never be reused. Nodes
are added with :meth:`GraphBuilder.make_node`, which resolves the operator
opset, validates the node against the built-in ONNX operator schemas, assigns
output names when the caller leaves them empty and runs incremental shape
inference. :meth:`GraphBuilder.to_onnx` finalises the accumulated graph into a
model (default), a graph or a function, writing the inferred shapes, the
in-place / release-after metadata, the value tags and the peak-memory
estimates.

Typical usage::

    from onnx_light.onnx_core.graph_builder import GraphBuilder
    from onnx_light.onnx_proto import TensorProto

    builder = GraphBuilder("g")
    builder.make_input("x", TensorProto.FLOAT, [2, 3])
    builder.make_input("y", TensorProto.FLOAT, [2, 3])
    (z,) = builder.make_node("Add", ["x", "y"])
    builder.make_output(z)
    model = builder.to_onnx("model")

The module is exposed as ``onnx_light.onnx_core.graph_builder``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from ..onnx_op import GetAllOnnxOpSchemasWithHistory, LightOpSchema
from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]

ConstantFoldingOptions: TypeAlias = _C.ConstantFoldingOptions
PatternOptimization: TypeAlias = _C.PatternOptimization

SchemaLookup: TypeAlias = Callable[[str], "list[LightOpSchema]"]


def _default_schema_lookup(op_type: str) -> list[LightOpSchema]:
    """Returns the built-in ONNX schema history for ``op_type``.

    The schemas live in the ``onnx_op`` extension, which the ``_onnxpycore``
    extension does not link against; this callable bridges the two so the
    builder can resolve opsets and validate nodes without that link.
    """
    return GetAllOnnxOpSchemasWithHistory(op_type, False)


class GraphBuilder(_C.GraphBuilder):
    """Incrementally builds an ONNX graph, model or function.

    See :mod:`onnx_light.onnx_core.graph_builder` for details. By default the
    builder validates nodes and resolves opsets using the built-in ONNX
    operator schemas; pass ``schema_lookup=None`` to disable this, or a custom
    ``op_type -> list[LightOpSchema]`` callable to use different schemas.
    """

    def __init__(
        self, name: str = "graph", schema_lookup: SchemaLookup | None = _default_schema_lookup
    ) -> None:
        super().__init__(name, schema_lookup)
        self._registered_patterns: dict[str, PatternOptimization] = {}

    def register_pattern(self, pattern: PatternOptimization) -> None:
        """Registers or replaces a pattern for this builder."""
        name = str(pattern.name)
        if not name:
            raise ValueError("A registered pattern must have a non-empty name.")
        self._registered_patterns[name] = pattern

    def unregister_pattern(self, name: str) -> bool:
        """Removes a builder-local pattern and returns whether it existed."""
        return self._registered_patterns.pop(name, None) is not None

    def clear_registered_patterns(self) -> None:
        """Removes every builder-local pattern."""
        self._registered_patterns.clear()

    def registered_patterns(self) -> tuple[PatternOptimization, ...]:
        """Returns builder-local patterns in registration order."""
        return tuple(self._registered_patterns.values())

    def registered_pattern_names(self) -> tuple[str, ...]:
        """Returns builder-local pattern names in registration order."""
        return tuple(self._registered_patterns)


__all__ = ["ConstantFoldingOptions", "GraphBuilder", "PatternOptimization", "SchemaLookup"]
