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

from typing import Any

from ..onnx_op import GetAllOnnxOpSchemasWithHistory
from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]

ConstantFoldingOptions = _C.ConstantFoldingOptions


def _default_schema_lookup(op_type: str) -> list:
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

    def __init__(self, name: str | Any = "graph", schema_lookup=_default_schema_lookup) -> None:
        super().__init__(name, schema_lookup)


__all__ = ["ConstantFoldingOptions", "GraphBuilder"]
