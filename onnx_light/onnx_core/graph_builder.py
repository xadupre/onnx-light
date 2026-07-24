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

from ..onnx_py._onnxpyoptim import builder as _C  # type: ignore[attr-defined]

GraphBuilder = _C.GraphBuilder

__all__ = [
    "GraphBuilder",
]
