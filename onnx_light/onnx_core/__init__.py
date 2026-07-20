"""Python facade for the ``onnx_core`` C++ library.

This module mirrors the C++ ``onnx_core`` namespace at the Python level,
providing the shared types and graph-manipulation helpers that sit below both
``onnx_op`` and ``onnx_shapes`` in the dependency hierarchy.

The ``TensorType`` enum and ``ToTypeString`` helper are the canonical Python
spelling for the element / sequence type descriptor defined in
``onnx_proto/type_helper.h``.  ``onnx_light.onnx_op.TensorType`` re-exports
the same object for backward compatibility.

The three graph-analysis functions (``collect_external_inputs``,
``collect_node_inputs``, ``collect_remaining_inputs``) mirror the helpers
defined in ``onnx_core/graph/graph_manipulations.h`` and are also re-exported
from ``onnx_light.onnx`` for convenience.
"""

from __future__ import annotations

from ..onnx_py._onnxpyprotoop import (  # type: ignore[attr-defined]
    collect_external_inputs,
    collect_node_inputs,
    collect_remaining_inputs,
)
from ..onnx_py._onnxpyprotoop import onnx_op as _C  # type: ignore[attr-defined]

TensorType = _C.TensorType
ToTypeString = _C.ToTypeString

__all__ = [
    "TensorType",
    "ToTypeString",
    "collect_external_inputs",
    "collect_node_inputs",
    "collect_remaining_inputs",
]
