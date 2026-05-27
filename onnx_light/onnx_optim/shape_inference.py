"""Top-level shape-inference helpers backed by a C++ library.

This module exposes the ``onnx_optim`` shape-inference pipeline at the
``ModelProto`` level. It mirrors the C++ helpers
:cpp:func:`ComputeShapeModel` and :cpp:func:`ApplyInferredShapesToModel`
combined into a single :func:`infer_shapes_model` entry point.

Typical usage::

    from onnx_light.onnx_optim.shape_inference import infer_shapes_model

    infer_shapes_model(model)
    # model.graph.output[i].type and model.graph.value_info now carry
    # the inferred element types and shapes.

The module is exposed as ``onnx_light.onnx_optim.shape_inference``.
"""

from __future__ import annotations

from ..onnx_py._onnxpy import shape_inference as _C  # type: ignore[attr-defined]


def infer_shapes_model(model) -> None:
    """Runs shape inference on a ``ModelProto`` in place.

    Seeds the shape-inference context from the model's ``opset_import``,
    graph initializers and graph inputs, runs per-operator shape
    inference on every node of the main graph, and writes the inferred
    element types and shapes back into ``model.graph.output`` and
    ``model.graph.value_info``.

    :param model: A ``ModelProto`` to mutate in place.
    :raises RuntimeError: If shape inference rejects a node (for
        example because of an unsupported op type) or if ``model``
        has no graph.
    """
    _C.infer_shapes_model(model)


__all__ = ["infer_shapes_model"]
