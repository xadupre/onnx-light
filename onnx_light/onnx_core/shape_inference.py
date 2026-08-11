"""Top-level shape-inference helpers backed by a C++ library.

This module exposes the ``onnx_shapes`` shape-inference pipeline at the
``ModelProto`` level. It mirrors the C++ helpers
:cpp:func:`ComputeShapeModel` and :cpp:func:`ApplyInferredShapesToModel`
combined into a single :func:`infer_shapes_model` entry point.

Typical usage::

    from onnx_light.onnx_core.shape_inference import infer_shapes_model

    infer_shapes_model(model)
    # model.graph.output[i].type and model.graph.value_info now carry
    # the inferred element types and shapes.

The module is exposed as ``onnx_light.onnx_core.shape_inference``.
"""

from __future__ import annotations

from typing import TypeAlias

from ..onnx_py._onnxpycore import shape_inference as _C  # type: ignore[attr-defined]

apply_inferred_shapes_to_model = _C.apply_inferred_shapes_to_model
check_inputs_available = _C.check_inputs_available
compute_inplace_reuse = _C.compute_inplace_reuse
compute_value_and_node_tags = _C.compute_value_and_node_tags
INPLACE_REUSE_METADATA_KEY = _C.INPLACE_REUSE_METADATA_KEY
RELEASE_AFTER_METADATA_KEY = _C.RELEASE_AFTER_METADATA_KEY
RELEASE_AFTER_SHAPE_TAG_METADATA_KEY = _C.RELEASE_AFTER_SHAPE_TAG_METADATA_KEY
NODE_TAG_METADATA_KEY = _C.NODE_TAG_METADATA_KEY
CONSTANT_METADATA_KEY = _C.CONSTANT_METADATA_KEY
VALUE_TAG_METADATA_KEY = _C.VALUE_TAG_METADATA_KEY
VALUE_TAGS_METADATA_KEY = _C.VALUE_TAGS_METADATA_KEY
compute_shape_model = _C.compute_shape_model
compute_shape_node = _C.compute_shape_node
compute_peak_memory = _C.compute_peak_memory
peak_memory_dispatch_table_keys = _C.peak_memory_dispatch_table_keys
Device = _C.Device
infer_value_and_node_tags = compute_value_and_node_tags
kUnknownOpsetVersion = _C.kUnknownOpsetVersion
kOnnxDomain = _C.kOnnxDomain
InPlaceReuse = _C.InPlaceReuse
InPlaceReuseKind = _C.InPlaceReuseKind
NODE_MEMORY_TOTAL_BYTES_KEY = _C.NODE_MEMORY_TOTAL_BYTES_KEY
NODE_MEMORY_ALREADY_ALLOCATED_BYTES_KEY = _C.NODE_MEMORY_ALREADY_ALLOCATED_BYTES_KEY
NODE_MEMORY_OUTPUT_ALLOCATION_BYTES_KEY = _C.NODE_MEMORY_OUTPUT_ALLOCATION_BYTES_KEY
NODE_MEMORY_INPUTS_KEY = _C.NODE_MEMORY_INPUTS_KEY
NODE_MEMORY_INITIALIZERS_KEY = _C.NODE_MEMORY_INITIALIZERS_KEY
NODE_MEMORY_INTERMEDIATES_KEY = _C.NODE_MEMORY_INTERMEDIATES_KEY
NODE_MEMORY_OUTPUTS_KEY = _C.NODE_MEMORY_OUTPUTS_KEY
NODE_PEAK_MEMORY_KEY = _C.NODE_PEAK_MEMORY_KEY
NodeMemoryProfile: TypeAlias = dict[str, int | str | dict[str, int | str]]
ComputeContext = _C.ComputeContext
ComputeEvent = _C.ComputeEvent
ComputeEventAction = _C.ComputeEventAction
SymDim = _C.SymDim
SymShape = _C.SymShape
SymTensor = _C.SymTensor
ShapeEvent = _C.ShapeEvent
ShapeEventAction = _C.ShapeEventAction
ShapesContext = _C.ShapesContext
write_inplace_reuse_to_metadata = _C.write_inplace_reuse_to_metadata
write_peak_memory_to_metadata = _C.write_peak_memory_to_metadata
write_value_and_node_tags_to_metadata = _C.write_value_and_node_tags_to_metadata
write_constant_info_to_metadata = _C.write_constant_info_to_metadata


def infer_shapes_model(model, prefill_with_value_info_output: bool = False) -> None:
    """Runs shape inference on a ``ModelProto`` in place.

    Seeds the shape-inference context from the model's ``opset_import``,
    graph initializers and graph inputs, runs per-operator shape
    inference on every node of the main graph, and writes the inferred
    element types and shapes back into ``model.graph.output`` and
    ``model.graph.value_info``.

    :param model: A ``ModelProto`` to mutate in place.
    :param prefill_with_value_info_output: When ``True``, prefill from
        ``model.graph.value_info`` and ``model.graph.output`` and prefer these
        anchors when there is a non-conflicting end-state alternative.
    :raises ValueError: If shape inference rejects a node (for
        example because of an unsupported op type) or if ``model``
        has no graph.
    """
    _C.infer_shapes_model(model, prefill_with_value_info_output)
