# source: https://github.com/onnx/onnx/blob/main/onnx/shape_inference.py
from __future__ import annotations

from typing import TYPE_CHECKING

from ..onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

if TYPE_CHECKING:
    pass

_shape_inference = _C.shape_inference

InferenceError = _shape_inference.InferenceError


def infer_function_output_types(function, input_types: list, attributes: list) -> list:
    """Infers the output types of a FunctionProto given input types and attributes.

    Calls the native C++ binding which runs per-node type and shape inference
    over the function body, resolving formal attribute references.  Input
    TypeProto and AttributeProto objects are serialized to bytes before being
    passed to the C++ binding; the returned bytes are deserialized back to
    TypeProto objects.

    Args:
        function: A FunctionProto.
        input_types: A list of TypeProto objects, one per function input.
        attributes: A list of AttributeProto objects providing values for the
            function's formal attribute parameters.

    Returns:
        A list of TypeProto objects, one per function output.

    Raises:
        InferenceError: If node-level type or shape inference fails.
    """
    result = _C.shape_inference.infer_function_output_types(function, input_types, attributes)
    return result


def infer_node_outputs(
    schema,
    node,
    input_types: dict,
    input_data: dict | None = None,
    input_sparse_data: dict | None = None,
    opset_imports: list | None = None,
    ir_version: int | None = None,
) -> dict:
    """Runs type and shape inference for a single ONNX node.

    Accepts both onnx_light and reference-onnx schemas.  When an
    onnx_light schema is given its native ``_infer_node_outputs`` method
    is used directly.  When a reference-onnx schema is given the node
    and type protos are serialised to bytes, inference is delegated to
    the reference schema, and the returned bytes are deserialised back
    into onnx_light TypeProto objects.

    :param schema: An onnx_light or reference-onnx OpSchema object.
    :param node: A NodeProto (onnx_light or reference-onnx).
    :param input_types: Mapping from input name to TypeProto.
    :param input_data: Optional mapping from input name to TensorProto.
    :param input_sparse_data: Optional mapping from input name to
        SparseTensorProto.
    :param opset_imports: Optional list of OperatorSetIdProto.
    :param ir_version: IR version to use (defaults to the onnx_light IR
        version).
    :returns: Dict mapping output name to inferred TypeProto.
    """
    if input_data is None:
        input_data = {}
    if input_sparse_data is None:
        input_sparse_data = {}
    if ir_version is None:
        ir_version = _C.IR_VERSION
    return schema._infer_node_outputs(
        node, input_types, dict(input_data), dict(input_sparse_data)
    )
