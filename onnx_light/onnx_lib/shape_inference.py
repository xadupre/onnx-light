# source: https://github.com/onnx/onnx/blob/main/onnx/shape_inference.py
from __future__ import annotations

from ..onnx_py import _onnxpyprotolib as _C  # type: ignore

_shape_inference = _C.shape_inference  # type: ignore

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
    result = _C.shape_inference.infer_function_output_types(function, input_types, attributes)  # type: ignore
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
        from ..onnx.defs import IR_VERSION

        ir_version = IR_VERSION  # type: ignore
    opset_imports_by_domain = (
        {opset.domain: opset.version for opset in opset_imports} if opset_imports else {}
    )
    return schema._infer_node_outputs(
        node,
        input_types,
        dict(input_data),
        dict(input_sparse_data),
        opset_imports_by_domain,
        ir_version,
    )


def infer_shapes(model, check_type=False, strict_mode=False, data_prop=False):
    """Runs whole-model shape inference in place on the given model.

    Calls the native C++ ``shape_inference::InferShapes(ModelProto&)``
    routine over the given :class:`~onnx_light.onnx.ModelProto`. The model is
    modified in place: shape and type information is populated on every
    intermediate ``ValueInfoProto`` and graph output. The same model object is
    returned for convenience.

    :param model: A :class:`~onnx_light.onnx.ModelProto`.
    :param check_type: Checks the type-equality for input and output.
    :param strict_mode: When True, raises an error on any shape inference
        failure; when False, silently stops on errors without raising.
    :param data_prop: Enables data propagation for limited operators to
        perform shape computation.
    :returns: The same model with inferred shapes/types.
    :raises InferenceError: If shape inference fails on any node.
    """
    _shape_inference.infer_shapes(model, check_type, strict_mode, data_prop)
    return model


def infer_shapes_path(
    model_path: str,
    output_path: str = "",
    check_type: bool = False,
    strict_mode: bool = False,
    data_prop: bool = False,
) -> None:
    """Runs shape inference on a model file and saves the result.

    Loads the model from *model_path*, runs shape inference, and writes the
    result to *output_path* (or overwrites *model_path* when *output_path* is
    empty).

    This avoids loading the full model into Python memory twice and mirrors
    the upstream ``onnx.shape_inference.infer_shapes_path`` API.

    :param model_path: Path to the input ONNX model.
    :param output_path: Path to write the inferred model.  When empty,
        the input file is overwritten.
    :param check_type: Checks the type-equality for input and output.
    :param strict_mode: When True, raises an error on any shape inference
        failure; when False, silently stops on errors without raising.
    :param data_prop: Enables data propagation for limited operators to
        perform shape computation.
    """
    from ..onnx_proto._io_helper import load, save

    model = load(model_path)
    _shape_inference.infer_shapes(model, check_type, strict_mode, data_prop)
    save(model, output_path or model_path)
