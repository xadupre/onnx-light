# source: https://github.com/onnx/onnx/blob/main/onnx/shape_inference.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
    from onnx_light.onnx import TypeProto
    from onnx_light.onnx.helper import _onnx_ir_version

    if input_data is None:
        input_data = {}
    if input_sparse_data is None:
        input_sparse_data = {}
    if ir_version is None:
        ir_version = _onnx_ir_version()

    # Check if this is an onnx_light schema (has native _infer_node_outputs
    # that accepts proto objects) or a reference-onnx schema (bytes-based).
    schema_module = type(schema).__module__
    if schema_module.startswith("onnx_light"):
        # Native onnx_light schema: pass proto objects directly.
        return schema._infer_node_outputs(
            node, input_types, dict(input_data), dict(input_sparse_data)
        )

    # Reference-onnx schema: serialise to bytes and deserialise the result.
    node_bytes = node.SerializeToString()

    # Serialise input types for each non-empty node input name.
    input_names_in_node = set(
        name.as_string() if hasattr(name, "as_string") else str(name)
        for name in node.input
        if (name.as_string() if hasattr(name, "as_string") else str(name)) != ""
    )
    passed_input_types = {
        key: value.SerializeToString()
        for key, value in input_types.items()
        if key in input_names_in_node
    }

    passed_input_data = {
        key: value.SerializeToString()
        for key, value in input_data.items()
        if key in input_names_in_node
    }
    passed_sparse_input_data = {
        key: value.SerializeToString()
        for key, value in input_sparse_data.items()
        if key in input_names_in_node
    }

    if opset_imports is None:
        passed_opset_imports: dict[str, int] = {}
    else:
        passed_opset_imports = {opset.domain: opset.version for opset in opset_imports}

    output_bytes = schema._infer_node_outputs(
        node_bytes,
        passed_input_types,
        passed_input_data,
        passed_sparse_input_data,
        passed_opset_imports,
        ir_version,
    )

    result = {}
    for key, bts in output_bytes.items():
        tp = TypeProto()
        tp.ParseFromString(bts)
        result[key] = tp
    return result
