"""Converts tensors to portable, self-describing quantized values.

Portable profiles describe onnx-light representations, not vendor binary layouts.
ORT_MATMULNBITS_INT2/INT4/INT8 produce ONNX Runtime operator inputs, not internal
kernel prepacking. No profile performs calibration. See :doc:`/howto/quantized_values`.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

QuantizationMethod = runtime.QuantizationMethod
Shape = runtime.Shape
QuantizationFormat = runtime.QuantizationFormat
QuantizationBlockLayout = runtime.QuantizationBlockLayout
QuantizationBlockParameters = runtime.QuantizationBlockParameters
QuantizationRun = runtime.QuantizationRun
QuantizationPlan = runtime.QuantizationPlan
MatMulNBitsInputs = runtime.MatMulNBitsInputs
quantization_formats = runtime.quantization_formats
quantization_format_name = runtime.quantization_format_name
parse_quantization_format = runtime.parse_quantization_format
make_quantization_plan = runtime.make_quantization_plan
make_quantization_type = runtime.make_quantization_type
make_matmul_nbits_plan = runtime.make_matmul_nbits_plan
export_matmul_nbits_inputs = runtime.export_matmul_nbits_inputs
quantize_tensor = runtime.quantize_tensor
dequantize_tensor = runtime.dequantize_tensor
quantize_tensor_proto = runtime.quantize_tensor_proto
dequantize_tensor_proto = runtime.dequantize_tensor_proto
SharedQuantizedValue = runtime.SharedQuantizedValue
materialize_quantized_value = runtime.materialize_quantized_value
make_shared_quantization_type = runtime.make_shared_quantization_type
quantize_tensor_shared = runtime.quantize_tensor_shared


def add_quantization_parameters(
    model,
    name,
    storage_type,
    logical_type,
    *,
    scales,
    zero_points=None,
    offsets=None,
    codebooks=None,
    permutation=None,
    forward=None,
    inverse=None,
    outliers=None,
):
    """Adds a fixed numerical parameter set backed by root graph initializers.

    Numerical arguments accept TensorProto or NumPy arrays. Descriptors and
    tensors are copied; the model is changed only after successful validation.
    Returns:
        The compact StructTypeProto used by encoded values referring to this set.
    """
    import numpy
    from .. import onnx
    from ..onnx import numpy_helper

    prefix = "onnx_light.quantization.parameters:"
    if not isinstance(name, str) or not name:
        raise ValueError("The parameter set name must be a nonempty string.")
    if any(a.tensor_name == prefix + name for a in model.graph.quantization_annotation):
        raise ValueError(f"Duplicate quantization parameter set {name!r}.")
    if scales is None:
        raise ValueError("Shared quantization requires fixed scales.")
    candidate = onnx.ModelProto()
    candidate.CopyFrom(model)
    annotation = onnx.TensorAnnotation(tensor_name=prefix + name)
    parameters = dict(
        storage_type=numpy.frombuffer(storage_type.SerializeToString(), dtype=numpy.uint8),
        logical_type=numpy.frombuffer(logical_type.SerializeToString(), dtype=numpy.uint8),
        scales=scales,
        zero_points=zero_points,
        offsets=offsets,
        codebooks=codebooks,
        permutation=permutation,
        forward=forward,
        inverse=inverse,
        outliers=outliers,
    )
    names = {tensor.name for tensor in candidate.graph.initializer}
    additions = []
    for role, value in parameters.items():
        if value is None:
            continue
        tensor_name = f"{prefix}{name}/{role}"
        if tensor_name in names:
            raise ValueError(f"Shared parameter initializer collision: {tensor_name!r}.")
        if isinstance(value, onnx.TensorProto):
            tensor = onnx.TensorProto()
            tensor.CopyFrom(value)
            tensor.name = tensor_name
        else:
            tensor = numpy_helper.from_array(numpy.asarray(value), name=tensor_name)
        candidate.graph.initializer.append(tensor)
        additions.append(tensor)
        annotation.quant_parameter_tensor_names.append(
            onnx.StringStringEntryProto(key=role, value=tensor_name)
        )
    candidate.graph.quantization_annotation.append(annotation)
    compact = make_shared_quantization_type(candidate, name)
    model.graph.initializer.extend(additions)
    model.graph.quantization_annotation.append(annotation)
    return compact


__all__ = [
    "MatMulNBitsInputs",
    "QuantizationBlockLayout",
    "QuantizationBlockParameters",
    "QuantizationFormat",
    "QuantizationMethod",
    "QuantizationPlan",
    "QuantizationRun",
    "Shape",
    "SharedQuantizedValue",
    "add_quantization_parameters",
    "dequantize_tensor",
    "dequantize_tensor_proto",
    "export_matmul_nbits_inputs",
    "make_matmul_nbits_plan",
    "make_quantization_plan",
    "make_quantization_type",
    "make_shared_quantization_type",
    "materialize_quantized_value",
    "parse_quantization_format",
    "quantization_format_name",
    "quantization_formats",
    "quantize_tensor",
    "quantize_tensor_proto",
    "quantize_tensor_shared",
]
