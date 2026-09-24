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

__all__ = [
    "MatMulNBitsInputs",
    "QuantizationBlockLayout",
    "QuantizationBlockParameters",
    "QuantizationFormat",
    "QuantizationMethod",
    "QuantizationPlan",
    "QuantizationRun",
    "Shape",
    "dequantize_tensor",
    "dequantize_tensor_proto",
    "export_matmul_nbits_inputs",
    "make_matmul_nbits_plan",
    "make_quantization_plan",
    "make_quantization_type",
    "parse_quantization_format",
    "quantization_format_name",
    "quantization_formats",
    "quantize_tensor",
    "quantize_tensor_proto",
]
