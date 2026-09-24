"""Converts tensors to portable, self-describing quantized values.

Profiles describe onnx-light representations, not vendor binary layouts or
calibration algorithms. See :doc:`/howto/quantized_values` for their contracts.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

QuantizationMethod = runtime.QuantizationMethod
QuantizationFormat = runtime.QuantizationFormat
QuantizationBlockLayout = runtime.QuantizationBlockLayout
QuantizationBlockParameters = runtime.QuantizationBlockParameters
QuantizationRun = runtime.QuantizationRun
QuantizationPlan = runtime.QuantizationPlan
quantization_formats = runtime.quantization_formats
quantization_format_name = runtime.quantization_format_name
parse_quantization_format = runtime.parse_quantization_format
make_quantization_plan = runtime.make_quantization_plan
quantize_tensor = runtime.quantize_tensor
dequantize_tensor = runtime.dequantize_tensor
quantize_tensor_proto = runtime.quantize_tensor_proto
dequantize_tensor_proto = runtime.dequantize_tensor_proto

__all__ = [
    "QuantizationBlockLayout",
    "QuantizationBlockParameters",
    "QuantizationFormat",
    "QuantizationMethod",
    "QuantizationPlan",
    "QuantizationRun",
    "dequantize_tensor",
    "dequantize_tensor_proto",
    "make_quantization_plan",
    "parse_quantization_format",
    "quantization_format_name",
    "quantization_formats",
    "quantize_tensor",
    "quantize_tensor_proto",
]
