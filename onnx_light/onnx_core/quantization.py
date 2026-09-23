"""Converts tensors to portable, self-describing quantized values.

Profiles describe onnx-light representations, not vendor binary layouts or
calibration algorithms. See :doc:`/howto/quantized_values` for their contracts.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

QuantizationMethod = runtime.QuantizationMethod
QuantizationBlock = runtime.QuantizationBlock
QuantizationPlan = runtime.QuantizationPlan
quantization_formats = runtime.quantization_formats
make_quantization_plan = runtime.make_quantization_plan
quantize_tensor = runtime.quantize_tensor
dequantize_tensor = runtime.dequantize_tensor
quantize_tensor_proto = runtime.quantize_tensor_proto
dequantize_tensor_proto = runtime.dequantize_tensor_proto

__all__ = [
    "QuantizationBlock",
    "QuantizationMethod",
    "QuantizationPlan",
    "dequantize_tensor",
    "dequantize_tensor_proto",
    "make_quantization_plan",
    "quantization_formats",
    "quantize_tensor",
    "quantize_tensor_proto",
]
