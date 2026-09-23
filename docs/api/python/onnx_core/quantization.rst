onnx_light.onnx_core.quantization
=================================

See :ref:`the quantization guide <l-quantized-values>` for the profile catalogue,
parameter reference and numerical contracts, and
:ref:`the runnable Python tutorial <l-example-quantization-profiles>` for examples
covering every profile.

The usual NumPy path is ``numpy_helper.from_array`` ->
``quantize_tensor_proto`` -> ``EncodedValueProto`` ->
``dequantize_tensor_proto`` -> ``numpy_helper.to_array``.
Profiles define portable storage defaults, not calibration algorithms or
vendor-compatible binary formats.

.. automodule:: onnx_light.onnx_core.quantization
    :members:
    :imported-members:
