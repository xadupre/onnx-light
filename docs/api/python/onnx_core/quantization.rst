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
The explicit ``ORT_MATMULNBITS_INT2/INT4/INT8`` profiles are the exception:
``make_matmul_nbits_plan`` and ``export_matmul_nbits_inputs`` produce the
standard ONNX Runtime operator inputs, not internal kernel prepacking.

.. automodule:: onnx_light.onnx_core.quantization
    :members:
    :imported-members:
