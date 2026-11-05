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
The separate ``quantize_nvfp4_weights``, ``export_nvfp4_weights`` and
``import_nvfp4_weights`` APIs preserve a complete fixed-weight NVFP4 hierarchy
(FP8 local scales and a FLOAT32 global scale), unlike the portable ``nvfp4``
profile. They do not implement activation-local dynamic scaling.
The ``MXFP6_E2M3`` profile and ``quantize_mxfp6_e2m3_weights`` implement the
:ref:`E2M3/group-32/E8M0 contract <l-mxfp6-e2m3-weights>`. The export/import
functions preserve packed weight and scale bytes. Legacy ``MXFP6`` and
``FP6_LLM`` remain E3M2 profiles.

The ``MXFP8`` profile and ``quantize_mxfp8_weights`` implement the
:ref:`complete E4M3FN/group-32/E8M0 contract <l-mxfp8-weights>` with automatic
scales. ``export_mxfp8_weights`` and ``import_mxfp8_weights`` preserve the
native FP8 and E8M0 bytes; they do not provide accelerated matrix multiplication.

For :ref:`model-level shared parameters <l-shared-quantization-parameters>`,
``add_quantization_parameters`` declares a fixed set and
``quantize_tensor_shared`` returns a resource-retaining ``SharedQuantizedValue``.
``materialize_quantized_value`` exports a self-contained message when the
declaring model will not accompany the encoded value.

.. automodule:: onnx_light.onnx_core.quantization
    :members:
    :imported-members:
