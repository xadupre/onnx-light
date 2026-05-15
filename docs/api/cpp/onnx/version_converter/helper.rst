helper.h
========

Broadcasting compatibility checks and input-validity assertions shared by the
per-opset adapter implementations in :cpp:any:`onnx::version_conversion`.

The helpers fall into two groups:

* **Broadcasting checks** –
  :cpp:func:`onnx::version_conversion::check_numpy_unibroadcastable_and_require_broadcast`
  and
  :cpp:func:`onnx::version_conversion::assert_numpy_multibroadcastable`
  verify that a pair of input shapes conform to NumPy broadcasting rules
  (unidirectional for pre-opset-7 operators; multidirectional for opset ≥ 7).

* **Input-validity assertions** –
  :cpp:func:`onnx::version_conversion::assertNotParams`,
  :cpp:func:`onnx::version_conversion::assertInputsAvailable`, and the inline
  utility :cpp:func:`onnx::version_conversion::ReadInt64Tensor` ensure that
  adapters receive well-formed, concretely-shaped inputs before attempting any
  shape arithmetic or tensor decoding.

.. doxygenfile:: helper.h
   :project: onnx-light
