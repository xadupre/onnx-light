onnx_kernels
============

This module documents the C++ static library ``lib_onnx_kernels`` that
bundles the ONNX operator kernel implementations together with the
runtime infrastructure they rely on (``TestCase``, ``Tensor``,
``run_nodes``, ``random`` …). It only depends on ``lib_onnx_proto`` and
exposes:

* a runtime :cpp:struct:`onnx_light::onnx_kernels::Tensor` (distinct from
  :cpp:class:`onnx_light::TensorProto`) that stores raw element bytes;
* a :cpp:struct:`onnx_light::onnx_backend_test::TestCase` bundle of
  :cpp:class:`onnx_light::ModelProto` + expected input/output data sets;
* the :cpp:func:`onnx_light::onnx_backend_test::Expect` helper and
  :cpp:func:`onnx_light::onnx_backend_test::CollectTestCases` registry;
* the ONNX operator kernel implementations themselves under
  ``onnx_kernels/kernels/``.

.. toctree::
    :maxdepth: 1

    simple_tensor
    simple_sequence
    random
    kernels/index
