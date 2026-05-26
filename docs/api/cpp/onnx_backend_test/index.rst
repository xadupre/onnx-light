onnx_backend_test
=================

This module bundles the small C++ library ``lib_onnx_backend_test`` that
mirrors the backend test node-case infrastructure from
``onnx_light.backend.test.case`` in pure C++. It only depends on
``lib_onnx_proto`` and exposes:

* a runtime :cpp:struct:`onnx::onnx_backend_test::Tensor` (distinct from
  :cpp:class:`onnx::TensorProto`) that stores raw element bytes;
* a :cpp:struct:`onnx::onnx_backend_test::TestCase` bundle of
  :cpp:class:`onnx::ModelProto` + expected input/output data sets;
* the :cpp:func:`onnx::onnx_backend_test::Expect` helper and
  :cpp:func:`onnx::onnx_backend_test::CollectTestCases` registry.

.. toctree::
    :maxdepth: 1

    simple_tensor
    simple_sequence
    test_case
    random
    cases/index
    kernels/index
