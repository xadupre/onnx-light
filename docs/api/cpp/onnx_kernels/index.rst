onnx_kernels
============

This module documents the C++ static library ``lib_onnx_kernels`` that
bundles the ONNX operator kernel implementations together with the
runtime infrastructure they rely on (``TestCase``, ``Tensor``,
``run_nodes``, ``random`` …). It only depends on ``lib_onnx_proto`` and
exposes:

* a runtime :cpp:struct:`onnx::onnx_kernels::Tensor` (distinct from
  :cpp:class:`onnx::TensorProto`) that stores raw element bytes;
* a :cpp:struct:`onnx::onnx_kernels::TestCase` bundle of
  :cpp:class:`onnx::ModelProto` + expected input/output data sets;
* the :cpp:func:`onnx::onnx_kernels::Expect` helper and
  :cpp:func:`onnx::onnx_kernels::CollectTestCases` registry;
* the ONNX operator kernel implementations themselves under
  ``onnx_kernels/kernels/``.

.. toctree::
    :maxdepth: 1

    simple_tensor
    simple_sequence
    random
    kernels/index
