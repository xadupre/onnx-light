onnx_backend_test
=================

This module bundles the small C++ library ``lib_onnx_backend_test`` that
mirrors the backend test node-case infrastructure from
``onnx_light.backend.test.case`` in pure C++. It depends on
``lib_onnx_proto`` and ``lib_onnx_kernels`` (for the runtime
:cpp:struct:`onnx::onnx_kernels::TestCase` and the operator kernel
implementations) and provides the per-operator test-case registries
under ``onnx_backend_test/cases/``.

See the :doc:`../onnx_kernels/index` module for the runtime data model
and the kernel implementations used by these test cases.

.. toctree::
    :maxdepth: 1

    cases/index
    test_case
