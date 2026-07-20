onnx_backend_test
=================

This module bundles the small C++ library ``lib_onnx_backend_test`` that
mirrors the backend test node-case infrastructure from
``onnx_light.onnx_lib.backend.test.case`` in pure C++. It depends on
``lib_onnx_proto`` and ``lib_onnx_kernels`` (for the runtime
:cpp:struct:`onnx_light::core::backend_test::TestCase` and the operator kernel
implementations) and provides the per-operator test-case registries
under ``onnx_backend_test/cases/``.

The :cpp:struct:`TestCase` definition and its registration helpers
(:cpp:func:`CollectTestCases`, :cpp:func:`CollectTestCasesByName`) live in
``onnx_core/backend_test`` (namespace ``core::backend_test``); see
:doc:`../onnx_core/backend_test/index`. The header
``onnx_backend_test/test_case.h`` re-exports those names into the
``onnx_backend_test`` namespace for backward compatibility.

See the :doc:`../onnx_kernels/index` module for the runtime data model
and the kernel implementations used by these test cases.

.. toctree::
    :maxdepth: 1

    cases/index
    cases_for_shapes/index
    cases_numerical/index
    cases_runtime/index
    test_case
