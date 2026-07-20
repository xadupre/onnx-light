backend_test
============

The ``backend_test`` sub-namespace of ``onnx_core``
(``core::backend_test``) contains the :cpp:struct:`TestCase` definition
and the :cpp:func:`CollectTestCases` / :cpp:func:`CollectTestCasesByName`
aggregator functions that assemble the full set of backend test cases from
the per-operator registries in ``onnx_backend_test/cases/``.

The companion compatibility header ``onnx_backend_test/test_case.h``
re-exports every name from this namespace into the legacy
``onnx_backend_test`` namespace so that existing case source files
continue to compile unchanged.

.. toctree::
    :maxdepth: 1

    test_case
