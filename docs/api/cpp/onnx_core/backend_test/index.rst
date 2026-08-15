backend_test
============

The ``backend_test`` sub-namespace of ``onnx_core``
(``core::backend_test``) contains the :cpp:struct:`TestCase` definition
and the :cpp:func:`CollectTestCases` / :cpp:func:`CollectTestCasesByName`
aggregator functions that assemble the full set of backend test cases from
the per-operator registries in ``onnx_extensions/backend_test/cases/``.

.. toctree::
    :maxdepth: 1

    test_case
    test_case_registry
    expect
    io_data
