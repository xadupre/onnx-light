.. _l-design-test-coverage:

Backend test-case coverage
===========================

This page describes how the test-case coverage of *onnx-light* is measured
against the lightweight ONNX operator schemas (``LightOpSchema``).

----

What is measured
----------------

The *baseline* is the set of lightweight operator schemas exposed by the
C++ ``onnx_op`` extension (one schema per operator, latest opset version).
For each operator, the union of allowed tensor types across all its type
constraints defines a list of supported types. Every
``(domain, operator_name, type)`` triple is called a *signature*.

The *covered signatures* are the signatures for which at least one backend
test case — collected via
:func:`onnx_light.backend.test.case.base.collect_test_case` — exists whose
single-node model uses that operator and that ONNX type on one of its typed
graph inputs or outputs.

The *coverage ratio* is the number of covered signatures divided by the
total number of signatures.

Computing the report
--------------------

The function :func:`onnx_light.backend.coverage.compute_test_case_coverage`
collects every available test case (Python-registered cases from
:class:`~onnx_light.backend.test.case.base.Base` subclasses and the
canonical C++ cases from ``lib_onnx_backend_test``) and produces a
:class:`~onnx_light.backend.coverage.CoverageReport`:

.. runpython::
    :showcode:

    from onnx_light.backend.coverage import compute_test_case_coverage

    report = compute_test_case_coverage()
    print(report)
    print(f"ratio = {report.ratio:.3f}")
    print(f"operators with no test case: {len(report.uncovered_operators)}")
    for domain, name in report.uncovered_operators[:10]:
        print(f"  - {domain}::{name}")

The report exposes:

* ``total_signatures`` — total number of ``(operator, type)`` signatures
  derived from the baseline schemas;
* ``covered_signatures`` — number of those signatures covered by at least
  one test case;
* ``ratio`` — coverage ratio in ``[0, 1]``;
* ``operator_coverages`` — per-operator breakdown including
  ``supported_types``, ``covered_types`` and ``missing_types``;
* ``uncovered_operators`` — list of ``(domain, operator_name)`` pairs that
  have **no** test case at all.

Listing operators without a test case
--------------------------------------

The ``uncovered_operators`` attribute makes it straightforward to drive
the prioritization of new backend test cases. It only contains operators
that have zero test cases — operators partially covered (for example
``Abs`` for which only ``tensor(float)`` is exercised) are listed in
``operator_coverages`` with a non-empty ``missing_types`` property.

See also
--------

* :ref:`l-api-backend-coverage`
* :ref:`l-design-backend-tests`
