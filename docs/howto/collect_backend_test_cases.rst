.. _l-howto-collect-backend-test-cases:

:html_theme.sidebar_secondary.remove:

How to collect backend test cases (by op type or by name)
=========================================================

*onnx-light* ships a catalog of C++-implemented ONNX backend test cases.
This page shows how to enumerate them, how to filter them by operator
type or category, and how to look one up by name (substring or full
regular expression).  The same data is exposed in Python and in C++.

List every test case
--------------------

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx.backend as bt

          cases = bt.collect_test_cases()
          print(len(cases), "cases")
          for tc in cases[:5]:
              print(tc.name, tc.kind, tc.tag)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/backend_test/test_case.h"

          auto cases = core::backend_test::CollectTestCases();
          for (const auto &tc : cases) {
              std::cout << tc.name << " " << tc.kind << " " << tc.tag << "\n";
          }

Filter by operator type or category
-----------------------------------

:func:`~onnx_light.onnx.backend.collect_test_cases` accepts an operator type
(for example ``"Add"``) or one of the special category strings ``"shape"``,
``"inference"``, or ``"nan_inf"`` to narrow down the result.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx.backend as bt

          add_cases = bt.collect_test_cases("Add")
          shape_cases = bt.collect_test_cases("shape")

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/backend_test/test_case.h"

          auto add_cases = core::backend_test::CollectTestCases("Add");
          auto shape_cases = core::backend_test::CollectTestCases("shape");

Collect a test case by name
---------------------------

Use :func:`onnx_light.onnx.backend.collect_test_cases_by_name` (or
:cpp:func:`onnx_light::core::backend_test::CollectTestCasesByName` in C++) to
look up one or more cases by their ``name``.  The pattern is matched with
``std::regex_search`` ECMAScript semantics, so a plain string acts as a
substring match; anchor it with ``^...$`` for a full match.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import re
          import onnx_light.onnx.backend as bt

          # Substring match: every case whose name contains "abs".
          cases = bt.collect_test_cases_by_name("abs")

          # Full regex: only the C++ "test_cc_add" cases.
          cases = bt.collect_test_cases_by_name(r"^test_cc_add(_|$)")

          # A pre-compiled re.Pattern is also accepted.
          cases = bt.collect_test_cases_by_name(re.compile(r"abs"))

          for tc in cases:
              print(tc.name)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/backend_test/test_case.h"

          // Substring match
          auto cases = core::backend_test::CollectTestCasesByName("abs");

          // Full regex
          auto cc_add = core::backend_test::CollectTestCasesByName(
              "^test_cc_add(_|$)");

          for (const auto &tc : cases) {
              std::cout << tc.name << "\n";
          }

Release materialized cases
--------------------------

Every case returned by the C++ collectors is initially unmaterialized,
including manually assembled control-flow, sequence, and shape-analysis
cases. Accessing ``model`` or ``data_sets`` builds and caches both values.
Use ``materialized`` to inspect that state without triggering the build and
``unload()`` to release the cached payload and build-time resources, including
kernel instances captured by the initial generator:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx.backend as bt

          case = bt.collect_test_cases_by_name(r"^test_cc_if_seq$")[0]
          assert not case.materialized

          model = case.model
          assert case.materialized

          case.unload()
          assert not case.materialized

          # Existing handles remain valid. A new access rebuilds the cache.
          assert model.graph.node[0].op_type == "If"
          assert case.model.graph.node[0].op_type == "If"

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/backend_test/test_case.h"

          auto cases = core::backend_test::CollectTestCasesByName(
              "^test_cc_if_seq$");
          auto &test_case = cases.front();
          auto model = test_case.model_handle();
          test_case.unload();
          assert(!test_case.materialized());
          assert(model->ref_graph().ref_node()[0].ref_op_type() == "If");

Python collection and report APIs use ``unload=True`` by default. Native-backed
Python cases retain only their lightweight C++ reconstruction recipe until
``model`` or ``data_sets`` is accessed. Processing helpers unload each case in
a ``finally`` block, so exceptional paths do not retain its payload. Pass
``unload=False`` to ``collect_test_case``, ``get_test_case``,
``get_test_cases_for_op``, ``make_test_class``,
``compute_test_case_coverage``, or ``compute_runtime_coverage`` when deliberate
retention is more useful than bounded peak memory.

``unload()`` only releases ownership held by the case. Existing Python
objects and C++ handles returned by ``model_handle()`` or
``data_set_handles()`` use shared ownership and therefore keep their
referenced payload alive until those handles are also discarded. Plain C++
references returned by ``model()`` or ``data_sets()`` become invalid when
the payload is unloaded. The lightweight collector fallback retained by the
case stores only the information needed to find and rebuild it; it does not
retain the original kernel instance. A later access rematerializes a fresh
payload and creates a temporary kernel for that build.

Notes
-----

* The Python ``pattern`` argument accepts either a :class:`str` or a
  pre-compiled :class:`re.Pattern`; in the latter case the ``pattern``
  source string is forwarded to the C++ side.  Passing any other type
  raises :class:`TypeError`, and an invalid regular expression raises
  :class:`ValueError`.
* The C++ overload throws ``std::regex_error`` for an invalid pattern.
* An empty ``name_regex`` matches every case and is equivalent to
  calling :func:`~onnx_light.onnx.backend.collect_test_cases` /
  :cpp:func:`onnx_light::core::backend_test::CollectTestCases` with no
  arguments.
* :class:`TestCase` exposes ``name``, ``kind``, ``tag``, ``rtol``, ``atol``,
  ``materialized``, ``unload()``, ``data_sets``, and a lazily resolved ``model``
  (:class:`onnx_light.onnx.ModelProto`), so a case retrieved by name can
  be fed directly to a :class:`~onnx_light.onnx.reference.ReferenceEvaluator`
  or to any other backend runner.
* Correctness cases registered with :cpp:func:`Expect` produce their expected
  outputs by directly calling the named built-in *onnx-light* kernel. They do
  not resolve a process-wide dispatch-table entry or a session/custom kernel,
  so an external override cannot become the oracle. Benchmark case generation
  is separate and is not covered by this oracle guarantee.

See also
--------

* :ref:`l-how-to` - other onnx-light how-to recipes.
