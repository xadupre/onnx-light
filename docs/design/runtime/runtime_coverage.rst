.. _l-design-runtime-coverage:

Runtime test coverage (onnxruntime and shape_inference)
=======================================================

.. role:: green
.. role:: red

This page reports, for every backend test case collected by
:func:`onnx_light.onnx_lib.backend.test.case.collect_test_case`, the outcome of
three independent scenarios:

* **onnxruntime (CPU)** — the model is executed with :epkg:`onnxruntime` on
  the CPU execution provider. The cell shows the maximum absolute
  discrepancy between the reference outputs and the ORT outputs, colored
  :green:`green` when it stays within the test-case ``atol``/``rtol``
  tolerances and :red:`red` otherwise (``n/a`` when ORT cannot load or run
  the model — typically when the model uses an op from a domain ORT does not
  register, such as ``ai.onnx.preview`` or ``ai.onnx.preview.training``).
* **static shape** — every recorded ``graph.value_info`` and output shape
  is stripped from the model, then the model is passed to
  :func:`onnx_light.onnx_lib.shape_inference.infer_shapes`. A
  :green:`yes` indicates shape inference completes without raising **and**
  every output shape originally recorded by the test author is recovered
  by inference (concrete dims must match exactly; symbolic or missing dims
  are accepted on either side).
* **dynamic_shapes** — same as **static shape**, but every numeric input
  dimension is first replaced with a symbolic ``dim_param`` (identical
  numeric values share the same symbol) before stripping the recorded
  output shapes and running shape inference on the resulting symbolic
  model.

----

Summary
-------

Global pass rates across all collected test cases:

.. runpython::
    :rst:

    from onnx_light.onnx_lib.backend.runtime_coverage import (
        compute_runtime_coverage,
        render_rst_summary,
    )

    _report = compute_runtime_coverage()
    print(render_rst_summary(_report))

Per-domain pass rates:

.. runpython::
    :rst:

    from onnx_light.onnx_lib.backend.runtime_coverage import (
        compute_runtime_coverage,
        render_rst_domain_summary,
    )

    print(render_rst_domain_summary(compute_runtime_coverage()))

----

Per-test-case status
--------------------

Each domain section below lists every backend test case from that domain. The
tables are rendered with the :epkg:`sphinx-datatables` extension, so they are
interactive: use the search box to filter by op or test name and click on a
column header to sort.

.. runpython::
    :rst:

    from onnx_light.onnx_lib.backend.runtime_coverage import (
        compute_runtime_coverage,
        render_rst_domain_sections,
    )

    print(render_rst_domain_sections(compute_runtime_coverage()))

----

See also
--------

* :doc:`/api/cpp/onnx_extensions/backend_test/index`
* :ref:`l-design-test-coverage`
