.. _l-design-schema-comparison:

Comparison: ``onnx`` ``OpSchema`` vs ``onnx_light`` ``LightOpSchema``
=====================================================================

This page programmatically compares the operator schemas exposed by the
reference :epkg:`onnx` package (:class:`onnx.defs.OpSchema`) with the
lightweight schemas exposed by ``onnx_light``
(``LightOpSchema``, from the C++ ``onnx_op`` extension).

For every operator known to either side the table below reports:

* whether the operator is described by an ``OpSchema`` (``onnx``);
* whether the operator is described by a ``LightOpSchema`` (``onnx_light``);
* whether a shape-inference function is registered on the ``onnx`` side
  (``OpSchema.has_type_and_shape_inference_function``);
* whether a shape-inference function is registered on the ``onnx_light`` side.
  Shape inference for ``onnx_light`` lives in the ``onnx_optim`` library
  (``onnx_light/onnx_optim/shapes/shape_inference.cc``), separate from the
  schema definitions themselves;
* how many node backend tests exercise the operator in each package.
  A test case is attributed to the operator whose lowercased or
  ``snake_case`` name matches its ``test_<op>(_<variant>)*`` name —
  mirroring the upstream ``onnx/backend/test/data/node`` subfolder
  naming convention. ``onnx_light``'s own ``test_cc_<op>`` cases are
  attributed the same way (the ``cc_`` prefix is stripped).

.. note::

    The page is regenerated every time the documentation is built, so the
    numbers always reflect the operator schemas and backend tests shipped
    with the installed ``onnx`` and ``onnx_light`` packages. When the
    reference ``onnx`` package is not installed, the ``onnx``-side columns
    are reported as empty / zero.

Summary
-------

.. runpython::
    :rst:

    from onnx_light.schema_comparison import (
        compute_schema_comparison,
        render_rst_summary,
    )

    _comparison = compute_schema_comparison()
    print(render_rst_summary(_comparison))

----

Per-operator comparison
-----------------------

The table below is rendered with the :epkg:`sphinx-datatables` extension,
as one table per domain. Each table is interactive: use the search box
to filter operators by any column (name, checkbox flags, test counts)
and click on a
column header to sort the table by that column (click again to toggle
ascending / descending order).

.. runpython::
    :rst:

    from onnx_light.schema_comparison import (
        compute_schema_comparison,
        render_rst_table,
    )

    print(render_rst_table(compute_schema_comparison(), css_class="sphinx-datatable"))

----

See also
--------

* :ref:`l-design-backend-tests`
* :ref:`l-design-test-coverage`
