.. _l-design-inference-coverage:

Shape-inference coverage (onnx_optim)
=====================================

.. role:: green
.. role:: red

This page reports, for every backend test case **tagged**
``"inference"`` (see :attr:`onnx_light.backend.test.case.base.TestCase.tag`),
the outcome of running the ``onnx_optim`` shape-inference pipeline against
the *expected* intermediate and output shapes recorded by the test author.

For every collected case, the report:

#. renders the original model — including the expected ``value_info`` and
   output shapes — as a :epkg:`Mermaid` flowchart;
#. clones the model and **strips its ``graph.value_info``** so shape
   inference cannot just reuse the recorded intermediate shapes;
#. runs :func:`onnx_light.onnx_optim.shape_inference.infer_shapes_model`
   on the stripped clone;
#. emits a side-by-side table contrasting the *expected* and the
   *computed* shape of every input, intermediate and output value, with
   the match status colored :green:`yes` / :red:`no`.

When shape inference itself raises, the error message is shown instead
of the comparison table.

----

Summary
-------

Overall pass rate across all ``"inference"``-tagged backend test cases:

.. runpython::
    :rst:

    from onnx_light.doc import (
        compute_inference_coverage,
        render_rst_summary,
    )

    _report = compute_inference_coverage()
    print(render_rst_summary(_report))

----

Per-case details
----------------

.. runpython::
    :rst:

    from onnx_light.doc import (
        compute_inference_coverage,
        render_rst_report,
    )

    _report = compute_inference_coverage()
    print(render_rst_report(_report))

See also
--------

* :ref:`l-design-backend-tests`
* :ref:`l-design-runtime-coverage`
* :func:`onnx_light.onnx_optim.shape_inference.infer_shapes_model`
* :func:`onnx_light.tools.to_mermaid`
