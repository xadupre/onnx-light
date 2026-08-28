.. _l-design-optimization:

Pattern optimization
====================

``onnx-light`` rewrites graphs with a pattern-based optimizer built
directly on top of :ref:`l-design-graph-builder`. The optimizer
recognizes local subgraphs and replaces them with cheaper equivalents,
in the spirit of the Python pattern optimizer it is ported from. The
implementation plan and the pull requests that delivered it are recorded
in :ref:`l-next-steps-graph-builder-optimization`.

Overview
--------

Optimization always operates on a
:class:`~onnx_light.onnx_core.graph_builder.GraphBuilder` through
:class:`~onnx_light.onnx_core.optimization.GraphGraph`. ``GraphGraph``
wraps a builder with a structural index (successors, predecessors, shape,
type and constant queries) and drives a match/apply loop:

.. code-block:: python

    from onnx_light.onnx import TensorProto
    from onnx_light.onnx_core.graph_builder import GraphBuilder
    from onnx_light.onnx_core.optimization import GraphGraph, standard_patterns

    builder = GraphBuilder("optimization")
    builder.set_opset_version("", 18)
    x = builder.inp("X", TensorProto.FLOAT, [4])
    y = builder.op.Cast(x, outputs="Y", to=TensorProto.FLOAT)
    builder.out(y, TensorProto.FLOAT, [4])

    graph = GraphGraph(
        builder,
        standard_patterns(["Cast"]),
        use_global_patterns=False,
    )
    rewrites, report = graph.optimize(report=True)
    optimized_model = builder.to_onnx("model")

    assert len(rewrites) == 1
    assert report.rewrites == 1
    assert optimized_model.graph.node[0].op_type == "Identity"

Because the optimizer reuses the builder, it inherits the builder's shape
and type inference, its constant knowledge and its cleanup passes
(``RemoveIdentityNodes``, ``RemoveUnusedNodes``,
``RemoveDuplicateNodes``) instead of duplicating them.

The rewrite invariant
---------------------

A pattern must never reuse an existing name: every value it produces is
new. This invariant keeps the successor and predecessor maps valid
between two rewrites of the same iteration, which is why the builder
records every name it hands out and never reuses one.

Pattern registration
--------------------

Patterns use the same global-plus-local model as shape functions.
Registries are merged by the stable :attr:`PatternOptimization.name`; a
more local entry replaces an entry with the same name:

* **global** patterns (:func:`register_pattern`) are used by every new
  ``GraphGraph``; the standard ONNX patterns are registered globally when
  the module is imported;
* **builder** patterns (``GraphBuilder.register_pattern``) override a
  global pattern for optimizers built over that builder;
* **graph** patterns (``GraphGraph(builder, patterns=[...])``) have the
  highest precedence and are retained for that optimizer, including
  recursive subgraphs.

Patterns can be written in C++ or in Python; both share the
:class:`~onnx_light.onnx_core.optimization.PatternOptimization`
interface, a ``match`` step that returns a
:class:`~onnx_light.onnx_core.optimization.MatchResult` and an ``apply``
step that produces the replacement nodes.

API reference
-------------

* **Python API**: :mod:`onnx_light.onnx_core.optimization`; the runtime
  list of registered patterns is available through
  :func:`~onnx_light.onnx_core.optimization.standard_pattern_names`.
* **C++ API**: :doc:`/api/cpp/onnx_core/builder/index`.

Examples
--------

* :ref:`l-example-plot-pattern-optimization` covers optimization statistics.
* :ref:`l-example-plot-pattern-replay` demonstrates deterministic replay from
  captured rewrites.
* :ref:`l-howto-add-custom-pattern` is a Python/C++ how-to on writing a
  custom pattern and choosing its priority.
