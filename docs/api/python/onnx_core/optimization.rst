onnx_light.onnx_core.optimization
=================================

.. currentmodule:: onnx_light.onnx_core.optimization

Optimization workflow
+++++++++++++++++++++

Optimization always operates on a
:class:`~onnx_light.onnx_core.graph_builder.GraphBuilder` through
:class:`~onnx_light.onnx_core.optimization.GraphGraph`:

.. code-block:: python

    from onnx_light.onnx_core.optimization import GraphBuilder, GraphGraph

    builder = GraphBuilder(model)
    graph = GraphGraph(builder)
    rewrites, report = graph.optimize(report=True)
    optimized_model = builder.to_onnx("model")

Pattern registration
++++++++++++++++++++

Patterns use the same global-plus-local model as shape functions. Registries
are merged by the stable :attr:`PatternOptimization.name`; a more local entry
replaces an entry with the same name:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Scope
     - Registration
     - Selection
   * - Global
     - :func:`register_pattern`
     - Used by every new ``GraphGraph``. The standard ONNX patterns are
       registered globally when this module is imported.
   * - Builder
     - :meth:`GraphBuilder.register_pattern`
     - Overrides a global pattern for optimizers built over that builder.
   * - Graph
     - ``GraphGraph(builder, patterns=[...])``
     - Has the highest precedence and is retained for that optimizer,
       including recursive subgraphs.

Pass ``use_global_patterns=False`` to ``GraphGraph`` to use only builder and
graph registrations. :func:`clear_registered_patterns` clears the global
registry; :func:`reset_registered_patterns` restores the standard patterns.

Registered standard patterns
++++++++++++++++++++++++++++

The following table lists the standard patterns registered when this module is
imported. It is generated from the live registry, so it always reflects the
currently available patterns.

.. runpython::
    :rst:

    from onnx_light.onnx_core.optimization import render_rst_standard_patterns_table

    print(render_rst_standard_patterns_table())

The runtime list is available through :func:`standard_pattern_names`.
The :ref:`complete pattern catalogue <l-api-pattern-catalog>` adds the C++
documentation link and the Before/After rewrite graph for every entry.

See :ref:`l-howto-add-custom-pattern` for a Python/C++ how-to on writing a
custom pattern and choosing its priority, and
:ref:`l-example-plot-pattern-optimization` for a runnable example covering
statistics and replay.

Custom Python pattern
+++++++++++++++++++++

.. code-block:: python

    import onnx_light.onnx.helper as oh
    from onnx_light.onnx_core.optimization import (
        GraphBuilder,
        GraphGraph,
        PatternOptimization,
    )

    class NegNegPattern(PatternOptimization):
        def __init__(self):
            super().__init__(priority=1, name="NegNeg")

        def fast_op_type(self):
            return {"Neg"}

        def match(self, graph, node):
            previous = graph.node_before(node.input[0])
            if previous is None or previous.op_type != "Neg":
                return self.no_match(node, "input is not produced by Neg")
            return self.result([previous, node], insert_at=node)

        def apply(self, graph, nodes):
            previous, node = nodes
            return [
                oh.make_node(
                    "Identity", [previous.input[0]], list(node.output)
                )
            ]

    builder = GraphBuilder(model)
    builder.register_pattern(NegNegPattern())
    graph = GraphGraph(builder)
    rewrites = graph.optimize()

API
+++

.. automodule:: onnx_light.onnx_core.optimization
    :members:
    :imported-members:
