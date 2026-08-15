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

.. list-table::
   :header-rows: 1
   :widths: 24 12 24 40

   * - Class / registered name
     - Priority
     - Candidate roots
     - Transformation
   * - :class:`CastPattern` / ``Cast``
     - 0
     - ``Cast``
     - Replaces a type-preserving Cast with Identity.
   * - :class:`CastCastPattern` / ``CastCast``
     - 1
     - ``Cast``
     - Collapses two compatible consecutive Cast nodes into one Cast or
       Identity.
   * - :class:`CastCastBinaryPattern` / ``CastCastBinary``
     - 1
     - Binary arithmetic operators
     - Moves matching floating-point input Cast nodes after the binary
       operation when precision and shared-use guards allow it.
   * - :class:`CastOpCastPattern` / ``CastOpCast``
     - 1
     - Supported unary and binary operators
     - Moves an operation to its result type and removes or relocates its
       surrounding Cast nodes.

The runtime list is available through :func:`standard_pattern_names`.

Custom Python pattern
+++++++++++++++++++++

.. code-block:: python

    from onnx_light.onnx import helper
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
                helper.make_node(
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
