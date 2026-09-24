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

The available patterns are merged by their stable
:attr:`PatternOptimization.name`. Builder registrations replace global entries
with the same name before selection:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Scope
     - Registration
     - Selection
   * - Global
     - :func:`register_pattern`
     - Makes a pattern available for selection. The standard ONNX patterns
       are registered globally when this module is imported.
   * - Builder
     - :meth:`GraphBuilder.register_pattern`
     - Overrides a global pattern for optimizers built over that builder.
   * - Graph
     - ``GraphGraph(builder, patterns=[...])``
     - Selects only the given names and instances. Explicit instances override
       earlier entries with the same name and are retained for that optimizer,
       including recursive subgraphs.

:func:`clear_registered_patterns` clears the global
registry; :func:`reset_registered_patterns` restores the standard patterns.

Pattern selection
+++++++++++++++++

``GraphGraph(builder, patterns=...)`` accepts one selector:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Selector
     - Selected patterns
   * - ``None`` (default)
     - All available device-independent patterns, even if ``builder.device``
       is already set.
   * - ``False`` or ``[]``
     - None, including builder registrations. The optimizer's ordinary
       cleanup and constant-folding passes still run.
   * - A concrete ``Device``
     - All device-independent patterns plus those targeting that exact device.
       Also sets ``builder.device`` and subgraph targets. A different,
       already-defined device in any of those builders raises ``ValueError``
       before changing any target.
   * - A regex string or compiled ``re.Pattern[str]``
     - Available names accepted by ``fullmatch``, including device-specific
       patterns. Use ``".*MatMul.*"`` to select names containing ``MatMul``.
   * - An iterable of names and/or pattern instances
     - Only those patterns, including device-specific ones. Names are exact,
       not regexes. Standard names may also be instantiated when absent from
       the global registry.

Regexes and explicit lists do not modify ``builder.device`` or automatically
add global or builder patterns. A regex matching nothing selects no patterns;
invalid regexes and invalid selector types raise exceptions. ``True`` is not
a selector, and ``Device.kUndefined`` is not a concrete device: use ``None``
for the default selection.
During recursive optimization, a subgraph with an undefined device inherits
its parent builder's device; explicitly defined subgraph devices are preserved.

.. code-block:: python

    import re
    from onnx_light.onnx_core.shape_inference import Device

    graph = GraphGraph(builder)                         # device-independent defaults
    graph = GraphGraph(builder, patterns=False)         # cleanup only
    graph = GraphGraph(builder, patterns=Device.kCPU)    # defaults + CPU patterns
    graph = GraphGraph(builder, patterns=r".*MatMul.*")  # full name regex
    graph = GraphGraph(builder, patterns=re.compile(r"Cast.*"))
    graph = GraphGraph(builder, patterns=["Cast", "TransposeMatMul"])

A pattern's ``device`` defaults to ``Device.kUndefined``, meaning
device-independent. Custom Python patterns declare a target with
``super().__init__(name="MyCPUFusion", device=Device.kCPU)``; native patterns
use the corresponding third ``PatternOptimization`` constructor argument.
Device equality is exact, not a GPU-family or execution-provider capability
query. Existing standard patterns remain device-independent, including patterns
whose matching heuristics inspect ``builder.device``.

The former ``use_global_patterns`` argument is removed. Replace an explicit
list plus ``use_global_patterns=False`` with just that list, use ``False`` for
no patterns, or pass ``builder.registered_patterns()`` for builder-only patterns.
To combine defaults with explicit patterns, supply the combined list explicitly,
for example ``[*standard_patterns(), custom_pattern]`` (which intentionally
includes every standard pattern, regardless of device).

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
The :ref:`complete pattern catalogue <l-api-pattern-catalog>` in the
:ref:`ByOp catalogue <l-onnx-operators>` adds the C++ documentation link and
the Before/After rewrite graph for every entry.

See :ref:`l-howto-add-custom-pattern` for a Python/C++ how-to on writing a
custom pattern and choosing its priority, and
:ref:`l-example-plot-pattern-optimization` for a runnable example covering
optimization statistics. Replay is demonstrated separately in
:ref:`l-example-plot-pattern-replay`.

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
