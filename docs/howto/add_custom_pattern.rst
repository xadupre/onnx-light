.. _l-howto-add-custom-pattern:

:html_theme.sidebar_secondary.remove:

How to add a custom graph-rewriting pattern and set its priority
================================================================

This page shows how to write and register a graph-rewriting **pattern** —
the building block of graph optimization
(:cpp:class:`onnx_light::core::builder::PatternOptimization` /
:class:`onnx_light.onnx_core.optimization.PatternOptimization`) — and how
``priority`` orders it relative to the standard patterns. For a runnable,
end-to-end walk-through (statistics, replay of the recorded rewrites, ...) see
:ref:`l-example-plot-pattern-optimization`.

A pattern implements three things:

* ``fast_op_type`` / ``FastOpType`` — the set of node ``op_type`` values from
  which the matcher may start (an empty set means "every node"); this keeps
  matching fast by skipping irrelevant candidates.
* ``match`` / ``Match`` — inspects one candidate node (and, through
  ``GraphGraph``, its neighbourhood) and returns either a successful match
  (``self.result(...)`` / a :cpp:class:`MatchResult`) or a rejection with a
  diagnostic (``self.no_match(...)`` / :cpp:func:`PatternOptimization::NoMatch`).
* ``apply`` / ``Apply`` — builds the replacement nodes for one accepted match.

.. contents::
    :local:

Write the pattern
------------------

The running example collapses two consecutive ``Neg`` nodes into an
``Identity``.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx.helper as oh
          from onnx_light.onnx_core.optimization import PatternOptimization

          class NegNegPattern(PatternOptimization):
              def __init__(self, priority: int = 1):
                  super().__init__(priority=priority, name="NegNeg")

              def fast_op_type(self):
                  return {"Neg"}

              def match(self, graph, node):
                  previous = graph.node_before(node.input[0])
                  if previous is None or previous.op_type != "Neg":
                      return self.no_match(node, "the input is not produced by Neg")
                  return self.result([previous, node], insert_at=node)

              def apply(self, graph, nodes):
                  previous, node = nodes
                  return [
                      oh.make_node("Identity", [previous.input[0]], list(node.output))
                  ]

      ``self.result(nodes, insert_at=...)`` records the matched nodes (in the
      order ``apply`` expects them) and where the replacement should be
      inserted; ``self.no_match(candidate, reason)`` records a rejection for
      the statistics shown in :ref:`l-example-plot-pattern-optimization`.
      A ``None``/``nullptr`` entry may reserve an optional positional role;
      the optimizer passes it to ``apply`` but ignores it for positioning,
      marking, removal, and replay.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/builder/graph_graph.h"
          #include "onnx_core/builder/pattern_optimization.h"

          namespace builder = onnx_light::core::builder;
          using onnx_light::NodeProto;

          class NegNegPattern final : public builder::PatternOptimization {
          public:
            explicit NegNegPattern(int priority = 1)
                : PatternOptimization(priority, "NegNeg") {}

            std::set<std::string> FastOpType() const override { return {"Neg"}; }

            builder::MatchResult Match(builder::GraphGraph &graph,
                                       const NodeProto &candidate) const override {
              const NodeProto *previous = graph.NodeBefore(candidate.input()[0].value());
              if (previous == nullptr || previous->op_type().value() != "Neg") {
                return NoMatch(candidate, "the input is not produced by Neg");
              }
              return builder::MatchResult{this, {previous, &candidate}, &candidate};
            }

            onnx_light::utils::RepeatedProtoField<NodeProto>
            Apply(builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
              onnx_light::utils::RepeatedProtoField<NodeProto> replacements;
              replacements.push_back(onnx_light::MakeNode(
                  "Identity", {nodes[0]->input()[0].value()}, {nodes[1]->output()[0].value()}));
              return replacements;
            }
          };

      ``NoMatch(candidate, reason)`` captures the call site (file and line)
      automatically through ``std::source_location``, so no explicit location
      needs to be passed.

Select the pattern for one optimizer
-------------------------------------

Passing pattern instances directly to ``GraphGraph`` keeps them local to that
optimizer; nothing is shared globally.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          from onnx_light.onnx_core.optimization import GraphBuilder, GraphGraph

          builder_ = GraphBuilder(model)
          graph = GraphGraph(builder_, [NegNegPattern()])
          rewrites = graph.optimize()

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          std::vector<std::unique_ptr<builder::PatternOptimization>> patterns;
          patterns.push_back(std::make_unique<NegNegPattern>());
          builder::GraphGraph optimizer(graph_builder, std::move(patterns));
          optimizer.Optimize();

Register the pattern globally
-------------------------------

A registered factory makes the pattern available to every new ``GraphGraph``
built without an explicit pattern list, alongside (or overriding, by name) the
standard patterns.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          from onnx_light.onnx_core.optimization import register_pattern, unregister_pattern

          register_pattern(NegNegPattern())
          try:
              graph = GraphGraph(builder_)  # picks up NegNegPattern automatically
              rewrites = graph.optimize()
          finally:
              unregister_pattern("NegNeg")

      A pattern can also be registered on one ``GraphBuilder`` only, which
      overrides a same-named global pattern for optimizers built over that
      builder (``builder_.register_pattern(NegNegPattern())``); see
      "Pattern registration" in
      :doc:`/api/python/onnx_core/optimization`
      for the full precedence rules (global < builder < ``GraphGraph``).

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/builder/pattern_registry.h"

          builder::RegisterPattern("NegNeg", [] { return std::make_unique<NegNegPattern>(); });
          builder::GraphGraph optimizer(graph_builder);  // picks up NegNeg automatically
          optimizer.Optimize();

      ``RegisterPattern`` throws ``PatternRegistrationError`` for an empty or
      already-registered name. ``examples/register_custom_pattern`` is a
      complete, standalone CMake project exercising both the direct and the
      registered selection modes.

Set the priority
-----------------

``priority`` (the constructor's first argument, or ``PatternOptimization``'s
public ``priority`` field) controls **when** a pattern is tried relative to
the others sharing the same optimizer: patterns are grouped by priority and
evaluated in **ascending** order, one priority level at a time, restarting
from the lowest whenever a rewrite happens at a higher one. A negative
priority disables the pattern.

The standard patterns illustrate the ordering: ``CastPattern`` runs at
priority ``0`` so a redundant ``Cast`` is simplified to ``Identity`` before
``CastCastPattern`` and the other priority-``1`` patterns look at its
neighbours. Give a custom pattern a lower priority to run it before the
standard patterns, or a higher one to run it only once they have stabilized:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          # Runs before every priority-1 standard pattern.
          early_pattern = NegNegPattern(priority=0)

          # Runs only once every priority-0/1 pattern stops rewriting.
          late_pattern = NegNegPattern(priority=2)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          NegNegPattern early_pattern(/*priority=*/0);
          NegNegPattern late_pattern(/*priority=*/2);

``priority`` can also be changed after construction (Python:
``pattern.priority = 0``; C++: the public ``priority`` field), as long as it
is set before the pattern is passed to ``GraphGraph``.

See also
--------

* :ref:`l-example-plot-pattern-optimization` - runnable example: statistics,
  the list of applied rewrites, and replaying them from the original model.
* :doc:`/api/python/onnx_core/optimization`
  - Python API reference, including the standard pattern table with their
  priorities.
* :cpp:class:`onnx_light::core::builder::PatternOptimization` - C++ interface
  reference.
* :ref:`l-howto-register-builtins` - companion how-to for registering
  built-in kernels, shape functions and schemas (a different extension
  point, unrelated to patterns).
