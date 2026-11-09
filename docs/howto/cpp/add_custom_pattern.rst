:html_theme.sidebar_secondary.remove:

.. _l-howto-add-custom-pattern-cpp:

How to add a custom graph-rewriting pattern and set its priority
================================================================

This guide covers the C++ interface. For the other language, see
:doc:`../python/add_custom_pattern`.

This page shows how to write and register a graph-rewriting **pattern** —
the building block of graph optimization
(:cpp:class:`onnx_light::core::builder::PatternOptimization` /
:class:`onnx_light.onnx_core.optimization.PatternOptimization`) — and how
``priority`` orders it relative to the standard patterns. For a runnable,
end-to-end walk-through of optimization statistics see
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
-----------------

The running example collapses two consecutive ``Neg`` nodes into an
``Identity``.


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
        if (!candidate.domain().empty() && candidate.domain() != "ai.onnx")
          return NoMatch(candidate, "the candidate is not in the default ONNX domain");
        const NodeProto *previous = graph.NodeBefore(candidate.input()[0].value());
        if (previous == nullptr || (!previous->domain().empty() && previous->domain() != "ai.onnx") ||
            previous->op_type().value() != "Neg") {
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

Primitive matchers check the domain and operator type of every interpreted node.
Model verification and builder registration reject local functions that reuse a
registered operator's domain and name, even with a different ``overload``.
Patterns construct ``MatchResult`` directly; no function-support hook is needed.

Select the pattern for one optimizer
------------------------------------

Passing pattern instances directly to ``GraphGraph`` keeps them local to that
optimizer; nothing is shared globally. The list is exclusive: it does not
implicitly include global or builder registrations.


.. code-block:: cpp

    std::vector<std::unique_ptr<builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<NegNegPattern>());
    builder::GraphGraph optimizer(graph_builder, std::move(patterns));
    optimizer.Optimize();

Register the pattern globally
-----------------------------

A registered pattern becomes available for selection. Device-independent
patterns are included by default, alongside the standard patterns. A
device-specific pattern requires a device selector or an explicit list or regex.


.. code-block:: cpp

    #include "onnx_core/builder/pattern_registry.h"

    builder::RegisterPattern("NegNeg", [] { return std::make_unique<NegNegPattern>(); });
    builder::GraphGraph optimizer(graph_builder);  // picks up NegNeg automatically
    optimizer.Optimize();

``RegisterPattern`` throws ``PatternRegistrationError`` for an empty or
already-registered name. ``examples/register_custom_pattern`` is a
complete, standalone CMake project exercising both the direct and the
registered selection modes.

Declare a target device
-----------------------

Patterns are device-independent by default. A Python subclass can pass
``device=Device.kCPU`` or ``device=Device.kGPU0`` to ``PatternOptimization.__init__``,
with ``Device`` imported from ``onnx_light.onnx_core.shape_inference``.
In C++, pass the corresponding ``core::symbolic::Device`` as the third
``PatternOptimization`` constructor argument.

``GraphGraph(builder, patterns=Device.kCPU)`` includes independent and CPU-specific
patterns and assigns that target to the builder. An already-defined, different
builder device is an error. A list or regex can deliberately select a
device-specific pattern without modifying the target. Declaring a device controls
automatic selection, not whether a caller may explicitly run the pattern.

Set the priority
----------------

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


.. code-block:: cpp

    NegNegPattern early_pattern(/*priority=*/0);
    NegNegPattern late_pattern(/*priority=*/2);

``priority`` can also be changed after construction (Python:
``pattern.priority = 0``; C++: the public ``priority`` field), as long as it
is set before the pattern is passed to ``GraphGraph``.

Higher-rank MatMul + Add: explicit opt-in
---------------------------------------

The A5 assessment in :issue:`5552` keeps
``MatMulAddPattern(priority=2, allow_reshape=false)`` unchanged by default,
including the registered ``"MatMulAdd"`` factory and Python constructor.
The existing ``allow_reshape=true`` mode is **not a generally safe default**:
it introduces shape-dependent operations whose preconditions the matcher does
not fully validate. This decision does not add a matcher or kernels.

The opt-in rewrite flattens the left input, replaces ``MatMul + Add`` with
``Gemm``, and restores the output rank. Tests cover nonempty rank-three and
rank-four inputs, vector and exact-output-shaped biases, both Add input orders,
symbolic batch/output dimensions at opset 15 and later, and an attention graph
containing projection, head-layout changes, batched score MatMul, mask Add and
Softmax. These cases compare the resulting graph and numerical outputs.

Restrictions and reasons not to enable it globally:

* The right input must have rank two. Batched attention score MatMuls remain
  unchanged; this mode only handles the projection part of such a graph.
* Biases must have the output width as their last dimension and must be either
  vectors or exactly output-shaped. Valid Add broadcasts such as ``[1, 1, N]``,
  ``[S, N]`` or ``[B, 1, N]`` for output ``[B, S, N]``, scalar biases, and
  last-dimension broadcasting are conservatively rejected.
* At least one reduction dimension must be a known integer. Symbolic prefixes
  use ``Shape(start=0, end=-1)``; a symbolic full-bias width uses
  ``Shape(start=-1)``. These attributes require opset 15, but the opt-in matcher
  does not enforce that minimum. Do not opt in for those layouts at older opsets.
* All runtime input, weight and bias dimensions must be nonempty. The generated
  ``Reshape`` targets use ``-1`` and the default meaning of ``0`` (copy an input
  dimension), which cannot generally restore empty higher-rank outputs. Static
  or symbolic shapes that can be empty are not safe opt-in inputs, even when
  the matcher accepts them.
* The output type must be supported by the target Gemm schema and backend;
  accepting a MatMul type does not establish Gemm support. Numerical coverage
  here uses float32; structural coverage also exercises double, float16 and
  bfloat16 at opset 28.
* The removed MatMul result must be private: shared consumers, graph outputs
  and subgraph captures prevent fusion.

The standard factory is tested to leave higher-rank graphs unchanged, including
old-opset symbolic and empty-shape cases. Explicit opt-in is appropriate only
when the caller establishes the above preconditions for every runtime shape.
Even for supported cases, the added Reshapes (and Shape/Concat nodes for dynamic
shapes) can interfere with other layout rewrites. Correctness tests are not
performance benchmarks and establish no end-to-end attention speedup.

For a validated workload, select the existing mode locally rather than
re-registering or changing the standard factory:

.. code-block:: cpp

    #include "onnx_extensions/patterns/matmul/matmul_pattern.h"

    std::vector<std::unique_ptr<builder::PatternOptimization>> patterns;
    patterns.push_back(
        std::make_unique<onnx_light::onnx_patterns::MatMulAddPattern>(
            /*priority=*/2, /*allow_reshape=*/true));
    builder::GraphGraph optimizer(graph_builder, std::move(patterns));
    optimizer.Optimize();

See also
--------

* :ref:`l-example-plot-pattern-optimization` - runnable optimization and
  statistics example.
* :ref:`l-example-plot-pattern-replay` - dedicated example that captures
  rewrites and replays them from the original model.
* :doc:`/api/python/onnx_core/optimization`
  - Python API reference, including the standard pattern table with their
  priorities.
* :cpp:class:`onnx_light::core::builder::PatternOptimization` - C++ interface
  reference.
* :ref:`l-howto-register-builtins-cpp` - companion how-to for registering
  built-in kernels, shape functions and schemas (a different extension
  point, unrelated to patterns).
