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
    )
    rewrites, report = graph.optimize(report=True)
    optimized_model = builder.to_onnx()

    assert len(rewrites.rewrites) == 1
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

Available registries are merged by the stable :attr:`PatternOptimization.name`;
the ``patterns`` selector then chooses which entries to run:

* **global** patterns (:func:`register_pattern`) are available to every new
  ``GraphGraph``; the standard ONNX patterns are registered globally when
  the module is imported;
* **builder** patterns (``GraphBuilder.register_pattern``) override a
  global pattern for optimizers built over that builder;
* **graph** patterns (``GraphGraph(builder, patterns=[...])``) form an exclusive
  selection. Explicit instances override earlier entries with the same name
  and are retained for that optimizer, including recursive subgraphs.

By default only device-independent patterns are selected. ``patterns=False``
disables patterns without disabling cleanup. A concrete ``Device`` selects
independent patterns and patterns targeting that exact device, and sets
``builder.device`` unless it conflicts with an already-defined target.
A regex selects available names using ``fullmatch``. Regexes and lists can
explicitly select device-specific patterns and leave the builder device unchanged.
See :mod:`onnx_light.onnx_core.optimization` for the selector and migration details.

In C++, ``GraphGraph(builder)`` likewise selects only device-independent
registered patterns. ``CreateRegisteredPatterns(device)`` returns independent
and matching-device patterns enabled by default; the no-argument factory returns
all default-enabled patterns. Explicit-only patterns are available through
``CreateRegisteredPattern(name)``.
When using a filtered factory result in C++, call ``optimizer.SetTargetDevice(device)``
before optimizing to set a common target and check for conflicts in subgraphs.

Patterns can be written in C++ or in Python; both share the
:class:`~onnx_light.onnx_core.optimization.PatternOptimization`
interface, a ``match`` step that returns a
:class:`~onnx_light.onnx_core.optimization.MatchResult` and an ``apply``
step that produces the replacement nodes.

Opt-in repeated layers
-----------------------

``RepeatedLayer`` discovers repeated connected regions from dataflow, not node
names or a prescribed transformer architecture. It is **excluded** from both
the C++ default factory and Python's default/global pattern set. Enable it
explicitly:

.. code-block:: python

    from onnx_light.onnx_core.optimization import GraphGraph, RepeatedLayerPattern

    GraphGraph(builder, patterns=[RepeatedLayerPattern()]).optimize()
    model = builder.to_onnx()

In C++, pass ``onnx_patterns::CreatePattern("RepeatedLayer")`` in the explicit
pattern vector.

Python enumeration excludes opt-in patterns by default: this applies to
``standard_pattern_names()`` and both low-level
``registered_pattern_names()`` bindings (``_onnxpypatterns`` and
``_onnxpycore.builder``). Pass ``include_opt_in=True`` to any of these functions
to list all available patterns. Callers relying on the previous low-level
default must now pass this argument explicitly. Listing never enables a pattern
or changes the process-global registry; ``registered_pattern_names()`` in
``onnx_core.optimization`` lists only that registry's active entries.
The C++ ``RegisteredPatternNames()`` enumeration default remains unchanged.

For example, two corresponding regions become two separate bodies, with their
own parameters passed explicitly::

    Before:  x, w0 → MatMul → Relu → h;  h, w1 → MatMul → Relu → y
    After:   x, w0 → Layer(...) → h;     h, w1 → Layer_1(...) → y

This is encapsulation, not weight tying: ``w0`` and ``w1`` need not be equal.
Each call references a local function stored in ``ModelProto.functions``.
Models can be consumed by ONNX runtimes that support local functions and the
operators used in their bodies.
Function exports and snapshots captured for rewrite undo place callees before
callers, including layers extracted inside an existing local function.

Layer function contract
~~~~~~~~~~~~~~~~~~~~~~~~

The extracted regions are represented by local functions in domain ``ai.rt``
with an ``ai.rt`` opset import of version 1. The first function is named
``Layer``; subsequent functions use an index suffix (``Layer_1``, ``Layer_2``,
...), so each body has a unique definition. Calls bind inputs and outputs
positionally to the function's formal inputs and outputs. Outputs must be
nonempty and unique; omitted positional inputs are not permitted. ``Layer`` is
a function identity, not an operator schema.

The function body is a closed lexical scope at its boundary. Every dependency
on a surrounding graph, including a dependency used only inside a nested graph
attribute, is supplied as an explicit function input. Nested control-flow
graphs may capture values declared within this body; they may not reach past
it. Local function calls and operators retain their original domain/overload
and use the enclosing model's opset imports and function definitions.

The extractor keeps all surrounding initializers and constant producers outside
the bodies and passes their values explicitly. Shared large weights are not
copied into multiple bodies. Initializers already owned by a nested graph stay
there. Original nodes, attributes, internal port wiring and fan-out are retained.
Every internal value needed by an outside consumer, nested capture or graph
output becomes a wrapper output, with the original outer name. Boundary values
are deduplicated in first-use order (inputs) and topological producer/output-port
order (outputs); repeated internal uses retain their original ports. Tensor
descriptors are copied to the body interface.

Shape inference and native execution expand the local function through the
existing function-call machinery. Extraction uses the existing
targeted builder mutation path, updating affected shapes and lifetime/memory
metadata rather than rebuilding the graph for each occurrence.
Incremental updates inspect live builder-owned body fields without exporting
or rebuilding the enclosing graph, including after a nested body mutation.
Body lifetimes include values captured only by nested graph attributes.

Matching and cut policy
~~~~~~~~~~~~~~~~~~~~~~~~

Supported primitive anchors are ``MatMul`` and ``Attention`` in the standard
ONNX domain (``""`` or ``ai.onnx``), ``com.microsoft::GroupQueryAttention`` and
``ai.rt::PagedAttention``, without an overload. A model-local function is also an
anchor when bounded recursive inspection of its exact domain/name/overload
definition reaches one of those identities. A suggestive operator name alone
does not qualify.

Matching preserves operator domain, type, overload, input/output arity, optional
port presence, attributes, internal producer/output-port relationships and
external-value alias classes. Attributes constrain semantics conservatively:
different explicit attributes are not silently considered equivalent, even if
one spells a schema default. Element types and ranks must agree when known;
incompatible known parameter dimensions and final activation dimensions are
rejected. Leading activation dimensions may vary, permitting different batches
and sequence lengths. Symbolic dimension names and parameter tensor values need
not agree. Each body retains its own original nodes and descriptors.
Nested GRAPH attributes are compared with scope-aware structural signatures,
ignoring graph, node and value names while retaining formal roles, captures and
aliasing. Tensor-valued operator attributes constrain their literal contents.

Corresponding regions grow together through predecessor/output and
successor/input port roles. A seed must grow to at least two nodes and have at
least two disjoint occurrences. Shared predecessors, ambiguous fan-out and
overlapping occurrences are not guessed into a match. Side branches may remain
outside a region provided their dependencies and outputs remain explicit.
Already extracted ``Layer`` nodes, and graphs nested within their bodies, are
not candidates.
Stochastic operators, Dropout and opaque custom computation outside the supported
attention identities are excluded;
local-function and nested-graph safety inspection must also succeed. Scheduling
preserves the relative order of remaining stochastic/opaque calls.
One-hop neighbor identities and connecting ports distinguish different roles
of otherwise identical anchors. Seed-family selection and structural
eligibility are separate from cut cost: a low-edge projection fragment must not
preempt a repeated region containing its matching attention/MatMul roles.
Within a family, cuts retain the largest jointly matched anchor-role skeleton
reached by expansion before comparing edge count and tensor size.
Equal-coverage families prefer the smaller repetition period rather than
combining adjacent repeated layers. The cut comparison below chooses boundaries
within an eligible family. Bounded anchor strides provide a fallback for
indistinguishable local neighborhoods.

Every accepted growth cut is compared lexicographically by:

1. Total **crossing dataflow edge count**, including repeated uses by different
   input ports, lexical captures and graph-output uses.
2. Number of crossing edges with unknown tensor size.
3. Sum of known crossing tensor sizes in bytes, with saturating arithmetic.
4. Larger corresponding regions, then original topological node positions.

Thus one fewer edge wins even if it carries a larger tensor. At equal edge
count, known sizes are preferred to unknown sizes; unknown rank, symbolic
dimensions, unavailable element types and unsupported storage types do not
count as zero-byte tensors. Known empty tensors can have zero size.
Sizes are charged per crossing edge, **not** per distinct wrapper argument.
Several cut edges using the same value still require just one positional
wrapper input or output.

The contracted dependency graph includes untouched nodes interleaved with the
regions and dependencies used only in nested graphs. A deterministic
topological schedule preserves these nodes and rejects a contraction that would
introduce a cycle, including paths that leave a region and re-enter it.

This is a bounded, conservative heuristic, **not** a globally optimal cut or a
guarantee to recover a model author's intended transformer block. Exploration
limits can leave valid repeated computations unwrapped. It does not search all
partitions, commute inputs, compare algebraic equivalence or equate different
local-function definitions.

Per detection scan, the implementation indexes operator signatures and value
uses once, caches function-anchor classification by exact operator identity, and
bounds exploration to 32 seed groups and a shared budget of 4096 compatibility
comparisons and extension attempts, including representative and anchor-bucket
comparisons. Region search
uses at most 32 occurrences, 64 nodes per occurrence, 128 ports per operator,
256 neighboring extension choices and anchor strides up to 8. A replacement
interval contains at most 4096 original nodes. Recursive scope/function
inspection is limited to depth 16 and 4096 visited operations; oversized
attribute signatures are rejected at 64 KiB. These limits deliberately trade
recall for bounded work on large models.
Occurrence batches leave at least two remaining matches instead of stranding
a singleton at the 32-occurrence limit (for example, 33 matches use 31 then 2).

Indexing costs scale with graph nodes, edges and inspected attribute payloads.
Each bounded extension checks corresponding region topology and metadata; a
candidate contraction also schedules its bounded interval. The index is local
to a scan, not a cache of mutable graph state. Further successful optimization
iterations may perform additional scans, so this is not a claim that the entire
optimization is a single linear-time pass.

Custom pattern example
----------------------

The following pattern replaces two consecutive ``Neg`` nodes with an
``Identity``. It restricts candidate nodes to ``Neg``, checks the producer of
the candidate's input, and returns the replacement from ``apply`` / ``Apply``.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx.helper as oh
          from onnx_light.onnx_core.optimization import GraphGraph, PatternOptimization

          class NegNegPattern(PatternOptimization):
              def __init__(self):
                  super().__init__(priority=1, name="NegNeg")

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

          graph = GraphGraph(builder, [NegNegPattern()])
          graph.optimize()

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/builder/graph_graph.h"
          #include "onnx_core/builder/pattern_optimization.h"

          namespace builder = onnx_light::core::builder;

          class NegNegPattern final : public builder::PatternOptimization {
          public:
            NegNegPattern() : PatternOptimization(/*priority=*/1, "NegNeg") {}

            std::set<std::string> FastOpType() const override { return {"Neg"}; }

            builder::MatchResult Match(builder::GraphGraph &graph,
                                       const onnx_light::NodeProto &candidate) const override {
              const auto *previous = graph.NodeBefore(candidate.input()[0].value());
              if (previous == nullptr || previous->op_type().value() != "Neg") {
                return NoMatch(candidate, "the input is not produced by Neg");
              }
              return builder::MatchResult{this, {previous, &candidate}, &candidate};
            }

            onnx_light::utils::RepeatedProtoField<onnx_light::NodeProto>
            Apply(builder::GraphGraph &,
                  const std::vector<const onnx_light::NodeProto *> &nodes) const override {
              onnx_light::utils::RepeatedProtoField<onnx_light::NodeProto> replacements;
              replacements.push_back(onnx_light::MakeNode(
                  "Identity", {nodes[0]->input()[0].value()}, {nodes[1]->output()[0].value()}));
              return replacements;
            }
          };

          std::vector<std::unique_ptr<builder::PatternOptimization>> patterns;
          patterns.push_back(std::make_unique<NegNegPattern>());
          builder::GraphGraph graph(graph_builder, std::move(patterns));
          graph.Optimize();

The pattern is local to this optimizer. See
:ref:`l-howto-add-custom-pattern` for global and builder registration,
diagnostics, and priority selection.

API reference
-------------

* **Python API and registered pattern list**:
  :doc:`/api/python/onnx_core/optimization`; the runtime list is available
  through :func:`~onnx_light.onnx_core.optimization.standard_pattern_names`.
* **C++ API**: :doc:`/api/cpp/onnx_core/builder/index`.

Examples
--------

* :ref:`l-example-plot-pattern-optimization` covers optimization statistics.
* :ref:`l-example-plot-pattern-replay` demonstrates deterministic replay from
  captured rewrites.
* :ref:`l-howto-add-custom-pattern` is a Python/C++ how-to on writing a
  custom pattern and choosing its priority.
