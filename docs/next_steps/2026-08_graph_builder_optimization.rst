.. _l-next-steps-graph-builder-optimization:

Pattern-based optimization in ``GraphBuilder``
==============================================

:Date: 2026-08

**implementation completed**

Objective
+++++++++

``onnx_core`` already builds and validates a graph through
``core::builder::GraphBuilder``. The missing piece is a rewriting
engine that recognizes local subgraphs and replaces them with cheaper
equivalents, in the spirit of the Python
`pattern optimizer
<https://github.com/xadupre/yet-another-onnx-builder/blob/main/yobx/xoptim/graph_builder_optim.py>`_.

The C++ optimizer is implemented directly by ``GraphGraph`` over an existing
``GraphBuilder``. It reuses the builder's shape and type inference, its
constant knowledge, and its existing cleanup passes
(``RemoveIdentityNodes``, ``RemoveUnusedNodes``,
``RemoveDuplicateNodes``) instead of duplicating them.

A pattern must never reuse an existing name: every value it produces is new.
This invariant keeps the successor and predecessor maps valid between two
rewrites of the same iteration.

Progress
++++++++

The initial design and implementation steps are delivered in separate pull
requests:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Pull request
     - Step
     - Result
   * - `PR #4351 <https://github.com/xadupre/onnx-light/pull/4351>`_
     - Initial plan
     - Defined the graph index, pattern API, rewrite loop, cleanup,
       constant-folding, subgraph, and statistics design.
   * - `PR #4369 <https://github.com/xadupre/onnx-light/pull/4369>`_
     - Graph index and value queries
     - Added ``GraphGraph`` over ``GraphBuilder`` with structural, shape, type,
       and constant queries.
   * - `PR #4389 <https://github.com/xadupre/onnx-light/pull/4389>`_
     - Pattern interfaces
     - Added ``PatternOptimization``, ``MatchResult``, the optimizer context,
       and the first ``Cast(Cast(x))`` pattern.
   * - `PR #4392 <https://github.com/xadupre/onnx-light/pull/4392>`_
     - Pattern library and registration
     - Moved concrete patterns to ``onnx_extensions`` and added explicit,
       stable-name registration.
   * - `PR #4394 <https://github.com/xadupre/onnx-light/pull/4394>`_
     - Match/apply loop and cleanup
     - Applies disjoint matches by priority and runs duplicate, identity, and
       unused-node cleanup passes until convergence.
   * - `PR #4412 <https://github.com/xadupre/onnx-light/pull/4412>`_
     - Local rewriting and replay
     - Returns persistent rewrite records from optimization and deterministically
       replays them over a fresh model.
   * - `PR #4416 <https://github.com/xadupre/onnx-light/pull/4416>`_
     - Constant folding
     - Folds all-constant replacement nodes into initializers and stores the
       materialized results in the persistent rewrite records.
   * - `PR #4425 <https://github.com/xadupre/onnx-light/pull/4425>`_
     - Recursive subgraph optimization
     - Optimizes nested subgraphs, replays their rewrites deterministically,
       and aggregates their statistics.
   * - `PR #4427 <https://github.com/xadupre/onnx-light/pull/4427>`_
     - Pattern integration documentation
     - Documents linking, registration, and selection, with a standalone
       custom-pattern example.
   * - `PR #4429 <https://github.com/xadupre/onnx-light/pull/4429>`_
     - First incremental pattern port
     - Ports ``CastPattern`` from ``yobx.xoptim.patterns`` with exact rewrite,
       optimized-graph, rejection, and registration tests, and adds
       source-located no-match diagnostics.
   * - `PR #4432 <https://github.com/xadupre/onnx-light/pull/4432>`_
     - First elementary canonicalization pattern
     - Ports ``CastCastBinaryPattern`` with precision and shared-use guards,
       family-scoped tests, and standard pattern registration.
   * - `PR #4433 <https://github.com/xadupre/onnx-light/pull/4433>`_
     - Cast-operation-Cast canonicalization
     - Ports ``CastOpCastPattern`` with guarded type migration, attribute
       preservation, and shared-output restoration.
   * - `PR #4445 <https://github.com/xadupre/onnx-light/pull/4445>`_
     - Python bindings and custom patterns
     - Exposes optimizer queries, rewrites, reports, replay, and concrete Cast
       patterns; supports global, builder-local, and graph-local Python pattern
       registration with recursive lifetime and exception propagation.
   * - `PR #4455 <https://github.com/xadupre/onnx-light/pull/4455>`_
     - Consecutive-Clip canonicalization
     - Ports ``ClipClipPattern``, merging two Clip nodes with complementary
       minimum and maximum bounds, with C++ and Python selection tests.
   * - `Issue #4462 <https://github.com/xadupre/onnx-light/issues/4462>`_
     - Elementary canonicalization patterns
     - Ports the remaining elementary canonicalization patterns
       (``ConstantToInitializerPattern``, ``ConvBiasNullPattern``,
       ``PadConvPattern``, ``DropoutPattern``, ``IdentityPattern``, and
       ``NotNotPattern``) with per-pattern rewrite and rejection tests,
       registration, and Python bindings.
   * - `Issue #4477 <https://github.com/xadupre/onnx-light/issues/4477>`_
     - Expand/Where/Equal batch start
     - Starts the next batch with ``NotWherePattern``,
      ``UnsqueezeEqualPattern``, and ``WhereAddPattern``, including
      rewrite/rejection tests, registration, and Python bindings.
   * - `PR #4491 <https://github.com/xadupre/onnx-light/pull/4491>`_
     - Redundant and broadcasting Expand
     - Ports ``ExpandPattern``, ``ExpandBroadcastPattern``, and
       ``ExpandSwapPattern`` with rewrite/rejection tests, registration, and
       Python bindings.
   * - `Issue #4494 <https://github.com/xadupre/onnx-light/issues/4494>`_
     - Expand/Unsqueeze reordering and fusion
     - Ports ``SwapExpandUnsqueezePattern`` and
       ``ExpandUnsqueezeExpandPattern`` with rewrite/rejection tests,
       registration, and Python bindings.
   * - `Issue #4503 <https://github.com/xadupre/onnx-light/issues/4503>`_
     - Transpose canonicalization and reordering
     - Ports ``TransposeTransposePattern`` and ``TransposeGatherPattern`` with
       rewrite/rejection tests, registration, and Python bindings.
   * - `Issue #4509 <https://github.com/xadupre/onnx-light/issues/4509>`_
     - Unsqueeze/Squeeze simplification
     - Ports ``UnsqueezeUnsqueezePattern`` and ``SqueezeUnsqueezePattern`` with
       rewrite/rejection tests, registration, and Python bindings.
   * - `Issue #4525 <https://github.com/xadupre/onnx-light/issues/4525>`_
     - Shape-based Transpose and Unsqueeze elimination
     - Ports ``ShapeTransposePattern`` and ``UnsqueezeShapePattern`` with
       rewrite/rejection tests, registration, and Python bindings.
   * - `PR #4532 <https://github.com/xadupre/onnx-light/pull/4532>`_
     - Shape-based Concat/Expand simplification
     - Ports ``ShapeBasedConcatExpandPattern`` with symbolic-shape rewrite and
       rejection tests, registration, and Python bindings.
   * - `PR #4534 <https://github.com/xadupre/onnx-light/pull/4534>`_
     - Shape-based Expand broadcast elimination
     - Ports ``ShapeBasedExpandBroadcastPattern`` and
       ``ShapeBasedExpandBroadcastMatMulPattern`` with optional positional
       match roles, symbolic-shape guards, tests, registration, and bindings.
   * - `PR #4539 <https://github.com/xadupre/onnx-light/pull/4539>`_
     - Final rewrite coordinates
     - Resolves ``MatchResult::insert_at`` while applying each batch, records
       final ``added_nodes_positions``, and replays only from those coordinates.
   * - `PR #4542 <https://github.com/xadupre/onnx-light/pull/4542>`_
     - Remaining Expand rewrites
     - Ports the four remaining shape-based and Reshape-adjacent ``Expand``
       patterns with tests, registration, and Python bindings.
   * - `PR #4551 <https://github.com/xadupre/onnx-light/pull/4551>`_
     - Tensor layout and algebra rewrites
     - Ports 13 Reshape, 6 layout, and 12 algebra/reduction/identity classes,
       including the intentionally inactive upstream placeholder.
   * - `PR #4566 <https://github.com/xadupre/onnx-light/pull/4566>`_
     - MatMul, normalization, and activation rewrites
     - Ports 9 matrix-multiplication and 11 normalization/activation patterns
       with rewrite/rejection tests, registration, and Python bindings.
   * - `PR #4567 <https://github.com/xadupre/onnx-light/pull/4567>`_
     - Rotary embedding and attention rewrites
     - Ports the final 9 rotary, causal-mask, cache, attention-function, and
       grouped-query-attention patterns.

Graph structure on Graph
++++++++++++++++++++++++

The optimizer does not own the graph; it owns an index over the builder nodes,
rebuilt after each iteration. The Python ``_build`` method becomes a
``GraphGraph`` class:

.. code-block:: cpp

    class GraphGraph {
    public:
      // Node producing ``name`` (nullptr for inputs and initializers).
      const NodeProto *NodeBefore(const std::string &name) const;
      // Nodes consuming ``name`` (in insertion order, deduplicated).
      const std::vector<const NodeProto *> &NextNodes(const std::string &name) const;

      // Node-level neighbours in the data-flow graph.
      std::vector<const NodeProto *> Predecessors(const NodeProto &node) const;
      std::vector<const NodeProto *> Successors(const NodeProto &node) const;

      bool IsOutput(const std::string &name) const;
      bool IsUsed(const std::string &name) const;
      bool IsUsedMoreThanOnce(const std::string &name) const;
      bool IsUsedBySubgraph(const std::string &name) const;

      std::size_t Position(const NodeProto &node) const;
    };

Nodes are identified by their address, so the index stores
``const NodeProto *`` and maps them to their position in
``GraphBuilder::Nodes``. This mirrors ``make_idn`` in Python, which
uses ``id(node)``. The index tracks:

* ``predecessors_``: value name -> producing node;
* ``successors_``: value name -> consuming nodes;
* ``outputs_`` and ``output_names_``: declared graph outputs;
* ``used_``: values captured by a nested subgraph from the enclosing scope.

The subgraph scan walks ``If``, ``Loop``, ``Scan`` and ``SequenceMap`` bodies
through ``GraphBuilder::ReferencedSubgraphs`` and marks every value read
from the parent scope as used, so a rewrite never deletes a producer a subgraph
still relies on.

Value queries
+++++++++++++

Patterns need read-only access to shapes, types and constants. The optimizer
draws on the builder's inferred information (``GraphBuilder::HasShape`` /
``GraphBuilder::GetShape``, whose ``SymTensor`` also carries the element type,
and the ``ShapesContext`` returned by ``GraphBuilder::Shapes``) rather than
re-implementing inference, and adds a small constant registry for the values it
folds:

.. code-block:: cpp

    bool HasShape(const std::string &name) const;   // from GraphBuilder::GetShape
    const SymTensor &GetShape(const std::string &name) const;
    bool HasType(const std::string &name) const;
    TensorType GetType(const std::string &name) const;

    bool IsConstant(const std::string &name) const;
    bool IsConstantScalar(const std::string &name, double value, bool broadcast) const;
    const TensorProto *GetComputedConstant(const std::string &name) const;

``IsConstantScalar`` reproduces the broadcast rules of the Python helper: a
value is scalar when its shape is ``()`` or ``(1,)``, and, with ``broadcast``,
when every dimension is ``1``. Constant folding results are cached by value
name because a name is assigned once and never reused.

Pattern API
+++++++++++

A pattern is a stateless matcher and rewriter. The two Python classes
``PatternOptimization`` and ``MatchResult`` become:

.. code-block:: cpp

    struct MatchResult {
      const PatternOptimization *pattern;
      std::vector<const NodeProto *> nodes;   // matched nodes (may contain gaps)
      const NodeProto *insert_at = nullptr;   // optional insertion anchor
    };

    class PatternOptimization {
    public:
      explicit PatternOptimization(int priority = 1) : priority(priority) {}
      virtual ~PatternOptimization() = default;

      // Fast pre-filter: operator types this pattern can start from.
      virtual std::set<std::string> FastOpType() const { return {}; }

      // Returns the match rooted at a candidate node (empty when none).
      virtual MatchResult
      Match(GraphGraph &graph,
            const NodeProto &candidate) const = 0;

      // Produces the replacement nodes for one match.
      virtual utils::RepeatedProtoField<NodeProto>
      Apply(GraphGraph &graph,
            const std::vector<const NodeProto *> &nodes) const = 0;

      int priority;
    };

``FastOpType`` lets the driver restrict a pattern to the nodes of one operator
type, exactly like ``fast_op_type`` gates ``subset_nodes`` in Python.

No-match diagnostics and contract errors
+++++++++++++++++++++++++++++++++++++++++

A candidate that does not satisfy a pattern is normal control flow, not an
error. Matchers return ``NoMatch(candidate, reason)``; this keeps
``MatchResult::pattern`` null while ``std::source_location`` captures the C++
file and line of the rejected condition. The optional ``OptimizationReport``
aggregates these diagnostics by pattern, location, and reason, so enabling a
report explains why candidates were rejected without printing during normal
optimization.

``Apply`` has a different contract: the optimizer calls it only with nodes
returned by a successful ``Match``. Invalid nodes passed directly to ``Apply``
therefore indicate a pattern implementation or API misuse and raise
``BuilderError``. ``BuilderError`` also captures its call-site file and line
and includes them in ``what()``, making such exceptional failures actionable.
This distinction mirrors the Python pattern API: ``none(node, line, reason)``
reports an ordinary failed match, while ``apply`` assumes that matching has
already succeeded.

Pattern library and registration
++++++++++++++++++++++++++++++++

The optimization engine and generic pattern interfaces belong to
``onnx_core``. Concrete ONNX rewrite patterns do not: they live under
``onnx_extensions/patterns`` and are built as a separate
``lib_onnx_patterns`` library. This follows the existing shape-inference
split:

* ``onnx_core`` owns the optimizer, ``PatternOptimization``, ``MatchResult``,
  and the pattern registry;
* ``onnx_extensions/patterns`` owns implementations such as
  ``CastCastPattern`` and depends on ``onnx_core``;
* ``onnx_core`` never depends on ``lib_onnx_patterns``.

The pattern currently introduced under ``onnx_core/builder/patterns`` is moved
to ``onnx_extensions/patterns`` as part of this separation.

As with ``onnx_shapes::RegisterShapeFunctions``, registration is explicit so a
static archive cannot discard an otherwise unreferenced translation unit:

.. code-block:: cpp

    namespace core::builder {

    using PatternFactory =
        std::function<std::unique_ptr<PatternOptimization>()>;

    void RegisterPattern(
        const std::string& name,
        PatternFactory factory);

    std::vector<std::unique_ptr<PatternOptimization>>
    CreateRegisteredPatterns();

    }  // namespace core::builder

    namespace onnx_patterns {

    void RegisterPatterns();

    }  // namespace onnx_patterns

``onnx_patterns::RegisterPatterns`` registers every built-in ONNX pattern once
and is idempotent. Entry points that want the standard pattern set call it
before constructing the optimizer; applications linking only ``onnx_core`` may
instead register their own patterns. Pattern names are stable diagnostic and
selection identifiers. Registering two different factories under the same name
is rejected rather than silently changing optimization behavior.

The public headers, CMake target, explicit registration requirement, pattern
selection, and an example custom pattern must be documented alongside the C++
API. Tests cover an ``onnx_core`` optimizer with no extension library, explicit
built-in registration, duplicate registration, and a custom application
pattern.

Replacement nodes use ``utils::RepeatedProtoField<NodeProto>`` because they
own protobuf messages. Moving that container transfers its pointer-backed
storage without copying each ``NodeProto`` and matches the node storage used
by ``GraphProto``. The pointer vectors above remain appropriate for
non-owning candidate and match views.

Match and apply loop
+++++++++++++++++++++

The iteration keeps the two-phase structure of the Python ``optimize`` method,
so that several disjoint rewrites can be applied in a single pass while the
local structure each match relied on stays intact:

.. code-block:: text

    build the graph index

    for every pattern (in priority order):
        for every candidate node:
            for every match the pattern yields:
                skip it if any node is already marked or must not be removed
                otherwise mark all its nodes and record the match

    for every recorded match:
        replacement = pattern.Apply(match)
        move replacement into the rebuilt node field at the first matched position

A match is skipped when one of its nodes is already claimed by an earlier match
of the same iteration (the ``marked`` set), or when a ``DoNotRemove`` predicate
protects it (the port of the Python ``do_not_remove`` guard, driven by node
name markers). Application rebuilds the builder node field in order. It moves
retained node pointers and replacement fields into a new
``utils::RepeatedProtoField<NodeProto>`` before replacing the old field, so no
``NodeProto`` needs a deep copy.

Rewrites as data: ``LocalRewriting`` and replay
+++++++++++++++++++++++++++++++++++++++++++++++

A ``MatchResult`` only describes a match in terms of the transient
``const NodeProto *`` pointers of one ``GraphGraph``; those pointers become
invalid as soon as the builder node field is rebuilt. To make rewrites
reusable, ``Optimize`` returns self-contained records of the applied rewrites
instead of only mutating the builder in place:

.. code-block:: cpp

    std::vector<LocalRewriting> GraphGraph::Optimize(...);

A ``LocalRewriting`` no longer refers to live node pointers: it records the
positions selected by the match and owns the nodes and initializers it adds,
the exact resulting positions of added nodes, and the initializer positions it
removes. Match positions are relative to the graph at the start of the recorded
rewrite batch, while added-node positions are relative to the graph after that
batch. Neither refers to the original model because earlier batches may create
or remove nodes. It also shares ownership of the stateless pattern or cleanup
operation that produced it, so callers can inspect that operation after
``GraphGraph`` is destroyed. Its stable name remains available through
``pattern->Name()`` for logging or serialization without storing a redundant
copy:

.. code-block:: cpp

    struct LocalRewriting {
      std::shared_ptr<const PatternOptimization> pattern;
      // Positions in the graph at the start of this iteration. They cannot be
      // positions in the original model because prior rewrites may add nodes.
      std::vector<std::size_t> matched_nodes;
      utils::RepeatedProtoField<NodeProto> added_nodes;   // replacement nodes
      std::vector<std::size_t> added_nodes_positions;     // positions after the batch
      utils::RepeatedProtoField<TensorProto> added_initializers;
      std::vector<std::size_t> added_initializer_positions;
      std::vector<std::size_t> removed_initializers;
      std::vector<std::pair<std::string, std::string>> value_renames;
      std::size_t iteration = 0;                  // ordered rewrite batch

      std::string ToString() const;
    };

``LocalRewriting``, ``MatchResult`` and ``PatternOptimization`` expose
``ToString()`` summaries for diagnostics without requiring callers to inspect
protobuf fields or transient node pointers.

Because a pattern never reuses an existing name, a ``LocalRewriting`` is fully
determined by the names it consumes and the new names it produces, so it does
not depend on the order in which other rewrites are applied within the same
iteration.

Given a ``ModelProto`` and an ordered list of ``LocalRewriting``, the final
optimized graph can be reconstructed by replaying the rewrites, without
re-running the matching phase:

.. code-block:: cpp

    GraphProto Replay(const ModelProto &model,
                      const std::vector<LocalRewriting> &rewrites);

Replay groups consecutive records from the same rewrite batch, drops their
matched nodes and removed initializers, and places every added node at its
recorded ``added_nodes_positions`` in one rebuild. ``MatchResult::insert_at``
remains the pattern's match-time insertion request, but ``GraphGraph`` resolves
it while applying the batch. The final ``LocalRewriting`` records do not contain
``insert_at``.
``added_nodes_positions`` has exactly one entry per ``added_nodes`` value;
positions are unique and in range for the graph after the batch. Cleanup passes
are records in the sequence too; replay does not rerun them. This preserves
simultaneous disjoint rewrites and gives a cheap, deterministic way to cache and
reproduce an optimization, to audit exactly which rewrites fired, and to apply
a captured sequence to a fresh ``ModelProto`` without paying the cost of
matching or cleanup again.

Cleanup and convergence
++++++++++++++++++++++++

After the matches are applied, each iteration runs the builder cleanup passes
that already exist in C++:

* ``RemoveDuplicateNodes`` for common subexpressions;
* ``RemoveIdentityNodes`` for the ``Identity`` nodes patterns introduce
  to avoid duplicating constants;
* ``RemoveUnusedNodes`` for the values a rewrite made dead.
* ``RemoveDuplicateInitializers`` for constants with identical contents.

Every effective cleanup pass appends a ``LocalRewriting`` under its stable
operation name. Since duplicate and identity removal may rewire many consumers,
the cleanup record replaces the complete node field before and after that pass.
Initializer deduplication similarly records the removed initializer positions
and the surviving initializer field. Identity, node deduplication and
initializer deduplication also persist their value renames so replay updates
consumers and values captured by nested subgraphs. These records form separate
ordered batches, so the returned sequence alone reconstructs the final graph.

If any pattern matched, or any cleanup pass removed a node, the loop continues.
When no pattern matches at the current priority, the driver raises the priority
threshold; it stops once the highest priority yields nothing. The default
``max_iter`` is ``max(node_count, 10) * priority_count``, matching the Python
bound.

Constant folding
++++++++++++++++

When ``Apply`` creates a node whose inputs are all materialized constants, the
optimizer folds that node through the builder's constant-folding machinery.
Only nodes introduced by the current rewrite batch are eligible, so unrelated
constant branches are not changed as a side effect. Folded outputs become
initializers and are stored in ``LocalRewriting::added_initializers`` while the
folded nodes are removed from ``LocalRewriting::added_nodes``. A later pattern
can therefore read the value through ``GetComputedConstant``, and replay does
not need to execute runtime kernels again.

Subgraphs
+++++++++

Optimization is recursive. Before optimizing the main graph, each control-flow
node's subgraphs are optimized in place, seeded with the values visible from the
enclosing scope. The subgraph builders are already owned by ``GraphBuilder``
(``GraphBuilder::Subgraphs``), so the optimizer runs on each of them
with an inherited context of input, initializer and preceding node names.
Each rewrite stores the stable nested-builder path from the root graph. Replay
uses that path to route an ordered rewrite batch to the same subgraph rather
than applying every record to the root builder.

Statistics
++++++++++

Every match, apply and cleanup step appends a record (pattern name, iteration,
instances, elapsed time). The records are returned from ``Optimize`` so callers
can profile which patterns fire and how long each phase takes, as the Python
``statistics`` list does.

The root report aggregates iteration, rewrite and per-pattern counters across
the complete graph hierarchy. It also exposes a deterministic per-subgraph list
containing the graph path, elapsed wall-clock time and recursive activity.
Root phase timings exclude subgraph work; ``subgraph_optimization_time_ns``
accounts for that work separately so ``TotalTimeNs`` does not double-count it.

Logging and phase timing
++++++++++++++++++++++++

At the very end of ``Optimize``, the collected records are summarized into a
report the user can print. The report breaks the total time down per phase, so
a caller can tell how long was spent:

* matching candidate nodes against the registered patterns;
* rewriting, that is applying the matches and rebuilding the node field;
* removing dead-end branches and other cleanup
  (``RemoveDuplicateNodes``, ``RemoveIdentityNodes``, ``RemoveUnusedNodes``);
* constant folding of all-constant rewrites;
* optimizing subgraphs of control-flow nodes.

Each ``LocalRewriting`` also carries the pattern name and its own match/apply
durations, so the summary can attribute time both per phase and per pattern.
The report is opt-in and populated through an optional output parameter,
keeping aggregation and formatting out of the default path:

.. code-block:: cpp

    OptimizationReport report;
    std::vector<LocalRewriting> rewrites = graph.Optimize(-1, &report);
    std::cout << report.ToString() << std::endl;

Python bindings and custom patterns
+++++++++++++++++++++++++++++++++++

The optimizer must also be usable and extensible from Python. The bindings
preserve the core/extension boundary:

* ``_onnxpycore.builder`` exposes ``GraphGraph``, ``PatternOptimization``,
  ``MatchResult``, ``LocalRewriting``, ``OptimizationReport``, and the graph
  structure, shape, type, and constant queries needed by a matcher;
* a ``PatternOptimization`` nanobind trampoline forwards ``fast_op_type``,
  ``match``, and ``apply`` to Python, reacquiring the GIL for every callback;
* ``_onnxpypatterns`` links ``lib_onnx_patterns`` and exposes the concrete ONNX
  classes such as ``CastPattern`` without making ``_onnxpycore`` depend on the
  extension library;
* ``GraphGraph`` accepts concrete C++ patterns and Python subclasses, while
  ``standard_patterns`` resolves registered names before construction;
  optimization always runs through ``GraphGraph.optimize`` and the optimized
  model is produced by the associated ``GraphBuilder``.
* standard patterns are registered globally at import; callers may register or
  replace a pattern globally, on one ``GraphBuilder``, or in one
  ``GraphGraph`` constructor. Registries are merged by stable pattern name,
  with the most local registration taking precedence.

A custom Python pattern follows the same contract as a C++ pattern:

.. code-block:: python

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
                make_node("Identity", [previous.input[0]], list(node.output))
            ]

    builder = GraphBuilder(model)
    optimizer = GraphGraph(
        builder,
        patterns=[CastPattern(), NegNegPattern()],
    )
    rewrites, report = optimizer.optimize(report=True)

``GraphGraph`` and matched ``NodeProto`` objects are borrowed views valid only
while the owning builder and callback are alive. Replacement nodes are copied
back into C++ before the callback returns. The optimizer retains every Python
pattern for the full optimization, including recursive subgraph passes, and
propagates Python exceptions without converting them into a successful
no-match result.

Python-defined patterns may be registered globally, on a ``GraphBuilder``, or
supplied explicitly to ``GraphGraph``. The Python global registry owns its
callbacks and is destroyed with the module, avoiding C++ static references
past interpreter shutdown. The bindings expose registered C++ pattern names
and selection so callers can combine the standard library with custom
patterns.

Tests cover a Python-only rewrite, positive and rejected matches, exact replay,
recursive subgraphs, callback exception propagation, pattern lifetime after
garbage collection, registered C++ pattern selection, and mixed C++/Python
pattern ordering by priority.

Implementation order
++++++++++++++++++++

1. Add ``GraphGraph`` and the value queries over ``GraphBuilder``
   (`PR #4369 <https://github.com/xadupre/onnx-light/pull/4369>`_).
2. Add the ``PatternOptimization`` / ``MatchResult`` interfaces and one trivial
   pattern, ``Cast(Cast(x))`` collapsing
   (`PR #4389 <https://github.com/xadupre/onnx-light/pull/4389>`_).
3. Create ``lib_onnx_patterns`` under ``onnx_extensions/patterns``, move the
   concrete pattern out of ``onnx_core``, and add the explicit core registry
   plus ``onnx_patterns::RegisterPatterns``
   (`PR #4392 <https://github.com/xadupre/onnx-light/pull/4392>`_).
4. Implement the match/apply loop and wire in the existing cleanup passes
   (`PR #4394 <https://github.com/xadupre/onnx-light/pull/4394>`_).
5. Refactor ``Optimize`` to return the applied ``LocalRewriting`` records and
   add ``Replay`` so a captured list of rewrites reconstructs the final graph
   from a ``ModelProto``
   (`PR #4412 <https://github.com/xadupre/onnx-light/pull/4412>`_).
6. Add logging with per-phase timing (match, rewrite, dead-branch removal,
   constant folding, subgraph optimization) reported at the end of ``Optimize``
   (`PR #4414 <https://github.com/xadupre/onnx-light/pull/4414>`_).
7. Add constant folding of all-constant rewrites
   (`PR #4416 <https://github.com/xadupre/onnx-light/pull/4416>`_).
8. Extend to subgraphs and add the statistics output
   (`PR #4425 <https://github.com/xadupre/onnx-light/pull/4425>`_).
9. Document the core/extension boundary, registration and selection APIs,
   linking requirements, and a custom-pattern example
   (`PR #4427 <https://github.com/xadupre/onnx-light/pull/4427>`_).
10. Add Python bindings for the optimizer classes and reports, expose the
    standard ONNX pattern classes through ``_onnxpypatterns``, and support
    globally or locally registered Python-defined ``PatternOptimization``
    subclasses (`PR #4445
    <https://github.com/xadupre/onnx-light/pull/4445>`_).
11. Port the pattern library incrementally, one pattern per commit and several
    related commits per pull request. Every pattern has a C++ test that checks
    the rewritten graph against the expected one. The first port,
    ``CastPattern``, is
    `PR #4429 <https://github.com/xadupre/onnx-light/pull/4429>`_; it also ports
    Python's source-located no-match diagnostics before further patterns are
    added. ``CastCastBinaryPattern`` starts the elementary canonicalization
    batch in `PR #4432 <https://github.com/xadupre/onnx-light/pull/4432>`_;
    ``CastOpCastPattern`` follows in
    `PR #4433 <https://github.com/xadupre/onnx-light/pull/4433>`_, and
    ``ClipClipPattern`` continues it in
    `PR #4455 <https://github.com/xadupre/onnx-light/pull/4455>`_.
12. Resolve every ``MatchResult::insert_at`` while applying its rewrite batch,
    record exact ``added_nodes_positions`` in the returned ``LocalRewriting``,
    and make ``Replay`` use only those final coordinates; test multiple
    disjoint replacements and multiple nodes added by one pattern
    (`PR #4539 <https://github.com/xadupre/onnx-light/pull/4539>`_).

Pattern batches
+++++++++++++++

The implementation is split by functional family; the root directory only
contains registration and dispatch:

.. code-block:: text

    onnx_extensions/patterns/
    ├── dispatch_table.{h,cc}
    ├── canonicalization/   # elementary local rewrites, including Cast
    ├── collections/        # Concat, Gather, Split, Slice, Sequence
    ├── expand/             # Expand, Where, Equal and broadcasting
    ├── reshape/            # Reshape canonicalization
    ├── layout/             # Squeeze, Unsqueeze and Transpose
    ├── algebra/            # generic arithmetic and reductions
    ├── matmul/             # MatMul and Gemm fusions
    ├── normalization/      # normalization and activation fusions
    └── attention/          # rotary embedding and attention functions

Tests added by each batch mirror these family directories under
``unittests/cc/onnx_extensions/patterns``. Helpers used by one family stay
beside its patterns; only genuinely cross-family graph queries belong in
``onnx_core``. This prevents both a flat pattern directory and a catch-all
``detail`` directory.

The upstream default list currently contains 104 enabled patterns.
``CastCastPattern``, ``CastPattern``, ``CastCastBinaryPattern``,
``CastOpCastPattern``, ``ClipClipPattern``, ``ConstantToInitializerPattern``,
``ConvBiasNullPattern``, ``PadConvPattern``, ``DropoutPattern``,
``IdentityPattern``, ``NotNotPattern``, ``NotWherePattern``,
``UnsqueezeEqualPattern``, ``WhereAddPattern``, ``ExpandPattern``,
``ExpandBroadcastPattern``, ``ExpandSwapPattern``,
``SwapExpandUnsqueezePattern``, ``ExpandUnsqueezeExpandPattern``,
``TransposeTransposePattern``, ``TransposeGatherPattern``,
``UnsqueezeUnsqueezePattern``, ``SqueezeUnsqueezePattern``,
``ShapeTransposePattern``, ``UnsqueezeShapePattern``,
``ConcatEmptyPattern``, ``ConcatGatherPattern``,
``ConcatTwiceUnaryPattern``, ``GatherConcatPattern``,
``GatherGatherPattern``, ``GathersSplitPattern``, ``GatherShapePattern``,
``SequenceConstructAtPattern``, ``SplitToSequenceSequenceAtPattern``,
``SliceSlicePattern``, ``SlicesSplitPattern``, ``SplitConcatPattern``, and
``ShapeBasedConcatExpandPattern``, ``ShapeBasedExpandBroadcastPattern``, and
``ShapeBasedExpandBroadcastMatMulPattern`` are already covered. PR #4542 adds
the last four ``Expand`` patterns, and the tensor-layout/algebra batch adds 30
active rewrites plus the upstream-compatible
``ShapeBasedShapeShapeAddPattern`` placeholder. PR #4566 adds the 20
matrix-multiplication and normalization/activation patterns, and PR #4567
adds the final 9 attention patterns. No pattern remains to migrate from the
upstream default pattern set.
They are grouped into cohesive pull requests below rather than
one pull request per pattern. Within a batch, each pattern remains a separate
commit with its exact positive rewrite test and at least one rejection test;
this keeps reviews and ``git bisect`` useful without creating 100 pull
requests. Commented-out, non-default upstream patterns are outside this plan.

#. **Elementary canonicalization (done).**
   ``ConstantToInitializerPattern``, ``ConvBiasNullPattern``,
   ``PadConvPattern``, ``DropoutPattern``, ``IdentityPattern``, and
   ``NotNotPattern`` are ported and registered.
#. **Concat, gather, split, slice, and sequence (done).**
   ``ConcatEmptyPattern``, ``ConcatGatherPattern``,
   ``ConcatTwiceUnaryPattern``, ``GatherConcatPattern``,
   ``GatherGatherPattern``, ``GathersSplitPattern``, ``GatherShapePattern``,
   ``SequenceConstructAtPattern``, ``SplitToSequenceSequenceAtPattern``,
   ``SliceSlicePattern``, ``SlicesSplitPattern``, and ``SplitConcatPattern``.
#. **Expand, where, and equal (done in PR #4542).**
   ``ShapeBasedExpandCastWhereSwapPattern``, ``ShapeBasedExpandSwapPattern``,
   ``ShapeBasedStaticExpandPattern``, and ``SwapExpandReshapePattern``.
   ``ExpandPattern``, ``ExpandBroadcastPattern``, ``ExpandSwapPattern``,
   ``SwapExpandUnsqueezePattern``, and ``ExpandUnsqueezeExpandPattern`` are
   ported and registered, together with ``ShapeBasedConcatExpandPattern``,
   ``ShapeBasedExpandBroadcastPattern``, and
   ``ShapeBasedExpandBroadcastMatMulPattern``.
#. **Reshape canonicalization (done in PR #4551).**
   ``ConcatReshapePattern``, ``ReshapePattern``, ``ReduceReshapePattern``,
   ``Reshape2Of3Pattern``, ``ReshapeReshapeBinaryPattern``,
   ``ReshapeReshapePattern``, ``ReshapeSqueezePattern``,
   ``ShapeBasedEditDistanceReshapePattern``,
   ``ShapeBasedReshapeIsSqueezePattern``, ``ShapedBasedReshapePattern``,
   ``StaticConcatReshapePattern``, ``UnsqueezeOrSqueezeReshapePattern``, and
   ``UnsqueezeReshapePattern``.
#. **Squeeze, unsqueeze, and transpose (done in PR #4551).**
   ``MulUnsqueezeUnsqueezePattern``, ``SqueezeAddPattern``,
   ``SqueezeBinaryUnsqueezePattern``, ``SqueezeUnsqueezePattern``,
   ``UnsqueezeUnsqueezePattern``, ``SwapUnsqueezeTransposePattern``,
   ``TransposeEqualReshapePattern``, ``TransposeGatherPattern``,
   ``TransposeReshapeTransposePattern``, ``TransposeTransposePattern``,
   ``ShapeTransposePattern``, and ``UnsqueezeShapePattern``.
   ``SqueezeUnsqueezePattern``, ``UnsqueezeUnsqueezePattern``,
   ``TransposeGatherPattern``, ``TransposeTransposePattern``,
   ``ShapeTransposePattern``, and ``UnsqueezeShapePattern`` are ported and
   registered.
#. **Generic algebra, reduction, and graph identities (done in PR #4551).**
   ``MulMulMulScalarPattern``, ``SwitchOrderBinaryPattern``,
   ``SwapRangeAddScalarPattern``, ``ReduceArgTopKPattern``,
   ``ReduceSumNormalizePattern``, ``Sub1MulPattern``, ``SwapUnaryPattern``,
   ``SameChildrenPattern``, ``SameChildrenFromInputPattern``,
   ``ShapeBasedIdentityPattern``, ``ShapeBasedSameChildrenPattern``, and
   ``ShapeBasedShapeShapeAddPattern``.
   The last class deliberately preserves the upstream placeholder contract:
   it is exposed and registered but never matches because upstream defines no
   proven rewrite.
#. **Matrix multiplication and linear algebra (done in PR #4566).**
   ``GemmTransposePattern``, ``MatMulAddPattern``,
   ``MatMulReshape2Of3Pattern``, ``MulMulMatMulPattern``,
   ``ReshapeMatMulReshapePattern``, ``ShapeBasedMatMulToMulPattern``,
   ``SwitchReshapeActivationPattern``, ``TransposeMatMulPattern``, and
   ``TransposeReshapeMatMulPattern``.
#. **Normalization and activations (done in PR #4566).**
   ``BatchNormalizationPattern``, ``BatchNormalizationTrainingPattern``,
   ``CastLayerNormalizationCastPattern``, ``LayerNormalizationPattern``,
   ``LayerNormalizationScalePattern``, ``RMSNormalizationPattern``,
   ``RMSNormalizationMulPattern``, ``GeluPattern``, ``LeakyReluPattern``,
   ``MaxReluPattern``, and ``SoftmaxCrossEntropyLossCastPattern``.
#. **Rotary embedding and attention functions (done in PR #4567).**
   ``RotaryEmbeddingPattern``, ``RotaryConcatPartPattern``,
   ``FunctionCausalMaskPattern``, ``FunctionCausalMaskMulAddPattern``,
   ``FunctionCosSinCachePattern``, ``FunctionHalfRotaryEmbeddingPattern``,
   ``FunctionAttentionPattern``, ``FunctionAttentionGQAPattern``, and
   ``AttentionGQAPattern``.

The batches are ordered from local, low-dependency rewrites toward
shape-sensitive and model-specific fusions. If a batch exposes a missing
generic graph query, that query is added and tested in the same pull request;
model-specific shortcuts are not added to the optimizer core.

Pull requests
+++++++++++++

* `PR #4351 <https://github.com/xadupre/onnx-light/pull/4351>`_: initial plan.
* `PR #4369 <https://github.com/xadupre/onnx-light/pull/4369>`_: graph index and value queries.
* `PR #4389 <https://github.com/xadupre/onnx-light/pull/4389>`_: pattern interfaces.
* `PR #4392 <https://github.com/xadupre/onnx-light/pull/4392>`_: pattern extension library.
* `PR #4394 <https://github.com/xadupre/onnx-light/pull/4394>`_: match/apply loop and cleanup.
* `PR #4396 <https://github.com/xadupre/onnx-light/pull/4396>`_: replay and phase-logging design.
* `PR #4414 <https://github.com/xadupre/onnx-light/pull/4414>`_: phase timing and
  optimization report.
* `PR #4416 <https://github.com/xadupre/onnx-light/pull/4416>`_: constant folding
  for replacement nodes.
* `PR #4425 <https://github.com/xadupre/onnx-light/pull/4425>`_: recursive subgraph
  optimization, replay paths, and aggregated statistics.
* `PR #4427 <https://github.com/xadupre/onnx-light/pull/4427>`_: pattern linking,
  registration, selection, and standalone custom-pattern example.
* `PR #4429 <https://github.com/xadupre/onnx-light/pull/4429>`_: first incremental
  library port, ``CastPattern``, plus source-located match diagnostics.
* `PR #4432 <https://github.com/xadupre/onnx-light/pull/4432>`_: first elementary
  canonicalization port, ``CastCastBinaryPattern``.
* `PR #4433 <https://github.com/xadupre/onnx-light/pull/4433>`_:
  ``CastOpCastPattern`` with guarded type migration and shared-output
  preservation.
* `PR #4455 <https://github.com/xadupre/onnx-light/pull/4455>`_:
  ``ClipClipPattern`` merging two Clip nodes with complementary bounds.
* `Issue #4462 <https://github.com/xadupre/onnx-light/issues/4462>`_: remaining
  elementary canonicalization patterns (``ConstantToInitializerPattern``,
  ``ConvBiasNullPattern``, ``PadConvPattern``, ``DropoutPattern``,
  ``IdentityPattern``, and ``NotNotPattern``).
* `Issue #4477 <https://github.com/xadupre/onnx-light/issues/4477>`_: first
  ``Expand/Where/Equal`` rewrites (``NotWherePattern``,
  ``UnsqueezeEqualPattern``, and ``WhereAddPattern``).
* `Issue #4490 <https://github.com/xadupre/onnx-light/issues/4490>`_: core
  ``Expand`` rewrites (``ExpandPattern``, ``ExpandBroadcastPattern``, and
  ``ExpandSwapPattern``).
* `Issue #4494 <https://github.com/xadupre/onnx-light/issues/4494>`_:
  ``Expand``/``Unsqueeze`` reordering and fusion
  (``SwapExpandUnsqueezePattern`` and ``ExpandUnsqueezeExpandPattern``).
* `PR #4532 <https://github.com/xadupre/onnx-light/pull/4532>`_:
  ``ShapeBasedConcatExpandPattern`` simplifying dynamic ``Expand`` target
  shapes when exactly one dimension changes.
* `PR #4534 <https://github.com/xadupre/onnx-light/pull/4534>`_:
  ``ShapeBasedExpandBroadcastPattern`` and
  ``ShapeBasedExpandBroadcastMatMulPattern`` removing unnecessary dynamic
  ``Expand`` nodes before element-wise binary operators and ``MatMul``.
* `PR #4539 <https://github.com/xadupre/onnx-light/pull/4539>`_: final
  ``LocalRewriting`` coordinates and deterministic replay.
* `PR #4542 <https://github.com/xadupre/onnx-light/pull/4542>`_: remaining
  shape-based and Reshape-adjacent ``Expand`` patterns.
* `PR #4551 <https://github.com/xadupre/onnx-light/pull/4551>`_: Reshape,
  remaining layout, algebra, reduction, and graph-identity patterns.
* `PR #4566 <https://github.com/xadupre/onnx-light/pull/4566>`_: matrix
  multiplication, normalization, and activation patterns.
* `PR #4567 <https://github.com/xadupre/onnx-light/pull/4567>`_: rotary
  embedding, causal masks, caches, attention functions, and grouped-query
  attention.
