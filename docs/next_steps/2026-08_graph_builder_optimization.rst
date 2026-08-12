.. _l-next-steps-graph-builder-optimization:

Pattern-based optimization in ``GraphBuilder``
==============================================

:Date: 2026-08

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

The initial design and the first two implementation steps were delivered in
separate pull requests:

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
   * - Current
     - Match/apply loop and cleanup
     - Applies disjoint matches by priority and runs duplicate, identity, and
       unused-node cleanup passes until convergence.

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
reusable, ``optimize`` is refactored so that it returns the list of applied
matches instead of only mutating the builder in place:

.. code-block:: cpp

    std::vector<MatchResult>
    GraphBuilderPatternOptimization::Optimize(...);

Each applied ``MatchResult`` is converted into a self-contained
``LocalRewriting`` object. A ``LocalRewriting`` no longer refers to live node
pointers: it records, by value name and node content, which nodes a rewrite
removes and which nodes (and initializers) it adds. It therefore survives the
index rebuild and can be serialized, logged, or stored:

.. code-block:: cpp

    struct LocalRewriting {
      std::string pattern;                        // pattern that produced it
      std::vector<std::string> removed_nodes;     // outputs of removed nodes
      utils::RepeatedProtoField<NodeProto> added_nodes;   // replacement nodes
      std::size_t insert_at = 0;                  // position of the first match
    };

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

Replay applies each ``LocalRewriting`` in order: it drops the removed nodes and
splices the added nodes in at ``insert_at``, then runs the same cleanup passes
as the live loop. This gives a cheap, deterministic way to cache and reproduce
an optimization, to audit exactly which rewrites fired, and to apply a captured
sequence to a fresh ``ModelProto`` without paying the cost of matching again.

Cleanup and convergence
++++++++++++++++++++++++

After the matches are applied, each iteration runs the builder cleanup passes
that already exist in C++:

* ``RemoveDuplicateNodes`` for common subexpressions;
* ``RemoveIdentityNodes`` for the ``Identity`` nodes patterns introduce
  to avoid duplicating constants;
* ``RemoveUnusedNodes`` for the values a rewrite made dead.

If any pattern matched, or any cleanup pass removed a node, the loop continues.
When no pattern matches at the current priority, the driver raises the priority
threshold; it stops once the highest priority yields nothing. The default
``max_iter`` is ``max(node_count, 10) * priority_count``, matching the Python
bound.

Constant folding
++++++++++++++++

When ``Apply`` creates a node whose inputs are all constant, the optimizer
records the new outputs as constants, so a later pattern can read them through
``GetComputedConstant``. This reuses the builder constant machinery; the
optimizer only caches computed values keyed by name.

Subgraphs
+++++++++

Optimization is recursive. Before optimizing the main graph, each control-flow
node's subgraphs are optimized in place, seeded with the values visible from the
enclosing scope. The subgraph builders are already owned by ``GraphBuilder``
(``GraphBuilder::Subgraphs``), so the optimizer runs on each of them
with an inherited context of input, initializer and preceding node names.

Statistics
++++++++++

Every match, apply and cleanup step appends a record (pattern name, iteration,
instances, elapsed time). The records are returned from ``Optimize`` so callers
can profile which patterns fire and how long each phase takes, as the Python
``statistics`` list does.

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
The report is returned alongside the rewrites and is opt-in: it is only
assembled when the caller asks for it, keeping the hot loop free of formatting
work.

Implementation order
++++++++++++++++++++

1. Add ``GraphGraph`` and the value queries over ``GraphBuilder``
   (`PR #4369 <https://github.com/xadupre/onnx-light/pull/4369>`_).
2. Add the ``PatternOptimization`` / ``MatchResult`` interfaces and one trivial
   pattern, ``Cast(Cast(x))`` collapsing
   (`PR #4389 <https://github.com/xadupre/onnx-light/pull/4389>`_).
3. Create ``lib_onnx_patterns`` under ``onnx_extensions/patterns``, move the
   concrete pattern out of ``onnx_core``, and add the explicit core registry
   plus ``onnx_patterns::RegisterPatterns``.
4. Implement the match/apply loop and wire in the existing cleanup passes.
5. Refactor ``Optimize`` to return the list of applied ``MatchResult``, add the
   ``LocalRewriting`` representation, and add ``Replay`` so a captured list of
   rewrites reconstructs the final graph from a ``ModelProto``.
6. Add logging with per-phase timing (match, rewrite, dead-branch removal,
   constant folding, subgraph optimization) reported at the end of ``Optimize``.
7. Add constant folding of all-constant rewrites.
8. Extend to subgraphs and add the statistics output.
9. Document the core/extension boundary, registration and selection APIs,
   linking requirements, and a custom-pattern example.
10. Port the pattern library incrementally, one pattern per change, each with a
    C++ test that checks the rewritten graph against the expected one.
