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
`GraphBuilderPatternOptimization
<https://github.com/xadupre/yet-another-onnx-builder/blob/main/yobx/xoptim/graph_builder_optim.py>`_.

The goal is a C++ ``GraphBuilderPatternOptimization`` that runs entirely on an
existing ``GraphBuilder``. It reuses the builder's shape and type inference,
its constant knowledge, and its existing cleanup passes
(``RemoveIdentityNodes``, ``RemoveUnusedNodes``,
``RemoveDuplicateNodes``) instead of duplicating them.

A pattern must never reuse an existing name: every value it produces is new.
This invariant keeps the successor and predecessor maps valid between two
rewrites of the same iteration.

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
      virtual ~PatternOptimization() = default;

      // Fast pre-filter: operator types this pattern can start from.
      virtual std::set<std::string> FastOpType() const { return {}; }

      // Yields every match rooted at a candidate node.
      virtual std::vector<MatchResult>
      Match(GraphBuilderPatternOptimization &opt,
            const std::vector<const NodeProto *> &candidates) const = 0;

      // Produces the replacement nodes for one match.
      virtual utils::RepeatedProtoField<NodeProto>
      Apply(GraphBuilderPatternOptimization &opt,
            const std::vector<const NodeProto *> &nodes) const = 0;

      int priority = 1;
    };

``FastOpType`` lets the driver restrict a pattern to the nodes of one operator
type, exactly like ``fast_op_type`` gates ``subset_nodes`` in Python.

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

Implementation order
++++++++++++++++++++

1. Add ``GraphGraph`` and the value queries over ``GraphBuilder``.
2. Add the ``PatternOptimization`` / ``MatchResult`` interfaces and one trivial
   pattern (for example ``Cast(Cast(x))`` collapsing).
3. Implement the match/apply loop and wire in the existing cleanup passes.
4. Add constant folding of all-constant rewrites.
5. Extend to subgraphs and add the statistics output.
6. Port the pattern library incrementally, one pattern per change, each with a
   C++ test that checks the rewritten graph against the expected one.
