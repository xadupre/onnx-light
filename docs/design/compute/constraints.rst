.. _l-design-shape-constraints:

Symbolic-dimension constraint mechanism
=======================================

This page describes how ``onnx_shapes`` shape inference records and
resolves *constraints* between symbolic tensor dimensions. The
constraint store lives on :cpp:class:`ShapesContext` (see
``onnx_light/onnx_core/shapes/shapes_context.h``) and is consumed by the
anchor-propagation pass in
``onnx_light/onnx_core/shapes/shape_inference.cc``.

Motivation
----------

Per-operator shape inference names the symbolic dimensions it produces
locally — for instance ``NonZero`` emits an output whose last dimension
is a fresh symbol such as ``NonZero_nz_nnz``. The model author, however,
usually annotates the graph outputs with their own names (``Y: [N, 4]``,
``Z: [2*dnz]``). Without a way to relate the two naming schemes, the same
runtime dimension would appear under two different symbols, and the
inferred ``value_info`` would not match the declared output shapes.

Constraints close that gap. When the merge of an inferred shape with a
user-declared *anchor* shape reveals that two symbolic expressions must
denote the same (or a bounded) value, that relationship is recorded as a
constraint. A later canonicalisation pass then rewrites the inferred
symbols to the user-visible names.

Two kinds of constraints
-------------------------

The context stores two independent families of constraints.

Equality constraints (``a == b``)
    Recorded with :cpp:func:`ShapesContext::AddConstraint`. The pair is
    canonicalised (smaller string first) so that ``(a, b)`` and
    ``(b, a)`` deduplicate, and the trivial self-equality ``a == a`` is
    dropped. :cpp:func:`ShapesContext::HasConstraint`,
    :cpp:func:`ShapesContext::ConstraintsSize` and
    :cpp:func:`ShapesContext::Constraints` give read access. Equality
    constraints typically arise when an output anchor declares
    ``Y: [ANCHOR, 4]`` while node-level inference produced
    ``Y: [N, 4]`` — the merge records ``N == ANCHOR``.

Upper-bound constraints (``lhs <= rhs``)
    Recorded with :cpp:func:`ShapesContext::AddLessEqualConstraint`.
    Here ``lhs`` is a symbolic dimension name and ``rhs`` is an arbitrary
    integer-string or symbolic expression that bounds it from above.
    The ordered pair is *not* canonicalised (direction matters); the
    trivial self-bound and empty operands are dropped.
    :cpp:func:`ShapesContext::HasLessEqualConstraint`,
    :cpp:func:`ShapesContext::LessEqualConstraintsSize` and
    :cpp:func:`ShapesContext::LessEqualConstraints` give read access.
    These bounds are produced by operators whose output dimension is
    data-dependent but provably bounded, for example:

    * ``NonZero(X)`` → ``(rank(X), nnz)`` with ``nnz <= prod(shape(X))``;
    * ``Compress(X, cond, axis=k)`` → output dim ``count <= X.shape[k]``;
    * ``If(...)`` merges two branches; a differing merged dim is bounded
      above by the ``max`` of the two branch expressions.

Both stores are cleared by :cpp:func:`ShapesContext::Clear`.

Deriving leaf-level equalities
------------------------------

An anchor may relate *compound* expressions rather than bare symbols.
For example an output anchor ``Y: [2*dnz]`` merged against the inferred
``Y: [2*NonZero_nz_nnz]`` yields the equality ``2*dnz == 2*NonZero_nz_nnz``.
Renaming only the compound expression would leave bare occurrences of
``NonZero_nz_nnz`` elsewhere untouched.

``AddSymbolicConstraintWithLeafDerivation`` (in ``shape_inference.cc``)
handles this. It records the original equality and then asks
:func:`~onnx_light.onnx_core.expressions.simplify_two_expressions` for
the algebraic difference of the two sides. When that difference reduces
to a coefficient map with exactly two non-zero entries of equal
magnitude and opposite sign (``c*x - c*y``), it additionally records the
implied leaf equality ``x == y`` (here ``dnz == NonZero_nz_nnz``).

Propagating constraints
-----------------------

After node-level inference completes,
``PropagateAnchorConstraintsIntoContext`` turns the recorded equality
constraints into a renaming of the context:

#. If no constraints were recorded, it returns immediately.
#. It collects the set of *preferred* names — the symbols (and the leaf
   tokens of compound expressions) attached to graph inputs, outputs and
   existing ``value_info`` entries. These user-provided names must not be
   renamed away.
#. It builds an undirected adjacency map from the equality constraints
   and calls
   :func:`~onnx_light.onnx_core.expressions.rename_dynamic_dimensions`,
   which picks, for each equivalence class, a canonical preferred name
   and returns a ``{internal name → canonical name}`` mapping.
#. Replacement entries whose key is a compound expression made
   exclusively of leaf tokens that are themselves graph-declared anchor
   symbols are dropped, so already-authoritative expressions such as
   ``b+c`` (both graph inputs) are preserved. The one exception is when
   the replacement *target* is itself a graph-**input** symbol: a compound
   expression such as ``past_seq+seq`` that an equality constraint proves
   equal to the input dimension ``total_seq`` is rewritten to that anchor
   rather than preserved, because the input symbol is the authoritative
   dimension. (Graph *output*-only symbols, by contrast, are mere labels,
   so their computed expression is kept.)
#. Every tensor's ``shape`` and ``value_as_shape`` is rewritten with the
   replacement mapping. The rewrite is repeated to a fixed point (up to a
   few iterations) because a dimension freshly populated during one pass
   may itself need renaming on the next.

   When rewriting a dimension,
   :func:`~onnx_light.onnx_core.expressions.rename_dynamic_expression`
   also substitutes whole compound *subexpressions* that match a
   replacement key — so ``past_seq+seq`` is replaced by ``total_seq`` even
   when it is nested inside a synthesized ``broadcast(past_seq+seq,
   total_seq)`` term — and collapses ``broadcast(x, x)`` to ``x`` once both
   operands become identical.

Because the preferred set includes the user-declared symbols, the
canonicalisation keeps ``X: [N, 4]`` as-is while rewriting an internally
named sibling to the same ``N`` — it never forces a user symbol to become
an internal one.

API reference
-------------

* **C++ API**: :cpp:class:`ShapesContext`
  (:doc:`/api/cpp/onnx_core/shapes/shapes_context`) and
  :doc:`/api/cpp/onnx_core/shapes/shape_inference`.
* **Expression simplifier**: :ref:`l-design-expressions`.
