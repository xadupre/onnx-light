.. _l-design-shape-inference:

Shape inference
===============

This section gathers the design documentation of the ``onnx_optim``
shape-inference engine: the overall algorithm, the symbolic
dimension-expression library it relies on, the constraint mechanism that
reconciles inferred and user-declared dimensions, and the coverage report
that tracks how well the inferred shapes match the backend test
expectations.

.. toctree::
    :maxdepth: 1

    expressions
    constraints
    inference_coverage

Overview
--------

Shape inference computes, for every value in an ONNX graph, the element
type and (possibly symbolic) shape of the tensor it carries, without
running the model. ``onnx-light`` implements it in C++ under
``onnx_light/onnx_optim/shapes/`` and exposes it through the Python entry
point :func:`onnx_light.onnx_optim.shape_inference.infer_shapes_model`.

The engine keeps its working state in a :cpp:class:`ShapesContext`: a
``name → OptimTensor`` map describing every known value, plus the opset
versions, the model-local functions, the registered custom callbacks and
the symbolic-dimension constraints discovered along the way. Each
:cpp:class:`OptimTensor` records an element type, a shape made of
:cpp:class:`OptimDim` entries (each either a concrete ``int`` or a
symbolic expression string), an optional ``value_as_shape`` annotation
and optional integer min/max bounds.

The algorithm
-------------

The top-level driver is ``ShapesContext::ComputeShapeModel`` (called by
``InferShapesModel`` / ``infer_shapes_model``). It runs the following
steps.

1. Register opsets and local functions
    The opset version of every imported domain is recorded, and each
    ``FunctionProto`` declared on the model is registered so that node
    dispatch can expand calls to it.

2. Collect output anchors
    The shapes declared on the graph **outputs** are collected as
    *anchors* — authoritative shape annotations supplied by the model
    author (for example ``Y: [2*dnz]``). When the caller passes
    ``prefill_with_value_info_output=True``, the existing ``value_info``
    entries are collected as anchors as well. A ``value_info`` entry that
    declares only an element type (no shape) is skipped so it cannot
    conflict with a non-scalar inferred shape.

3. Seed and walk the graph
    ``ComputeShapeGraph`` seeds the context from the graph initializers
    (which shadow same-named inputs) and the graph inputs, then calls
    ``ComputeShapes`` over the nodes in topological order. For each node,
    ``ComputeShapeNode`` dispatches to the matching shape function:

    * a **model-local function** call is expanded by recursively running
      shape inference over the function body;
    * a registered **custom callback** for the ``(domain, op_type)`` pair
      is invoked if present;
    * otherwise the built-in **dispatch table** entry for
      ``domain:op_type`` computes the output descriptors.

    Before each dispatch the engine checks that all declared inputs are
    already known and that the outputs are not yet defined, so missing
    inputs or duplicate definitions are reported early.

4. Merge anchors
    ``MergeAnchorsIntoContext`` reconciles each anchor with the inferred
    shape of the same value via ``MergeWithAnchor``. The merge privileges
    anchor information while checking for contradictions: incompatible
    element types, ranks or concrete integer dimensions are conflicts;
    one concrete and one symbolic dim resolves to the concrete value; two
    differing symbolic dims keep the anchor's symbol and **record an
    equality constraint**. Output anchors are merged leniently (a
    conflict skips that anchor rather than aborting the pass), while the
    ``prefill`` value_info anchors are merged strictly.

5. Propagate constraints
    ``PropagateAnchorConstraintsIntoContext`` turns the recorded
    constraints into a renaming of the inferred symbols so that they
    adopt the user-visible names. This step is detailed in
    :ref:`l-design-shape-constraints`.

6. Write back
    ``ApplyInferredShapesToModel`` writes the descriptors stored in the
    context back into the graph: it updates the graph outputs and the
    existing ``value_info`` entries in place and appends a new
    ``value_info`` entry (in deterministic, sorted order) for every other
    inferred tensor that has a known element type. Graph inputs and
    initializers keep their authoritative annotations.

Symbolic dimensions
-------------------

Dynamic dimensions are represented as symbolic expression strings such as
``"2*batch"`` or ``"cache_length + seq_length"``. The
:mod:`onnx_light.onnx_optim.expressions` library parses, simplifies,
evaluates and renames these expressions so that symbolic arithmetic stays
canonical throughout inference. It is described in
:ref:`l-design-expressions`.

Constraints
-----------

When the merge step discovers that two symbolic expressions must be equal
(or that one is bounded above by another), it records a *constraint* on
the context. Constraints let the final renaming pass unify the symbols
emitted by per-operator inference with the names the model author
declared on the graph boundary. The constraint store and its propagation
are described in :ref:`l-design-shape-constraints`.

Coverage
--------

:ref:`l-design-inference-coverage` runs the pipeline against every
backend test case tagged ``"inference"`` and reports, value by value,
whether the computed shape matches the expected one.

API reference
-------------

* **Python API**: :func:`onnx_light.onnx_optim.shape_inference.infer_shapes_model`.
* **C++ API**: :doc:`/api/cpp/onnx_optim/shapes/index`.
