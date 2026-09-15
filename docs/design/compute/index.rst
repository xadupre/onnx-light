.. _l-design-compute:
.. _l-design-shape-inference:

Compute
========

This section gathers the design documentation for analyses consumed by the
runtime and optimizer. It currently focuses on the ``onnx_shapes``
shape-inference engine: the overall algorithm, value-as-shape propagation, the
constraint mechanism that reconciles inferred and user-declared dimensions,
and the coverage report that tracks how well inferred shapes match backend
test expectations.

.. toctree::
    :maxdepth: 1

    value_as_shape
    constraints
    sequences_and_subgraphs
    events
    inference_coverage

Overview
--------

Shape inference computes, for every value in an ONNX graph, the element
type and (possibly symbolic) shape of the tensor it carries, without
running the model. ``onnx-light`` implements the traversal engine (the
:cpp:class:`ShapesContext`, the ``ComputeShape*`` drivers and the
dispatch table) in C++ under ``onnx_light/onnx_core/shapes/`` and the
per-operator shape functions under
``onnx_light/onnx_extensions/shapes/shapes/``; it exposes the whole
pipeline through the Python entry point
:func:`onnx_light.onnx_core.shape_inference.infer_shapes_model`.

The engine keeps its working state in a :cpp:class:`ShapesContext`: a
``name → SymTensor`` map describing every known value, plus the opset
versions, the model-local functions, the registered custom callbacks and
the symbolic-dimension constraints discovered along the way. Each
:cpp:class:`SymTensor` records an element type, a shape made of
:cpp:class:`SymDim` entries (each either a concrete ``int`` or a
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
    inputs or duplicate definitions are reported early. Sequence- and
    map-typed values and the nested graphs of control-flow operators are
    handled here too; see :ref:`l-design-shape-sequences`.

    Shape operator can create a small tensor which can be concatenated,
    modified with addition or any other numerical operator and then
    used to expand or reshape a another tensor. The algorithm also keep
    tracks of such values, whether they numerical or dynamic:
    it can propagate a shape such as ``('N', 1)``.
    That's also one occasion where symbolic expressions are introduced but
    they also appear with operators such as Slice, Pad, Conv, ...

    Before setting any new shape, the symbolic expression is simplified.
    ``(2*H)//H`` becomes ``H``. It handles many cases found in LLMs.

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

This write-back is **not mandatory**. Populating ``value_info`` is a
convenience for callers that want the inferred shapes serialised on the
``ModelProto``; the descriptors themselves live on the
:cpp:class:`ShapesContext` and can be read directly without ever
touching the model. The context exposes ``names`` (every inferred
tensor name), ``has(name)`` and ``get(name)`` (the
:cpp:class:`SymTensor` descriptor, with its element type and shape)
so the full result is accessible programmatically:

.. code-block:: python

    from onnx_light.onnx_core.shape_inference import ShapesContext

    ctx = ShapesContext()
    ctx.compute_shape_model(model)
    for name in ctx.names():
        tensor = ctx.get(name)  # SymTensor: element type + shape
    # ctx.apply_inferred_shapes_to_model(model)  # optional value_info write-back

This basically implements function :func:`onnx_light.onnx_core.shape_inference.infer_shapes_model`
which does not return a context but populates missing ``value_info`` in the original model.
Class :class:`~onnx_light.onnx_core.shape_inference.ShapesContext` provides accessors to access the shapes
without modifying the original model.

Symbolic dimensions
-------------------

Dynamic dimensions are represented as symbolic expression strings such as
``"2*batch"`` or ``"cache_length + seq_length"``. The
:mod:`onnx_light.onnx_core.expressions` library parses, simplifies,
evaluates and renames these expressions so that symbolic arithmetic stays
canonical throughout inference. It is described in
:ref:`l-design-expressions`.

Value-as-shape propagation
--------------------------

Many operators accept a shape tensor whose *runtime values* determine the
output shape. The ``value_as_shape`` annotation on an
:cpp:class:`SymTensor` carries the symbolic content of such tensors so
that consumers like ``Reshape`` or ``Expand`` can infer precise shapes even
when the shape tensor is not a literal initializer. The annotation is
seeded by ``Shape`` and ``Size``, propagated through element-wise
arithmetic and structural operators, and consumed by every shape-tensor
input. This mechanism is described in :ref:`l-design-value-as-shape`.

Sequences, maps and subgraphs
-----------------------------

Besides tensors, the engine tracks sequence- and map-typed values and
infers the nested graphs of control-flow operators (``If`` / ``Loop`` /
``Scan``) and model-local functions in child contexts. These richer
values and nested graphs are described in :ref:`l-design-shape-sequences`.

Constraints
-----------

When the merge step discovers that two symbolic expressions must be equal
(or that one is bounded above by another), it records a *constraint* on
the context. Constraints let the final renaming pass unify the symbols
emitted by per-operator inference with the names the model author
declared on the graph boundary. The constraint store and its propagation
are described in :ref:`l-design-shape-constraints`.

Tracing inference
-----------------

:cpp:class:`ShapesContext` carries an opt-in event log: when
``events_enabled`` is set, every descriptor insertion/replacement, every
node dispatch and every recorded constraint is appended as a
:cpp:class:`ShapeEvent`. Replaying that log is the easiest way to see
which node produced a given dimension or where an inference error
originates. The log is described in :ref:`l-design-shape-events`.

Coverage
--------

:ref:`l-design-inference-coverage` runs the pipeline against every
backend test case tagged ``"inference"`` and reports, value by value,
whether the computed shape matches the expected one.

API reference
-------------

* **Python API**: :func:`onnx_light.onnx_core.shape_inference.infer_shapes_model`.
* **C++ API**: :doc:`/api/cpp/onnx_extensions/shapes/shapes/index`.
