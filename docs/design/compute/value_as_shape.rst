.. _l-design-value-as-shape:

Value-as-shape propagation
==========================

Many ONNX operators accept a **shape tensor** as one of their inputs — a
1-D integer tensor whose runtime *values* define the output shape.
Examples include the ``shape`` input of ``Reshape``, the ``shape`` input
of ``Expand``, the ``repeats`` input of ``Tile``, and the ``sizes`` input
of ``Resize``.

Static shape inference cannot read runtime tensor values in general.
However, a common pattern in real models is:

.. code-block:: text

    y   = Shape(x)         # y = [N, C, H, W]  (values = x's symbolic dims)
    new_shape = Concat([y[0:2], const_hw], axis=0)  # [N, C, new_H, new_W]
    out = Reshape(data, new_shape)

Here the ``new_shape`` tensor is not an initializer but its values are
fully determined by the symbolic shape of ``x``.  If shape inference could
track those values, the output shape of ``Reshape`` would be known
exactly.

The ``value_as_shape`` annotation is ``onnx_shapes``'s mechanism to do exactly
that: it is an optional secondary shape stored on an
:cpp:class:`SymTensor` that records the **symbolic content** of the
tensor as an :cpp:class:`SymShape`.  When a consumer like ``Reshape``
sees that its shape input carries a ``value_as_shape``, it uses those
symbolic dimension expressions instead of falling back to a fully generic
output shape.

----

Representation
--------------

:cpp:class:`SymTensor` carries two independent shape structures:

``shape``
    The *storage shape* of the tensor — its rank and the size of each
    dimension.  For the output of ``Shape(x, start=0, end=2)`` applied to
    a rank-4 input, this is ``[2]`` (a 1-D tensor with 2 elements).

``value_as_shape``
    The *content* of the tensor interpreted as shape dimensions.  For the
    same ``Shape`` output, this is ``[x.shape[0], x.shape[1]]`` — the
    two selected symbolic or concrete dimensions of ``x``.

The annotation is optional.  When absent, ``has_value_as_shape`` returns
``False`` and the consumer falls back to symbolic or rank-only output
shapes.

A non-empty ``value_as_shape`` annotation is only meaningful for integer
tensors.  The function :cpp:func:`IsIntegerTensorType` checks whether a
dtype qualifies.

----

Seeding the annotation
----------------------

The annotation is seeded from two sources.

**Initializer literals**
    :cpp:func:`SymTensorFromTensorProto` reads an integer tensor
    initializer and, when its element count is below
    :cpp:var:`kOptimValueAsShapeMaxElements`
    (``kMaxOptimRank + 1``), stores its concrete integer values as a
    ``value_as_shape``.  This handles the common case where the target shape
    is a model constant such as ``[0, 0, -1]``.

**``Shape`` operator**
    :cpp:func:`ComputeShapeShape` lifts the input tensor's selected
    symbolic dimensions directly into the output's ``value_as_shape``.
    The output storage shape is always ``[length]`` where
    ``length = end - start``, but the ``value_as_shape`` carries
    ``[dim_start, dim_start+1, ..., dim_end-1]`` — possibly symbolic
    entries such as ``"batch"`` or ``"2*N"``.

**``Size`` operator**
    :cpp:func:`ComputeShapeSize` stores the total element count of the
    input (as a single-entry ``value_as_shape``) on the scalar output.
    Concrete sizes produce a concrete ``value_as_shape``; symbolic
    shapes produce a symbolic expression.

----

Propagating the annotation
--------------------------

Several operators forward the ``value_as_shape`` annotation through their
outputs when its presence makes the result more precise.

Concatenation and splitting
    ``Concat`` (1-D, axis 0): when every input carries a ``value_as_shape``
    annotation, the outputs' ``value_as_shape`` is the concatenation of
    all input annotations.

    ``Split`` (1-D, axis 0): the inverse of ``Concat`` — each output
    receives the corresponding contiguous slice of the input's
    ``value_as_shape``.

Element-wise arithmetic (``Add``, ``Sub``)
    :cpp:func:`PropagateValueAsShapeArithmetic` applies
    :func:`~onnx_light.onnx_core.expressions.dim_add` or
    :func:`~onnx_light.onnx_core.expressions.dim_sub` element-wise to
    the two operands' ``value_as_shape`` annotations.  Symbolic arithmetic
    is preserved: adding ``["N", 1]`` and ``[1, 1]`` yields
    ``["1+N", 2]``.  When either operand lacks a ``value_as_shape``, the
    annotation is not set on the output.

Indexing and slicing
    ``Gather`` (1-D data, axis 0): when both ``data`` and ``indices``
    carry ``value_as_shape``, the output's ``value_as_shape`` is the
    sub-sequence selected by the index values.

    ``Slice`` (1-D): slices the input's ``value_as_shape`` by the
    concrete start/end/step values.

Reshaping
    ``Reshape`` propagates the ``value_as_shape`` of the *data* input
    when the output is also 1-D and has the same element count as the
    data's ``value_as_shape``.  This covers the pattern
    ``Reshape(Shape(x), [-1])`` used to flatten a shape vector.

    ``Unsqueeze`` and ``Squeeze``: pass the ``value_as_shape`` through
    when the rank change is compatible.

----

Consuming the annotation
------------------------

When a shape-function encounters a shape-input tensor with a
``value_as_shape`` annotation, it reads those symbolic dimension values
directly rather than using the fall-back heuristic.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Operator
     - Shape input
     - How ``value_as_shape`` is used
   * - ``Reshape``
     - ``shape``
     - Output shape dimensions are taken directly from ``value_as_shape``
       (``0`` keeps the corresponding input dim; ``-1`` is inferred).
   * - ``Expand``
     - ``shape``
     - Target shape for broadcast: ``BroadcastShapes(input.shape, value_as_shape)``.
   * - ``Tile``
     - ``repeats``
     - Each output dim is ``input.shape[i] * repeats_value_as_shape[i]``.
   * - ``Resize``
     - ``sizes``
     - Output spatial dimensions are taken from ``value_as_shape``.
   * - ``ConstantOfShape``
     - ``input``
     - Output rank and dimensions are taken from ``value_as_shape``.
   * - ``Pad``
     - ``pads``, ``axes``
     - Padding amounts per axis read from ``value_as_shape`` of ``pads``;
       axis mapping read from ``value_as_shape`` of ``axes`` when present.
   * - ``CenterCropPad``
     - ``shape``
     - Target spatial dimensions read from ``value_as_shape``.
   * - ``OneHot``
     - ``depth``
     - Depth (number of classes) read from ``value_as_shape``.
   * - ``HannWindow``, ``HammingWindow``, ``BlackmanWindow``
     - ``size``
     - Output length read from ``value_as_shape``.
   * - ``MelWeightMatrix``
     - ``output_datatype`` / size args
     - Output dimensions read from ``value_as_shape`` when present.
   * - ``Col2Im``
     - ``image_shape``, ``block_shape``
     - Spatial dimensions read from ``value_as_shape``.
   * - ``MaxUnpool``
     - ``output_shape``
     - Target output shape read from ``value_as_shape``.

This is not an exhaustive list.

----

End-to-end example
------------------

The following model illustrates the full propagation chain.  ``x`` has
shape ``[N, 1]`` (``N`` symbolic); the goal is to ``Expand`` it to
``[N, 1]`` after a round-trip through ``Shape → Concat → Add → Sub``:

.. runpython::
    :rst:

    import numpy
    from onnx_light.onnx_lib import numpy_helper, TensorProto
    from onnx_light.onnx.helper import (
        make_model, make_node, make_graph, make_tensor_value_info, make_opsetid,
    )
    from onnx_light.onnx_core.shape_inference import ShapesContext

    # Inputs and output
    x_vi = make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1])
    out_vi = make_tensor_value_info("expanded", TensorProto.FLOAT, ["N", 1])

    # Initializer: int64[1] = [1]
    one = numpy_helper.from_array(numpy.array([1], dtype=numpy.int64), name="one")

    # Nodes
    n_node = make_node("Shape", ["x"], ["n"], start=0, end=1)
    b_node = make_node("Shape", ["x"], ["b"], start=1, end=2)
    cat_node = make_node("Concat", ["n", "b"], ["shape"], axis=0)
    add_node = make_node("Add", ["shape", "one"], ["shape1"])
    sub_node = make_node("Sub", ["shape1", "one"], ["shape2"])
    exp_node = make_node("Expand", ["x", "shape2"], ["expanded"])

    graph = make_graph(
        [n_node, b_node, cat_node, add_node, sub_node, exp_node],
        "value_as_shape_example",
        [x_vi],
        [out_vi],
        [one],
    )
    model = make_model(graph, opset_imports=[make_opsetid("", 20)])

    ctx = ShapesContext()
    ctx.compute_shape_model(model)

    rows = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 15 30 30",
        "",
        "   * - Tensor",
        "     - storage shape",
        "     - value_as_shape",
    ]
    for name in ("n", "b", "shape", "shape1", "shape2", "expanded"):
        if not ctx.has(name):
            continue
        t = ctx.get(name)
        storage = "[" + ", ".join(str(d) for d in t.shape) + "]"
        if t.has_value_as_shape():
            vas = "[" + ", ".join(str(d) for d in t.value_as_shape()) + "]"
        else:
            vas = "—"
        rows.append(f"   * - ``{name}``")
        rows.append(f"     - ``{storage}``")
        rows.append(f"     - ``{vas}``")

    print("\n".join(rows))

The table shows how the symbolic dimension ``N`` is carried through the
``Shape → Concat → Add → Sub`` pipeline as a ``value_as_shape``
annotation and finally consumed by ``Expand`` to produce the precise
output shape ``[N, 1]``.

----

Interaction with symbolic expressions and constraints
-----------------------------------------------------

The ``value_as_shape`` entries are :cpp:class:`SymDim` values — the same
type used for ordinary shape dimensions.  Each entry may therefore be a
concrete integer *or* a symbolic expression string such as ``"N"`` or
``"1+N"``.

When arithmetic propagation (``Add``, ``Sub``) produces a new
``value_as_shape`` entry, it calls
:func:`~onnx_light.onnx_core.expressions.dim_add` or
:func:`~onnx_light.onnx_core.expressions.dim_sub`, which in turn invokes
the expression simplifier.  This keeps the entries canonical: ``N + 1 - 1``
simplifies back to ``N`` rather than accumulating noise.

During anchor reconciliation (see :ref:`l-design-shape-constraints`), the
``PropagateAnchorConstraintsIntoContext`` pass rewrites **both** the
``shape`` and the ``value_as_shape`` of every context tensor.  This means
that internal symbols (e.g. ``"DYN_0"``) introduced by per-operator
inference are replaced by the user-visible names declared on the graph
boundary.

----

API reference
-------------

* **C++ API**: :cpp:class:`SymTensor`
  (:doc:`/api/cpp/onnx_core/sym_tensor`),
  :cpp:func:`PropagateValueAsShapeArithmetic`
  (:doc:`/api/cpp/onnx_core/shapes/shape_broadcast`).
* **Python API**: :class:`SymTensor`
  (:doc:`/api/python/onnx_core/shape_inference`).
* **Expression library**: :ref:`l-design-expressions`.
* **Constraint mechanism**: :ref:`l-design-shape-constraints`.
