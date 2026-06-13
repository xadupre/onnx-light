.. _l-howto-use-custom-shape-inference:

:html_theme.sidebar_secondary.remove:

How to use a custom optim shape inference function
==================================================

The built-in shape-inference dispatch table covers all standard ONNX
operators. Models that include custom operators from a private domain
cannot be shape-inferred out of the box: the engine encounters a node it
does not recognise and raises :class:`ValueError`.

*onnx-light* lets you plug in a Python callback that handles the shape
inference for any ``(domain, op_type)`` pair. Once registered, the
callback is invoked transparently by
:func:`~onnx_light.onnx_optim.shape_inference.compute_shape_node` and
:func:`~onnx_light.onnx_optim.shape_inference.compute_shape_model`.

Import the shape-inference module
----------------------------------

All public symbols are accessible via the high-level Python module:

.. code-block:: python

    from onnx_light.onnx_optim.shape_inference import (
        ShapesContext,
        OptimTensor,
        compute_shape_node,
        compute_shape_model,
    )

Write the callback
------------------

The callback must have the signature ``fn(ctx, node) -> None``.

* ``ctx`` is the current :class:`~onnx_light.onnx_optim.shape_inference.ShapesContext`.
  Call :meth:`~onnx_light.onnx_optim.shape_inference.ShapesContext.get` to read
  input descriptors and :meth:`~onnx_light.onnx_optim.shape_inference.ShapesContext.set`
  to write output descriptors.
* ``node`` is the :class:`~onnx_light.onnx_lib.NodeProto` being processed.
  Use ``node.input``, ``node.output``, and ``node.attribute`` to inspect
  the operator's operands and attributes.
* The callback must call ``ctx.set(name, OptimTensor(dtype, shape))`` for
  **every** output of the node before returning.

.. code-block:: python

    def infer_my_op(ctx, node):
        """Infers the output shape of MyDomain::MyOp."""
        x = ctx.get(str(node.input[0]))
        # derive the output shape from the input(s) and attributes
        out_shape = list(x.shape)
        ctx.set(str(node.output[0]), OptimTensor(x.dtype, out_shape))

Symbolic dimensions (string values such as ``"N"`` or ``"batch"``) are
preserved automatically; compare them with ``isinstance(d, int)`` before
doing arithmetic on them.

Register the callback
---------------------

Call :meth:`~onnx_light.onnx_optim.shape_inference.ShapesContext.set_custom_shape_inference_function`
on the context before running inference:

.. code-block:: python

    ctx = ShapesContext()
    ctx.set_custom_shape_inference_function("my.domain", "MyOp", infer_my_op)

Passing an empty string as the domain normalises it to ``"ai.onnx"``
(the default ONNX domain). The registration is stored on the context
object and applied for every subsequent call to ``compute_shape_node`` or
``compute_shape_model`` that uses the same context.

You can check whether a callback is registered:

.. code-block:: python

    ctx.has_custom_shape_inference_function("my.domain", "MyOp")  # True
    list(ctx.custom_shape_inference_keys())  # ["my.domain:MyOp"]

Infer shapes for a single node
-------------------------------

:func:`~onnx_light.onnx_optim.shape_inference.compute_shape_node` processes
one node at a time. Seed the context with the node's input descriptors
first:

.. code-block:: python

    import onnx_light.onnx as onnxl
    import onnx_light.onnx.defs as defs
    import onnx_light.onnx.helper as oh

    defs.register_onnx_operator_set_schema()

    node = oh.make_node("MyOp", ["X"], ["Y"], domain="my.domain")

    ctx = ShapesContext()
    ctx.set_opset_version("my.domain", 1)
    ctx.set("X", OptimTensor(onnxl.TensorProto.FLOAT, [4, 8]))
    ctx.set_custom_shape_inference_function("my.domain", "MyOp", infer_my_op)

    compute_shape_node(ctx, node)
    print(ctx.get("Y").shape)   # [4, 8]

Infer shapes for a whole model
-------------------------------

:func:`~onnx_light.onnx_optim.shape_inference.compute_shape_model` walks
every node of the main graph in topological order. Register the callback
on the context **before** calling it:

.. code-block:: python

    ctx = ShapesContext()
    ctx.set_custom_shape_inference_function("my.domain", "MyOp", infer_my_op)
    compute_shape_model(ctx, model)

    for name in ["intermediate", "output"]:
        t = ctx.get(name)
        print(name, list(t.shape), t.dtype)

:func:`~onnx_light.onnx_optim.shape_inference.infer_shapes_model` creates
its own internal context and does **not** accept a pre-configured
context. Use :func:`~onnx_light.onnx_optim.shape_inference.compute_shape_model`
instead when your model contains custom operators, then apply the inferred
shapes back manually if needed:

.. code-block:: python

    from onnx_light.onnx_optim.shape_inference import (
        compute_shape_model,
        apply_inferred_shapes_to_model,
    )

    ctx = ShapesContext()
    ctx.set_custom_shape_inference_function("my.domain", "MyOp", infer_my_op)
    compute_shape_model(ctx, model)
    apply_inferred_shapes_to_model(ctx, model)

See also
--------

* :ref:`l-example-plot-shape-inference-custom-op` — end-to-end runnable
  example that defines a ``ScaledLinear`` custom operator and registers a
  shape callback for it.
* :ref:`l-example-plot-shape-inference` — comparison of model-level and
  node-by-node shape inference for standard operators.
* :ref:`l-how-to` — other onnx-light how-to recipes.
