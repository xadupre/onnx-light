.. _l-howto-use-custom-shape-inference:

:html_theme.sidebar_secondary.remove:

How to use a custom optim shape inference function
==================================================

The built-in shape-inference dispatch table covers all standard ONNX
operators. Models that include custom operators from a private domain
cannot be shape-inferred out of the box: the engine encounters a node it
does not recognise and raises :class:`ValueError`.

*onnx-light* lets you plug in a callback that handles the shape
inference for any ``(domain, op_type)`` pair, in Python and in C++. Once
registered, the callback is invoked transparently by
:func:`~onnx_light.onnx_core.shape_inference.compute_shape_node` and
:func:`~onnx_light.onnx_core.shape_inference.compute_shape_model` in
Python, and by
:cpp:func:`onnx_light::core::shapes::ShapesContext::ComputeShapeNode` and
:cpp:func:`onnx_light::core::shapes::ShapesContext::ComputeShapeModel` in
C++.

Import the shape-inference module
----------------------------------

All public symbols are accessible via the high-level Python module, or the
matching C++ headers:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          from onnx_light.onnx_core.shape_inference import (
              ShapesContext,
              SymTensor,
              compute_shape_node,
              compute_shape_model,
          )

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/symbolic/sym_tensor.h"
          #include "onnx_core/shapes/shape_inference.h"
          #include "onnx_core/shapes/shapes_context.h"

          using namespace onnx::core::symbolic;      // SymTensor, SymShape, SymDim
          using namespace onnx::core::shapes;        // ShapesContext

Write the callback
------------------

The callback must have the signature ``fn(ctx, node) -> None`` in Python and
``void fn(ShapesContext &ctx, const NodeProto &node)`` in C++.

* ``ctx`` is the current :class:`~onnx_light.onnx_core.shape_inference.ShapesContext`.
  Call :meth:`~onnx_light.onnx_core.shape_inference.ShapesContext.get` to read
  input descriptors and :meth:`~onnx_light.onnx_core.shape_inference.ShapesContext.set`
  to write output descriptors.
* ``node`` is the :class:`~onnx_light.onnx_lib.NodeProto` being processed.
  Use ``node.input``, ``node.output``, and ``node.attribute`` to inspect
  the operator's operands and attributes.
* The callback must register an :class:`SymTensor` for **every** output of
  the node before returning.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          def infer_my_op(ctx, node):
              """Infers the output shape of MyDomain::MyOp."""
              x = ctx.get(str(node.input[0]))
              # derive the output shape from the input(s) and attributes
              out_shape = list(x.shape)
              ctx.set(str(node.output[0]), SymTensor(x.dtype, out_shape))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          void infer_my_op(ShapesContext &ctx, const NodeProto &node) {
            const SymTensor &x = ctx.Get(node.input(0));
            // derive the output shape from the input(s) and attributes
            SymShape out_shape = x.Shape();
            ctx.Set(node.output(0), SymTensor(/*data=*/nullptr, x.Dtype(), out_shape));
          }

Symbolic dimensions (string values such as ``"N"`` or ``"batch"``) are
preserved automatically; compare them with ``isinstance(d, int)`` before
doing arithmetic on them in Python (or ``SymDim::IsInt`` in C++).

Register the callback
---------------------

Call :meth:`~onnx_light.onnx_core.shape_inference.ShapesContext.set_custom_shape_inference_function`
(C++ :cpp:func:`~onnx::core::shapes::ShapesContext::SetCustomShapeInferenceFunction`)
on the context before running inference:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          ctx = ShapesContext()
          ctx.set_custom_shape_inference_function("my.domain", "MyOp", infer_my_op)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          ShapesContext ctx;
          ctx.SetCustomShapeInferenceFunction("my.domain", "MyOp", infer_my_op);

Passing an empty string as the domain normalises it to ``"ai.onnx"``
(the default ONNX domain). The registration is stored on the context
object and applied for every subsequent call to ``compute_shape_node`` or
``compute_shape_model`` that uses the same context.

You can check whether a callback is registered:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          ctx.has_custom_shape_inference_function("my.domain", "MyOp")  # True
          list(ctx.custom_shape_inference_keys())  # ["my.domain:MyOp"]

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          ctx.GetCustomShapeInferenceFunction("my.domain", "MyOp") != nullptr;  // true
          for (const auto &kv : ctx.CustomShapeInferenceFunctions()) {
            // kv.first == "my.domain:MyOp"
          }

Infer shapes for a single node
-------------------------------

:func:`~onnx_light.onnx_core.shape_inference.compute_shape_node`
(C++ :cpp:func:`~onnx::core::shapes::ShapesContext::ComputeShapeNode`)
processes one node at a time. Seed the context with the node's input
descriptors first:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx as onnxl
          import onnx_light.onnx.defs as defs
          import onnx_light.onnx.helper as oh

          defs.register_onnx_operator_set_schema()

          node = oh.make_node("MyOp", ["X"], ["Y"], domain="my.domain")

          ctx = ShapesContext()
          ctx.set_opset_version("my.domain", 1)
          ctx.set("X", SymTensor(onnxl.TensorProto.FLOAT, [4, 8]))
          ctx.set_custom_shape_inference_function("my.domain", "MyOp", infer_my_op)

          compute_shape_node(ctx, node)
          print(ctx.get("Y").shape)   # [4, 8]

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_proto/onnx_helper.h"  // make_node helpers, etc.

          NodeProto node;
          node.set_op_type("MyOp");
          node.set_domain("my.domain");
          node.add_input("X");
          node.add_output("Y");

          ShapesContext ctx;
          ctx.SetOpsetVersion("my.domain", 1);
          ctx.Set("X", SymTensor(/*data=*/nullptr, TensorType::kFloat,
                                   {SymDim(4), SymDim(8)}));
          ctx.SetCustomShapeInferenceFunction("my.domain", "MyOp", infer_my_op);

          ctx.ComputeShapeNode(node);
          // ctx.Get("Y").Shape() == {4, 8}

Infer shapes for a whole model
-------------------------------

:func:`~onnx_light.onnx_core.shape_inference.compute_shape_model`
(C++ :cpp:func:`~onnx::core::shapes::ShapesContext::ComputeShapeModel`)
walks every node of the main graph in topological order. Register the
callback on the context **before** calling it:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          ctx = ShapesContext()
          ctx.set_custom_shape_inference_function("my.domain", "MyOp", infer_my_op)
          compute_shape_model(ctx, model)

          for name in ["intermediate", "output"]:
              t = ctx.get(name)
              print(name, list(t.shape), t.dtype)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          ShapesContext ctx;
          ctx.SetCustomShapeInferenceFunction("my.domain", "MyOp", infer_my_op);
          ctx.ComputeShapeModel(model);

          for (const char *name : {"intermediate", "output"}) {
            const SymTensor &t = ctx.Get(name);
            // t.Shape(), t.Dtype()
          }

:func:`~onnx_light.onnx_core.shape_inference.infer_shapes_model`
(C++ :cpp:func:`onnx_light::core::shapes::InferShapesModel`) creates its own
internal context and does **not** accept a pre-configured context. Use
:func:`~onnx_light.onnx_core.shape_inference.compute_shape_model`
(C++ :cpp:func:`onnx_light::core::shapes::ShapesContext::ComputeShapeModel`)
instead when your model contains custom operators, then apply the inferred
shapes back manually if needed:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          from onnx_light.onnx_core.shape_inference import (
              compute_shape_model,
              apply_inferred_shapes_to_model,
          )

          ctx = ShapesContext()
          ctx.set_custom_shape_inference_function("my.domain", "MyOp", infer_my_op)
          compute_shape_model(ctx, model)
          apply_inferred_shapes_to_model(ctx, model)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          ShapesContext ctx;
          ctx.SetCustomShapeInferenceFunction("my.domain", "MyOp", infer_my_op);
          ctx.ComputeShapeModel(model);
          ctx.ApplyInferredShapesToModel(model);

See also
--------

* :ref:`l-example-plot-shape-inference-custom-op` — end-to-end runnable
  example that defines a ``ScaledLinear`` custom operator and registers a
  shape callback for it.
* :ref:`l-example-plot-shape-inference` — comparison of model-level and
  node-by-node shape inference for standard operators.
* :ref:`l-how-to` — other onnx-light how-to recipes.
