.. _l-howto-use-custom-kernel:

:html_theme.sidebar_secondary.remove:

How to use a custom kernel
==========================

:class:`~onnx_light.onnx.reference.ReferenceEvaluator` dispatches every
:class:`~onnx_light.onnx_lib.NodeProto` against the static C++
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`. An operator that is not
built in — typically an operator from a user-defined domain, an experimental
op, or a stand-in for one that is not yet implemented — would otherwise fail
with ``unsupported op_type``.

This page shows how to plug a custom kernel into the runtime so such a
graph runs, in Python and in C++. The hook is exposed at three layers; pick
the one that matches your use case.  The per-session layers share the single
C++ entry point
:cpp:func:`onnx_light::core::runtime::RuntimeContext::RegisterCustomKernel`; a
kernel can also be registered globally (see
`Register globally or per session`_).

Register a numpy kernel (recommended)
-------------------------------------

The high-level
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.register_custom_kernel`
wrapper is the easiest entry point. The callable is invoked as
``fn(node, *numpy_inputs)`` and returns either a single
:class:`numpy.ndarray` or a tuple/list of arrays for multi-output ops. It
receives the :class:`~onnx_light.onnx_lib.NodeProto` first, so it can read
attributes.  The C++ counterpart registers a
:cpp:type:`onnx_light::core::runtime::CustomKernelFn` that reads its inputs from
the :cpp:class:`onnx_light::core::runtime::RuntimeContext` and writes the outputs
back through :cpp:func:`onnx_light::core::runtime::RuntimeContext::Put`.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import numpy as np
          from onnx_light.onnx_lib import parser
          from onnx_light.onnx.reference import ReferenceEvaluator

          model = parser.parse_model(
              '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>'
              "agraph (float[3] x) => (float[3] y) {"
              "  y = my.domain.Scale<factor=3.0>(x)"
              "}"
          )

          def scale(node, x):
              factor = next(a for a in node.attribute if str(a.name) == "factor").f
              return x * float(factor)

          sess = ReferenceEvaluator(model)
          sess.register_custom_kernel("my.domain", "Scale", scale)
          (y,) = sess.run(None, {"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
          # y == [3., 6., 9.]

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/runtime/kernels/run_nodes.h"
          #include "onnx_core/runtime/runtime_context.h"

          #include <vector>

          using namespace onnx_light::core::runtime;

          RuntimeContext rt(KernelContext(/*opset=*/18));
          rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

          // Register a "my.domain.Scale" kernel that multiplies its single
          // input by the "factor" attribute.
          rt.RegisterCustomKernel(
              "my.domain", "Scale",
              [](const NodeProto &node, RuntimeContext &ctx) {
                float factor = 1.0f;
                for (int i = 0; i < node.attribute_size(); ++i) {
                  if (node.attribute(i).name() == "factor") {
                    factor = node.attribute(i).f();
                  }
                }
                const Tensor &in = ctx.Get(node.input(0));
                std::vector<float> out(static_cast<size_t>(in.element_count()));
                const float *src = in.AsFloat();
                for (size_t i = 0; i < out.size(); ++i) {
                  out[i] = src[i] * factor;
                }
                ctx.Put(node.output(0),
                        Tensor::FromFloat(node.output(0), in.shape, out));
              });

          // node has op_type="Scale", domain="my.domain", input "x", output "y".
          RunNode(node, rt);  // y == [3., 6., 9.]

Registrations are stored on the evaluator's persistent
:class:`RuntimeContext`, so the same evaluator can be reused across runs.
Registering or unregistering a kernel invalidates the evaluator's cached
runtime sessions; the next
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.run` recreates them
and picks up the updated dispatch.

Register globally or per session
--------------------------------

The examples above register a kernel on a single
:class:`~onnx_light.onnx.reference.ReferenceEvaluator` (equivalently, on one
:class:`RuntimeContext`) — the kernel is only visible to that object. onnx-light
also supports **global** (process-wide) registration: a global kernel is picked
up by *every* :class:`RuntimeContext` created afterwards, so you install it once
instead of on every evaluator.

Both scopes are supported, and a per-session registration always overrides a
global one for the same ``(domain, op_type)``. Resolution precedence, from
highest to lowest, is: model-local functions, the built-in control-flow
operators (``If`` / ``Loop`` / ``Scan`` / ``SequenceMap``), per-session custom
kernels, global custom kernels, then the built-in
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`.

Because an evaluator caches its runtime sessions on first
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.run`, register a global
kernel *before* running the evaluators that should use it.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          from onnx_light.onnx.reference import ReferenceEvaluator

          def square(node, x):
              return x * x

          # Registered once; visible to every evaluator created afterwards.
          ReferenceEvaluator.register_custom_kernel_global("my.domain", "Square", square)

          sess = ReferenceEvaluator(model)  # no per-session registration needed
          (y,) = sess.run(None, {"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})

          # Remove the global registration when done.
          ReferenceEvaluator.unregister_custom_kernel_global("my.domain", "Square")

      The low-level counterparts live on the ``runtime`` submodule:
      ``runtime.register_custom_kernel(domain, op_type, fn)``,
      ``runtime.unregister_custom_kernel(domain, op_type)`` and
      ``runtime.clear_custom_kernels()`` (module-level, i.e. global), as opposed
      to the identically named methods on :class:`RuntimeContext` (per session).

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/runtime/kernels/kernel_dispatch_table.h"

          using namespace onnx_light::core::runtime;

          // Global: picked up by every RuntimeContext.
          RegisterGlobalCustomKernel(
              "my.domain", "Scale",
              [](const NodeProto &node, RuntimeContext &c) {
                const Tensor &x = c.Get(node.input(0));
                // ...
                c.Put(node.output(0), /* Tensor */ ...);
              });

          // Per session: only this context (overrides the global one above).
          RuntimeContext ctx(KernelContext(/*opset=*/18));
          ctx.RegisterCustomKernel("my.domain", "Scale", /* ... */);

          UnregisterGlobalCustomKernel("my.domain", "Scale");  // remove global

Override a built-in kernel
--------------------------

The empty default ONNX domain is normalised to ``"ai.onnx"``, so
registering a kernel under the default domain takes precedence over the
entry that :cpp:func:`onnx_light::core::runtime::KernelDispatchTable` would
otherwise dispatch. This is convenient to instrument or replace a specific
kernel without patching the C++ runtime.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          sess = ReferenceEvaluator(
              parser.parse_model(
                  '<ir_version: 10, opset_import: ["" : 18]>'
                  "agraph (float[3] x) => (float[3] y) { y = Abs(x) }"
              )
          )
          sess.register_custom_kernel("", "Abs", lambda node, x: -x)
          (y,) = sess.run(None, {"x": np.array([-1.0, -2.0, -3.0], dtype=np.float32)})
          # Abs replaced by negation: y == [1., 2., 3.]

Unregister a kernel and restore the original
--------------------------------------------

:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.unregister_custom_kernel`
removes a previously registered custom kernel. Because custom kernels are
consulted before the built-in
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`, unregistering one
that overrode a built-in operator restores the original built-in kernel on the
next :py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.run`. It returns
``True`` when a custom kernel was removed and ``False`` otherwise; the empty
domain is normalised to ``"ai.onnx"`` just like when registering.

.. code-block:: python

    sess.register_custom_kernel("", "Abs", lambda node, x: -x)
    # ... use the negated override ...
    sess.unregister_custom_kernel("", "Abs")  # restores the built-in Abs
    (y,) = sess.run(None, {"x": np.array([-1.0, -2.0, -3.0], dtype=np.float32)})
    # y == [1., 2., 3.] (built-in Abs again)

At the C++ / low-level binding layer the same is achieved with
:cpp:func:`onnx_light::core::runtime::RuntimeContext::UnregisterCustomKernel`,
which erases the custom entry so :cpp:func:`RunNode` falls back to the built-in
kernel.

.. code-block:: cpp

    using namespace onnx_light::core::runtime;

    RuntimeContext rt(KernelContext(/*opset=*/18));
    rt.Set("x", Tensor::FromFloat("x", {3}, {-1.0f, -2.0f, -3.0f}));

    // The empty domain is normalised to "ai.onnx", so this overrides
    // the built-in Abs entry with a negation.
    rt.RegisterCustomKernel(
        "", "Abs", [](const NodeProto &node, RuntimeContext &ctx) {
          const Tensor &in = ctx.Get(node.input(0));
          std::vector<float> out(static_cast<size_t>(in.element_count()));
          const float *src = in.AsFloat();
          for (size_t i = 0; i < out.size(); ++i) {
            out[i] = -src[i];
          }
          ctx.Put(node.output(0),
                  Tensor::FromFloat(node.output(0), in.shape, out));
        });
    // Abs replaced by negation: y == [1., 2., 3.]

Use the low-level context binding
---------------------------------

For kernels that need direct access to the runtime context — for example
to read sequences or to participate in the event log — use the low-level
:class:`RuntimeContext` binding. The callback receives the raw
:class:`NodeProto` and :class:`RuntimeContext` and is responsible for any
tensor encoding/decoding.  This Python binding mirrors the C++
:cpp:class:`onnx_light::core::runtime::RuntimeContext` one-to-one.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          from onnx_light.onnx_py._onnxpykernels import runtime as rt

          ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
          ctx.set("x", ...)

          def scale(node, c):
              x = c.get(str(node.input[0]))
              ...
              c.put(str(node.output[0]), ..., "output")

          ctx.register_custom_kernel("my.domain", "Scale", scale)
          rt.register_model_functions(model, ctx)
          plan = ctx.get_execution_plan(model.graph)
          rt.RuntimeSession(plan).run(ctx)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx_core/runtime/kernels/run_nodes.h"
          #include "onnx_core/runtime/runtime_context.h"

          using namespace onnx_light::core::runtime;

          RuntimeContext ctx(KernelContext(/*opset=*/18));
          ctx.Set("x", /* Tensor */ ...);

          ctx.RegisterCustomKernel(
              "my.domain", "Scale",
              [](const NodeProto &node, RuntimeContext &c) {
                const Tensor &x = c.Get(node.input(0));
                // ...
                c.Put(node.output(0), /* Tensor */ ...);
              });
          RegisterModelFunctions(model, ctx);
          const auto &plan = ctx.GetExecutionPlan(model.graph());
          RuntimeSession(plan).Run(ctx);

The low-level binding and the C++ tab above are two faces of the same
:cpp:func:`onnx_light::core::runtime::RuntimeContext::RegisterCustomKernel` entry
point, which is also how C++ extension modules ship additional kernels
without rebuilding ``lib_onnx_kernels``.

See also
--------

* :ref:`l-example-plot-custom-kernel` - end-to-end example that builds a
  model, registers a numpy kernel, and overrides a built-in op.
* :ref:`l-how-to` - other onnx-light how-to recipes.
