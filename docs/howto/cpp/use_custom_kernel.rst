.. _l-howto-use-custom-kernel-cpp:

:html_theme.sidebar_secondary.remove:

How to use a custom kernel
==========================

This guide covers the C++ interface. For the other language, see
:doc:`../python/use_custom_kernel`.

The native runtime dispatches every ``NodeProto`` against the static C++
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`. An operator that is not
built in — typically an operator from a user-defined domain, an experimental
op, or a stand-in for one that is not yet implemented — would otherwise fail
with ``unsupported op_type``.

This page shows how to plug a custom kernel into the runtime so such a
graph runs in C++. The hook is exposed at several layers; pick
the one that matches your use case.  The per-session layers share the single
C++ entry point
:cpp:func:`onnx_light::core::runtime::RuntimeContext::RegisterCustomKernel`; a
kernel can also be registered globally (see
`Register globally or per session`_).

Register a custom kernel
------------------------

Register a
:cpp:type:`onnx_light::core::runtime::CustomKernelFn` that reads its inputs from
the :cpp:class:`onnx_light::core::runtime::RuntimeContext` and writes the outputs
back through :cpp:func:`onnx_light::core::runtime::RuntimeContext::Put`.


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

Registrations are stored on the ``RuntimeContext``, which can be reused across runs.
Registering, replacing, or unregistering a kernel affects **future resolutions
only**. Already-resolved kernels retain their callable until their session is
destroyed. To use a replacement for those nodes, create a new session and
register the replacement on its context before its first run.

Publishing and replacing values
-------------------------------

A ``RuntimeContext`` is expected to have one value category per name. ``Put`` (Python ``put``),
``PutValue`` (``put_value``), ``PutMap`` (``put_map``), and ``PutShape`` replace
any previous category under that name. ``PutSequence`` (``put_sequence``)
overwrites an existing sequence without first removing it, which allows a runtime
context to execute the same graph repeatedly. It neither checks nor modifies other
value stores: callers must ensure the sequence name is absent from those stores.
``PutValue`` routes ordinary tensors
to the tensor store and structures or encoded values to the structured-value store.
``Set`` (``set``) rejects a name already present in any store. ``Remove``
(``remove``) removes any category, and ``Clear`` (``clear``) clears all value stores.

Kernels must finish computing their result before publishing it: replacing a name
invalidates references to its previous value, including potentially aliased inputs.
C++ kernels should use these publication APIs or ``SetOutput``, which takes
a ``RuntimeContext``, rather than inserting into the mutable maps directly.
Callers of ``PutSequence`` and low-level map mutations must preserve the
single-category invariant. With that invariant maintained, Python
``get_value`` and native consumers such as ``Identity`` cannot select different
values based on lookup order.

Tensor publication retains its allocator handling and add/replace events; changing
from another category to a tensor records a replacement. Non-tensor publication
does not emit add/replace events, but removing an old tensor emits a removal event.
Existing values of any category override graph initializers with the same name.

Prepare a native kernel once
----------------------------

Native kernels with preparation or mutable workspace should register a
:cpp:type:`onnx_light::core::runtime::NodeKernelFn`, globally through
:cpp:func:`onnx_light::core::runtime::RegisterKernelFn` or locally through
:cpp:func:`onnx_light::core::runtime::RuntimeContext::RegisterKernelFn`.
Both use exactly the same lifecycle:

.. code-block:: text

    Global or RuntimeContext-local registry
        -> NodeKernelFn
        -> one KernelBase per resolved node, owned by RuntimeSession
        -> repeated Run(RuntimeContext&) calls

For example, prepare an immutable scale factor once and adapt the output to
the current input shape on each run:

.. code-block:: cpp

    #include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
    #include "onnx_core/runtime/runtime_session.h"

    using namespace onnx_light;
    using namespace onnx_light::core::runtime;

    class ScaleKernel : public KernelBase {
    public:
      ScaleKernel(const NodeProto &node, RuntimeContext &rt, float factor)
          : KernelBase(rt.kernel_ctx()), factor_(factor) {
        set_node(node);
      }

      void Run(RuntimeContext &rt) override {
        const Tensor &x = rt.Get(node_->input(0));
        std::vector<float> values(static_cast<size_t>(x.element_count()));
        for (size_t i = 0; i < values.size(); ++i)
          values[i] = x.AsFloat()[i] * factor_;
        rt.Put(node_->output(0),
               Tensor::FromFloat(node_->output(0), x.shape, values, rt.allocator()));
      }

    private:
      const float factor_;
    };

    NodeKernelFn factory = [](const NodeProto &node, RuntimeContext &rt) {
      float factor = 1.0f;
      for (const auto &attribute : node.attribute())
        if (attribute.name() == "factor")
          factor = attribute.f();
      return std::make_unique<ScaleKernel>(node, rt, factor);
    };

    // Choose one scope. A local factory overrides the global one.
    // RegisterKernelFn("my.domain", "Scale", core::symbolic::Device::kCPU, factory);
    RuntimeContext rt;
    rt.RegisterKernelFn("my.domain", "Scale", core::symbolic::Device::kCPU, factory);
    RuntimeSession session(model);  // model outlives session
    // Supply inputs in rt, then call session.Run(rt) repeatedly.

The factory must return a fresh kernel, attach the original node, and defer
computation to ``Run``. It must not retain the construction-time
``RuntimeContext`` by reference: nested contexts can be short-lived.
The original graph/model and its nodes must remain alive and unchanged until
the session and its kernels are destroyed. Neither the factory adapter nor
dispatch copies attribute-heavy nodes or serializes them for inference.

Kernel identity is independent of input shape. Inputs ``[1, 20]``,
``[100, 20]``, and ``[10, 20]`` reuse the same instance and immutable
parameters. The kernel may resize batch-dependent output/workspace, adapt its
internal plan, or explicitly reject an unsupported shape change.

Callback convenience APIs, including Python, register factories that create
session-owned callback adapters; there is no callback-owned kernel cache.
C++ callable values are copied per resolved node. Shared captures and Python
callable objects remain the caller's responsibility: use native factories
for private mutable preparation state, not a callback cache keyed by node
serialization or input shape. ``RunNode`` is a one-shot resolution; reuse a
``RuntimeSession`` for repeated inference.

``CustomKernelMap`` now stores ``NodeKernelFn`` values. Code that previously
inserted callbacks directly into ``custom_kernels()`` should use
``RegisterCustomKernel`` instead (or explicitly call ``MakeCustomKernelFactory``).
Global custom overrides can also register factories directly through
``RegisterGlobalCustomKernelFactory``.

Register globally or per session
--------------------------------

The examples above register a kernel on a single
:class:`~onnx_light.onnx.reference.ReferenceEvaluator` (equivalently, on one
:class:`RuntimeContext`) — the kernel is only visible to that object. onnx-light
also supports **global** (process-wide) registration: a global kernel is picked
up by future resolutions in *every* :class:`RuntimeContext`, so you install it once
instead of on every evaluator.

Both scopes are supported, and a per-session registration always overrides a
global one for the same ``(domain, op_type)``. Resolution precedence, from
highest to lowest, is: model-local functions, the built-in control-flow
operators (``If`` / ``Loop`` / ``Scan`` / ``SequenceMap``), per-session custom
kernels, global custom kernels, then the built-in
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`.

Local factories use the same device-qualified keys as global factories
(``CPU`` and ``Undefined`` share the host entry). Local callback registration
uses the context's device; global callback overrides remain device-independent.
Subgraph and model-local function contexts inherit a copy of the parent's
local factory registrations at child creation, not its kernel instances.
Their nested sessions retain already-resolved kernels across invocations;
an uninitialized branch picks up the registrations visible at its first run.

Register or mutate global registries before concurrent resolution, or
externally synchronize access; registries are not internally synchronized.
Independent sessions with independent contexts can run concurrently provided
user factories/callbacks are thread-safe and do not share mutable kernel state.
Concurrent runs on the **same** session or context are not supported.

Because an evaluator caches its runtime sessions on first
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.run`, register a global
kernel *before* running the evaluators that should use it.


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

Unregister a kernel and restore the original
--------------------------------------------

:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.unregister_custom_kernel`
removes a previously registered custom kernel. Because custom kernels are
consulted before the built-in
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`, unregistering one
that overrode a built-in operator restores the original built-in kernel on the
next resolution, not the next run of an already-prepared session. It returns
``True`` when a custom kernel was removed and ``False`` otherwise; the empty
domain is normalised to ``"ai.onnx"`` just like when registering.

See the :doc:`Python recipe </howto/python/use_custom_kernel>`
for this example.

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

Use a runtime context
---------------------

For kernels that need direct access to the runtime context — for example
to read sequences or to participate in the event log — use the low-level
:class:`RuntimeContext` binding. The callback receives the raw
:class:`NodeProto` and :class:`RuntimeContext` and is responsible for any
tensor encoding/decoding.  This Python binding mirrors the C++
:cpp:class:`onnx_light::core::runtime::RuntimeContext` one-to-one.


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

The low-level binding and native implementation share the same
:cpp:func:`onnx_light::core::runtime::RuntimeContext::RegisterCustomKernel` entry
point, which is also how C++ extension modules ship additional kernels
without rebuilding ``lib_onnx_kernels``.

See also
--------

* :ref:`l-example-plot-custom-kernel` - end-to-end example that builds a
  model, registers a numpy kernel, and overrides a built-in op.
* :ref:`l-how-to` - other onnx-light how-to recipes.
