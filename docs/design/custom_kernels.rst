.. _l-design-custom-kernels:

Custom kernels for ReferenceEvaluator
=====================================

The C++ runtime in ``onnx_light/onnx_core/runtime`` dispatches every
:class:`~onnx_light.onnx_lib.NodeProto` against a mutable
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`, populated with
every built-in operator kernel from ``onnx_light/onnx_extensions/onnx_kernels`` via
:cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions`.
Any operator that is not built in — typically an operator from a
user-defined domain, an experimental op, or a stand-in for one not yet
implemented — would otherwise fail with ``unsupported op_type``.

The custom-kernel hook documented here lets callers extend (or override)
the runtime without modifying the built-in dispatch table. It is exposed
at three layers:

* C++: :cpp:func:`onnx_light::core::runtime::RuntimeContext::RegisterCustomKernel`,
* low-level Python: ``RuntimeContext.register_custom_kernel(domain, op_type, fn)``,
* high-level Python: :py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.register_custom_kernel`.

All three share the same model: a name-keyed
:cpp:type:`onnx_light::core::runtime::CustomKernelMap` stored on the active
:cpp:class:`~onnx_light::core::runtime::RuntimeContext`. ``RunNode`` consults
this map *before* the built-in dispatch table; model-local
:cpp:class:`FunctionProto` definitions and the dedicated control-flow
paths (``If``, ``Loop``, ``Scan``, ``SequenceMap``) still take
precedence so existing graphs are unaffected.

Dispatch precedence
-------------------

For each :class:`~onnx_light.onnx_lib.NodeProto` evaluated by ``RunNode``:

#. ``If`` / ``Loop`` / ``Scan`` / ``SequenceMap`` are routed to their
   dedicated subgraph runners.
#. Model-local :class:`~onnx_light.onnx_lib.FunctionProto` definitions found on the
   :cpp:class:`RuntimeContext` are inlined.
#. The custom-kernel map is consulted by the ``"<domain>:<op_type>"``
   key (the empty default ONNX domain is normalised to ``"ai.onnx"``).
   A registered entry overrides the static table.
#. ``KernelDispatchTable`` resolves the built-in kernel.
#. Otherwise the dispatcher throws ``unsupported op_type``.

C++ usage
---------

A custom kernel is any callable compatible with
:cpp:type:`onnx_light::core::runtime::CustomKernelFn`, i.e.
``std::function<void(const NodeProto &, RuntimeContext &)>``. The
callback is in charge of reading its inputs from the context's tensor
map and writing its outputs back through
:cpp:func:`RuntimeContext::Put` (or :cpp:func:`SetOutput`).

.. code-block:: cpp

  #include "onnx_core/runtime/run_nodes.h"
  #include "onnx_core/runtime/runtime_context.h"

  using namespace onnx_light::core::runtime;

  RuntimeContext rt(KernelContext(/*opset=*/18));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  // Register a "my.domain.Scale" custom kernel that multiplies its
  // single input by the "factor" attribute.
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
  RunNode(node, rt);

The empty domain is normalised to ``"ai.onnx"``, so registering
``("", "Abs", fn)`` overrides the built-in ``Abs`` entry.

The same mechanism is the entry point for C++ extension modules: a
shared library exposing additional kernels can register them on a
:class:`~onnx_light.onnx.reference.RuntimeContext` (either directly, or through the Python binding
when loaded as a Python module) without rebuilding ``lib_onnx_kernels``.

Low-level Python binding
------------------------

The :class:`RuntimeContext` exposed by
:mod:`onnx_light.onnx_py._onnxpykernels` mirrors the C++ API:

.. code-block:: python

  from onnx_light.onnx_py._onnxpykernels import runtime as rt

  ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
  ctx.set("x", ...)

  def scale(node, c):
      # Direct access to the raw NodeProto / RuntimeContext.
      x = c.get(str(node.input[0]))
      ...
      c.put(str(node.output[0]), ..., "output")

  ctx.register_custom_kernel("my.domain", "Scale", scale)
  rt.run_model(model, ctx)

This binding is intentionally close to the C++ API: the callback
receives the raw :class:`NodeProto` and :class:`RuntimeContext` and is
responsible for any tensor encoding/decoding. It is the appropriate
entry point for C++ extension modules and for kernels that need to
inspect or mutate the runtime context beyond their declared inputs and
outputs (for example to read sequences or to participate in the event
log).

Execution plans
~~~~~~~~~~~~~~~

When :attr:`RuntimeContext.release_intermediates` is enabled, the runtime
uses a precomputed per-graph release schedule — an :class:`ExecutionPlan`
— to drop intermediate tensors as soon as their last reference has been
consumed. The plan depends only on the graph topology, so it is built once
and reused across every run of the same model. It is exposed on the
low-level Python binding too:

.. code-block:: python

  from onnx_light.onnx_py._onnxpykernels import runtime as rt

  # Build a plan directly from a GraphProto (or FunctionProto)...
  plan = rt.ExecutionPlan(model.graph)
  plan.num_nodes          # nodes covered by the plan
  plan.keep()             # names that must never be released
  plan.actions()          # ordered ExecuteAction steps derived from metadata

  # ...or let a RuntimeContext build and cache it on first use.
  ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
  plan = ctx.get_execution_plan(model.graph)
  ctx.clear_execution_plans()   # drop the cache if the model was mutated

:func:`ExecutionPlan.keep` returns the structural set of names seeded from
the graph's inputs, initializers and outputs (for a function, its declared
inputs and outputs), while :func:`ExecutionPlan.actions` returns the ordered
list of :class:`ExecuteAction` steps derived from the in-place / lifetime
metadata written to each node by the in-place reuse pass.
:func:`ExecutionPlan.release_after` removes, for a given node, the
intermediates whose last use falls there from a
:class:`RuntimeContext`. :func:`RuntimeContext.get_execution_plan` caches
one plan per ``GraphProto`` / ``FunctionProto`` identity, so the analysis
is paid only once for the lifetime of the context.

High-level Python API
---------------------

For most users the
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.register_custom_kernel`
wrapper is the easiest entry point: it accepts numpy inputs and a
numpy output (or a tuple/list of arrays for multi-output ops).

.. code-block:: python

  import numpy as np
  from onnx_light.onnx_lib import parser
  from onnx_light.onnx.reference import ReferenceEvaluator

  model = parser.parse_model(
      '<ir_version: 10, opset_import: ["" : 18, "my.domain" : 1]>'
      'agraph (float[3] x) => (float[3] y) {'
      '  y = my.domain.Scale<factor=3.0>(x)'
      '}'
  )

  def scale(node, x):
      factor = next(a for a in node.attribute if str(a.name) == "factor").f
      return x * float(factor)

  sess = ReferenceEvaluator(model)
  sess.register_custom_kernel("my.domain", "Scale", scale)
  (y,) = sess.run(None, {"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
  # y == [3., 6., 9.]

Registrations are stored on the evaluator and reapplied to the fresh
:class:`RuntimeContext` built on every
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.run` call, so
the same evaluator can be reused across runs.

Symmetry with shape inference
-----------------------------

The custom-kernel hook is intentionally symmetric to
:cpp:func:`onnx::core::shapes::ShapesContext::SetCustomShapeInferenceFunction`
in the shape-inference layer: both allow callers to plug a callable
keyed by ``(domain, op_type)`` into a per-invocation context and both
are consulted *before* the built-in domain dispatch.

See also
--------

* :ref:`l-design-kernels`
* :ref:`l-example-plot-custom-kernel`
* :doc:`../api/cpp/onnx_extensions/onnx_kernels/index`
