.. _l-howto-use-custom-kernel:
.. _l-howto-use-custom-kernel-python:

:html_theme.sidebar_secondary.remove:

How to use a custom kernel
==========================

This guide covers the Python interface. For the other language, see
:doc:`../cpp/use_custom_kernel`.

:class:`~onnx_light.onnx.reference.ReferenceEvaluator` dispatches every
:class:`~onnx_light.onnx_lib.NodeProto` against the static C++
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`. An operator that is not
built in — typically an operator from a user-defined domain, an experimental
op, or a stand-in for one that is not yet implemented — would otherwise fail
with ``unsupported op_type``.

This page shows how to plug a custom kernel into the runtime so such a
graph runs in Python. The hook is exposed at several layers; pick
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

Registrations are stored on the evaluator's persistent
:class:`RuntimeContext`, so the same evaluator can be reused across runs.
Registering, replacing, or unregistering a kernel affects **future resolutions
only**. Already-resolved kernels retain their callable until their session is
destroyed. To use a replacement for those nodes, create a new evaluator and
register the replacement on it before its first run.

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

Override a built-in kernel
--------------------------

The empty default ONNX domain is normalised to ``"ai.onnx"``, so
registering a kernel under the default domain takes precedence over the
entry that :cpp:func:`onnx_light::core::runtime::KernelDispatchTable` would
otherwise dispatch. This is convenient to instrument or replace a specific
kernel without patching the C++ runtime.


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
next resolution, not the next run of an already-prepared session. It returns
``True`` when a custom kernel was removed and ``False`` otherwise; the empty
domain is normalised to ``"ai.onnx"`` just like when registering.

.. code-block:: python

    sess.register_custom_kernel("", "Abs", lambda node, x: -x)
    # ... use the negated override ...
    sess.unregister_custom_kernel("", "Abs")  # leaves prepared kernels intact
    sess = ReferenceEvaluator(model)  # a new evaluator resolves the built-in Abs
    (y,) = sess.run(None, {"x": np.array([-1.0, -2.0, -3.0], dtype=np.float32)})
    # y == [1., 2., 3.] (built-in Abs again)

At the C++ / low-level binding layer the same is achieved with
:cpp:func:`onnx_light::core::runtime::RuntimeContext::UnregisterCustomKernel`,
which erases the custom entry so :cpp:func:`RunNode` falls back to the built-in
kernel.

See the :doc:`C++ recipe </howto/cpp/use_custom_kernel>`
for this example.

Use the low-level context binding
---------------------------------

For kernels that need direct access to the runtime context — for example
to read sequences or to participate in the event log — use the low-level
:class:`RuntimeContext` binding. The callback receives the raw
:class:`NodeProto` and :class:`RuntimeContext` and is responsible for any
tensor encoding/decoding.  This Python binding mirrors the C++
:cpp:class:`onnx_light::core::runtime::RuntimeContext` one-to-one.


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

The low-level binding and native implementation share the same
:cpp:func:`onnx_light::core::runtime::RuntimeContext::RegisterCustomKernel` entry
point, which is also how C++ extension modules ship additional kernels
without rebuilding ``lib_onnx_kernels``.

See also
--------

* :ref:`l-example-plot-custom-kernel` - end-to-end example that builds a
  model, registers a numpy kernel, and overrides a built-in op.
* :ref:`l-how-to` - other onnx-light how-to recipes.
