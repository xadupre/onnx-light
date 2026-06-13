.. _l-howto-use-custom-kernel:

:html_theme.sidebar_secondary.remove:

How to use a custom kernel
==========================

:class:`~onnx_light.onnx.reference.ReferenceEvaluator` dispatches every
:class:`~onnx_light.onnx_lib.NodeProto` against the static C++
``KernelDispatchTable``. An operator that is not built in — typically an
operator from a user-defined domain, an experimental op, or a stand-in for
one that is not yet implemented — would otherwise fail with
``unsupported op_type``.

This page shows how to plug a custom kernel into the runtime so such a
graph runs. The hook is exposed at three layers; pick the one that matches
your use case.

Register a numpy kernel (recommended)
-------------------------------------

The high-level
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.register_custom_kernel`
wrapper is the easiest entry point. The callable is invoked as
``fn(node, *numpy_inputs)`` and returns either a single
:class:`numpy.ndarray` or a tuple/list of arrays for multi-output ops. It
receives the :class:`~onnx_light.onnx_lib.NodeProto` first, so it can read
attributes.

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

Registrations are stored on the evaluator and reapplied to the fresh
:class:`RuntimeContext` built on every
:py:meth:`~onnx_light.onnx.reference.ReferenceEvaluator.run` call, so the
same evaluator can be reused across runs.

Override a built-in kernel
--------------------------

The empty default ONNX domain is normalised to ``"ai.onnx"``, so
registering a kernel under the default domain takes precedence over the
entry that ``KernelDispatchTable`` would otherwise dispatch. This is
convenient to instrument or replace a specific kernel without patching the
C++ runtime.

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

Use the low-level context binding
---------------------------------

For kernels that need direct access to the runtime context — for example
to read sequences or to participate in the event log — use the low-level
:class:`RuntimeContext` binding. The callback receives the raw
:class:`NodeProto` and :class:`RuntimeContext` and is responsible for any
tensor encoding/decoding.

.. code-block:: python

    from onnx_light.onnx_py._onnxpykernels import runtime as rt

    ctx = rt.RuntimeContext(rt.KernelContext(rt.default_opset(18)))
    ctx.set("x", ...)

    def scale(node, c):
        x = c.get(str(node.input[0]))
        ...
        c.put(str(node.output[0]), ..., "output")

    ctx.register_custom_kernel("my.domain", "Scale", scale)
    rt.run_model(model, ctx)

The same mechanism is available in C++ through
:cpp:func:`onnx::onnx_kernels::RuntimeContext::RegisterCustomKernel`, which
is the entry point for C++ extension modules that ship additional kernels
without rebuilding ``lib_onnx_kernels``.

See also
--------

* :ref:`l-example-plot-custom-kernel` - end-to-end example that builds a
  model, registers a numpy kernel, and overrides a built-in op.
* :ref:`l-design-custom-kernels` - design notes covering dispatch
  precedence and the matching C++ and low-level Python entry points.
* :ref:`l-how-to` - other onnx-light how-to recipes.
