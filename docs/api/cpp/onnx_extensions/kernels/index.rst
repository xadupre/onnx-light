onnx_kernels
============

This module documents the C++ static library ``lib_onnx_kernels`` that
bundles the per-operator ONNX kernel implementations (``Add``, ``Conv``,
``Resize``, ...) for every standard domain. It only depends on
``lib_onnx_proto`` and ``lib_onnx_core``.

The generic execution engine that ``onnx_kernels`` plugs into — the
runtime value types (:cpp:struct:`onnx_light::core::runtime::Tensor`,
:cpp:struct:`onnx_light::core::runtime::Sequence`,
:cpp:struct:`onnx_light::core::runtime::Map`),
:cpp:class:`onnx_light::core::runtime::RuntimeContext`, the
:cpp:func:`onnx_light::core::runtime::RunNode` /
:cpp:func:`onnx_light::core::runtime::RunModel` traversal, random-number
helpers, and the raw-buffer allocator — lives in ``onnx_core`` instead (see
:doc:`../../onnx_core/runtime/index`), so that it has no dependency on any
particular set of operator kernels. Control-flow operators (``If``,
``Loop``, ``Scan``) live there too, for the same reason.

``onnx_core``'s kernel dispatch table
(:cpp:func:`onnx_light::core::runtime::KernelDispatchTable`) starts out
empty. ``onnx_kernels`` populates it with its own per-operator trampolines
by calling
:cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions` once, which
also registers the ``SequenceMap`` output-packing callback used by
:cpp:func:`onnx_light::core::runtime::RunNode`. Any consumer that runs
nodes/graphs/models built from standard ONNX operators (Python bindings,
the backend-test runner, the gtest binary, ...) must call
:cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions` once before
doing so.

This module also documents:

* a :cpp:struct:`onnx_light::onnx_backend_test::TestCase` bundle of
  :cpp:class:`onnx_light::ModelProto` + expected input/output data sets;
* the :cpp:func:`onnx_light::onnx_backend_test::Expect` helper and
  :cpp:func:`onnx_light::onnx_backend_test::CollectTestCases` registry;
* the ONNX operator kernel implementations themselves under
  ``onnx_extensions/kernels/kernels/``.

.. toctree::
    :maxdepth: 1

    kernel_dispatch_table
    kernel_run_helpers
    kernels/index
    tuning/index
