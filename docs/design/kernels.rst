.. _l-design-kernels:

C++ Kernels
===========

The kernel layer is split across two libraries:

* :epkg:`onnx_light/onnx_core/runtime` (built as :epkg:`lib_onnx_core`)
  hosts the generic execution engine: the runtime tensor/container types
  (:cpp:struct:`onnx_light::core::runtime::Tensor`,
  :cpp:struct:`onnx_light::core::runtime::Sequence`,
  :cpp:class:`onnx_light::core::runtime::RuntimeContext`), the
  graph/node execution helpers
  (:cpp:func:`onnx_light::core::runtime::RunNode`,
  :cpp:func:`onnx_light::core::runtime::RunGraph`,
  :cpp:func:`onnx_light::core::runtime::RunModel`,
  :cpp:func:`onnx_light::core::runtime::RunSubgraph`), the kernel
  dispatch table, and the control-flow kernels (``If``, ``Loop``, ``Scan``).
* :epkg:`onnx_light/onnx_extensions/onnx_kernels` (built as :epkg:`lib_onnx_kernels`)
  contains every other per-operator kernel implementation, grouped under
  :epkg:`onnx_light/onnx_extensions/onnx_kernels/kernels`\ ``/<domain>/``.

Control-flow operators are the one exception to "all kernels live in
``onnx_kernels``": running their subgraphs recursively calls
:cpp:func:`onnx_light::core::runtime::RunGraph`, which must live in
``onnx_core``, so keeping ``If``/``Loop``/``Scan`` in ``onnx_kernels``
would require ``onnx_core`` to depend on ``onnx_kernels``.

:epkg:`lib_onnx_backend_test` depends on both libraries and uses these
kernels to compute expected outputs for backend test cases.

Registration mechanism
-----------------------

``onnx_core`` never depends on ``onnx_kernels``, so
:cpp:func:`onnx_light::core::runtime::KernelDispatchTable` starts out
empty: it is a mutable registry populated at runtime via
:cpp:func:`onnx_light::core::runtime::RegisterKernelFn`. ``onnx_kernels``
keeps its own table of every built-in operator trampoline and exposes
:cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions`, which
iterates that table calling ``RegisterKernelFn`` for each operator and
also registers the ``SequenceMap`` output-packing callback (via
:cpp:func:`onnx_light::core::runtime::RegisterSequenceMapPackFn`).

Any consumer that runs nodes/graphs/models built from standard ONNX
operators — the Python bindings, the backend-test runner, the gtest
binary, examples — must call
:cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions` once before
doing so. The call is idempotent, so it is safe to call from multiple
independent entry points (module init, a gtest global environment, ...).

Kernel organization
--------------------

Kernels are grouped by ONNX domain family:

* ``math``, ``logical``, ``tensor``, ``reduction``, ``nn``,
* ``sequence``, ``optional``, ``quantization``,
* ``traditionalml``, ``training``, ``image``, ``text``,
  ``object_detection``, ``preview``, ``generator``

live in ``onnx_kernels``; ``controlflow`` (``If``, ``Loop``, ``Scan``)
lives in ``onnx_core/runtime`` instead. Each family has an
``include_<family>_kernels.h`` umbrella header exposing the kernel classes
for that group.

Runtime model
-------------

:cpp:func:`~onnx_light::core::runtime::RunNode` executes one
:class:`~onnx_light.onnx_lib.NodeProto` against a
:cpp:type:`onnx_light::core::runtime::TensorMap` stored in
:cpp:class:`onnx_light::core::runtime::RuntimeContext`:

* inputs are looked up by tensor name,
* outputs are written back by output name,
* dispatch is keyed by ``(domain, op_type)`` through
  :cpp:func:`onnx_light::core::runtime::KernelDispatchTable`,
* model-local :class:`~onnx_light.onnx_lib.FunctionProto` definitions are resolved from
  ``RuntimeContext::functions``.

Control-flow operators (``If``, ``Loop``, ``Scan``) are handled by
dedicated paths that evaluate subgraphs through
:cpp:func:`onnx_light::core::runtime::RunSubgraph`.

:cpp:func:`~onnx_light::core::runtime::RunGraph` seeds initializers and executes nodes in topological order.
:cpp:func:`~onnx_light::core::runtime::RunModel` additionally registers ``ModelProto::functions`` before evaluating
``model.graph``.

How backend tests use kernels
-----------------------------

Backend test cases in
:epkg:`onnx_light/onnx_extensions/onnx_backend_test/cases`
create ONNX nodes
and compute expected outputs with C++ kernels, then register them with
:cpp:func:`onnx_light::onnx_backend_test::Expect`.

In other words, kernels are not only used as an execution runtime; they are
also the reference implementation used to generate deterministic expected values
for the backend test suite.

Adding or extending a kernel
----------------------------

Typical workflow:

#. Implement/extend the kernel class in
   :epkg:`onnx_light/onnx_extensions/onnx_kernels/kernels`\ ``/<family>/`` and export it from the
   corresponding ``include_<family>_kernels.h`` (or, for control-flow
   style operators, under
   :epkg:`onnx_light/onnx_core/runtime/controlflow`).
#. Add or update C++ backend test cases in
   :epkg:`onnx_light/onnx_extensions/onnx_backend_test/cases`\ ``/<family>/``; compute expected outputs
   through the kernel and register them with
   :cpp:func:`onnx_light::onnx_backend_test::Expect`.
#. If the operator should be executable through
   :cpp:func:`~onnx_light::core::runtime::RunNode`/:cpp:func:`~onnx_light::core::runtime::RunModel`,
   add a trampoline entry to the built-in table in
   :epkg:`onnx_light/onnx_extensions/onnx_kernels/kernel_dispatch_table.cc`
   (it is registered with ``onnx_core`` by
   :cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions`).
#. Run the C++ tests (for example ``ctest -R OnnxOp`` or
   ``ctest -R Backend --output-on-failure`` after configuring with
   ``ONNX_LIGHT_BUILD_TESTS=ON``).

Parallelization
---------------

Kernel implementations are allowed to parallelize their computation (for
example across independent elements or rows of a tensor) when it speeds up
the operator, **except where it would change the order of floating-point
accumulation**. The C++ kernels are the reference implementation that
generates the expected values of the backend test suite, so their results
must stay bit-stable and independent of the number of threads. Parallel
floating-point accumulation is not associative and would make those
expected values depend on the thread count.

Concretely, any operator that accumulates values internally must keep a
deterministic accumulation order and therefore stay sequential on the
reduced/accumulated axis. This includes (non-exhaustively):

* the reduction operators (``ReduceSum``, ``ReduceMean``, ``ReduceProd``,
  ``ReduceMax``, ``ReduceMin``, ``ReduceL1``, ``ReduceL2``,
  ``ReduceSumSquare``, ``ReduceLogSum``, ``ReduceLogSumExp``, ``ArgMax``
  and ``ArgMin``),
* operators with hidden accumulation such as ``MatMul``, ``Gemm``,
  ``Conv``, ``Attention``, ``Einsum``, ``LRN`` and similar dot-product or
  pooling style kernels.

Operators without such accumulation (purely element-wise or
independent-row computations) may be parallelized freely.

See also
--------

* :ref:`l-design-backend-tests`
* :doc:`../api/cpp/onnx_extensions/onnx_kernels/index`
* :doc:`../api/cpp/onnx_core/runtime/index`
