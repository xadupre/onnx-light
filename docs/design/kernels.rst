.. _l-design-kernels:

C++ Kernels
===========

The kernel layer is implemented in
:epkg:`onnx_light/onnx_kernels`
and built as :epkg:`lib_onnx_kernels`. It contains:

* runtime tensor/container types used by the backend runtime
  (:cpp:struct:`onnx::onnx_kernels::Tensor`,
  :cpp:struct:`onnx::onnx_kernels::Sequence`,
  :cpp:class:`onnx::onnx_kernels::RuntimeContext`),
* operator kernel implementations under
  :epkg:`onnx_light/onnx_kernels/kernels`\ ``/<domain>/``,
* graph/node execution helpers
  (:cpp:func:`onnx::onnx_kernels::RunNode`,
  :cpp:func:`onnx::onnx_kernels::RunGraph`,
  :cpp:func:`onnx::onnx_kernels::RunModel`,
  :cpp:func:`onnx::onnx_kernels::RunSubgraph`).

:epkg:`lib_onnx_backend_test` depends on this library and uses these kernels to
compute expected outputs for backend test cases.

Kernel organization
-------------------

Kernels are grouped by ONNX domain family:

* ``math``, ``logical``, ``tensor``, ``reduction``, ``nn``,
* ``controlflow``, ``sequence``, ``optional``, ``quantization``,
* ``traditionalml``, ``training``, ``image``, ``text``,
  ``object_detection``, ``preview``, ``generator``.

Each family has an ``include_<family>_kernels.h`` umbrella header exposing the
kernel classes for that group.

Runtime model
-------------

:cpp:func:`~onnx::onnx_kernels::RunNode` executes one :class:`~onnx_light.onnx_lib.NodeProto` against a
:cpp:type:`onnx::onnx_kernels::TensorMap` stored in
:cpp:class:`onnx::onnx_kernels::RuntimeContext`:

* inputs are looked up by tensor name,
* outputs are written back by output name,
* dispatch is keyed by ``(domain, op_type)``,
* model-local :class:`~onnx_light.onnx_lib.FunctionProto` definitions are resolved from
  ``RuntimeContext::functions``.

Control-flow operators (``If``, ``Loop``, ``Scan``) are handled by dedicated
paths that evaluate subgraphs through :cpp:func:`onnx::onnx_kernels::RunSubgraph`.

:cpp:func:`~onnx::onnx_kernels::RunGraph` seeds initializers and executes nodes in topological order.
:cpp:func:`~onnx::onnx_kernels::RunModel` additionally registers ``ModelProto::functions`` before evaluating
``model.graph``.

How backend tests use kernels
-----------------------------

Backend test cases in
:epkg:`onnx_light/onnx_backend_test/cases`
create ONNX nodes
and compute expected outputs with C++ kernels, then register them with
:cpp:func:`onnx::onnx_backend_test::Expect`.

In other words, kernels are not only used as an execution runtime; they are
also the reference implementation used to generate deterministic expected values
for the backend test suite.

Adding or extending a kernel
----------------------------

Typical workflow:

#. Implement/extend the kernel class in
   :epkg:`onnx_light/onnx_kernels/kernels`\ ``/<family>/`` and export it from the
   corresponding ``include_<family>_kernels.h``.
#. Add or update C++ backend test cases in
   :epkg:`onnx_light/onnx_backend_test/cases`\ ``/<family>/``; compute expected outputs
   through the kernel and register them with
   :cpp:func:`onnx::onnx_backend_test::Expect`.
#. If the operator should be executable through :cpp:func:`~onnx::onnx_kernels::RunNode`/:cpp:func:`~onnx::onnx_kernels::RunModel`, add a
   trampoline/dispatch-table entry in
   :epkg:`onnx_light/onnx_kernels/run_nodes.cc`
   (or a dedicated path for control-flow style operators).
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
* :doc:`../api/cpp/onnx_kernels/index`
