.. _l-design-kernels:

C++ Kernels
===========

The kernel layer is implemented in ``onnx_light/onnx_kernels`` and built as
``lib_onnx_kernels``. It contains:

* runtime tensor/container types used by the backend runtime
  (:cpp:struct:`onnx::onnx_kernels::Tensor`,
  :cpp:struct:`onnx::onnx_kernels::Sequence`,
  :cpp:class:`onnx::onnx_kernels::RuntimeContext`),
* operator kernel implementations under
  ``onnx_light/onnx_kernels/kernels/<domain>/``,
* graph/node execution helpers
  (:cpp:func:`onnx::onnx_kernels::RunNode`,
  :cpp:func:`onnx::onnx_kernels::RunGraph`,
  :cpp:func:`onnx::onnx_kernels::RunModel`,
  :cpp:func:`onnx::onnx_kernels::RunSubgraph`).

``lib_onnx_backend_test`` depends on this library and uses these kernels to
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

``RunNode`` executes one ``NodeProto`` against a
:cpp:type:`onnx::onnx_kernels::TensorMap` stored in
:cpp:class:`onnx::onnx_kernels::RuntimeContext`:

* inputs are looked up by tensor name,
* outputs are written back by output name,
* dispatch is keyed by ``(domain, op_type)``,
* model-local ``FunctionProto`` definitions are resolved from
  ``RuntimeContext::functions``.

Control-flow operators (``If``, ``Loop``, ``Scan``) are handled by dedicated
paths that evaluate subgraphs through :cpp:func:`onnx::onnx_kernels::RunSubgraph`.

``RunGraph`` seeds initializers and executes nodes in topological order.
``RunModel`` additionally registers ``ModelProto::functions`` before evaluating
``model.graph``.

How backend tests use kernels
-----------------------------

Backend test cases in ``onnx_light/onnx_backend_test/cases/`` create ONNX nodes
and compute expected outputs with C++ kernels, then register them with
:cpp:func:`onnx::onnx_backend_test::Expect`.

In other words, kernels are not only used as an execution runtime; they are
also the reference implementation used to generate deterministic expected values
for the backend test suite.

Adding or extending a kernel
----------------------------

Typical workflow:

#. Implement/extend the kernel class in
   ``onnx_light/onnx_kernels/kernels/<family>/`` and export it from the
   corresponding ``include_<family>_kernels.h``.
#. Add or update C++ backend test cases in
   ``onnx_light/onnx_backend_test/cases/<family>/``; compute expected outputs
   through the kernel and register them with
   :cpp:func:`onnx::onnx_backend_test::Expect`.
#. If the operator should be executable through ``RunNode``/``RunModel``, add a
   trampoline/dispatch-table entry in ``onnx_light/onnx_kernels/run_nodes.cc``
   (or a dedicated path for control-flow style operators).
#. Run the C++ tests (for example ``ctest -R OnnxOp`` or
   ``ctest -R Backend --output-on-failure`` after configuring with
   ``ONNX_LIGHT_BUILD_TESTS=ON``).

See also
--------

* :ref:`l-design-backend-tests`
* :doc:`../api/cpp/onnx_kernels/index`
