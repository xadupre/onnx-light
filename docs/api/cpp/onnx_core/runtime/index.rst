runtime
=======

The ``runtime`` sub-namespace of ``onnx_core`` (``core::runtime``) hosts the
generic execution engine: the runtime value types
(:cpp:struct:`Tensor`, :cpp:struct:`Sequence`, :cpp:struct:`Map`),
:cpp:class:`RuntimeContext`, the node/graph/function/model traversal
(:cpp:func:`RunNode`, :cpp:func:`RunNodes`, :cpp:func:`RunGraph`,
:cpp:func:`RunFunction`, :cpp:func:`RunModel`),
random-number helpers, and low-level cast/promotion helpers shared by many
kernels.

``onnx_core`` never depends on ``onnx_kernels``, so the kernel dispatch
table starts out empty: it is a mutable registry
(:cpp:func:`RegisterKernelFn`) that ``onnx_kernels`` populates with its
per-operator trampolines (see :doc:`../../onnx_extensions/kernels/kernel_dispatch_table`)
via :cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions`. Any
consumer of the runtime (Python bindings, tests, examples, ...) must call
that function once before using :cpp:func:`RunNode` / :cpp:func:`RunModel`
or any other entry point that dispatches to a registered kernel.

Control-flow operators (``If``, ``Loop``, ``Scan``) are the one exception
to "all kernels live in ``onnx_kernels``": since running their subgraphs
recursively calls :cpp:func:`RunGraph`, which must live in ``onnx_core``,
their kernel classes live here too, under ``runtime/controlflow``, to avoid
a dependency from ``onnx_core`` back onto ``onnx_kernels``.

.. toctree::
    :maxdepth: 1

    simple_tensor
    simple_sequence
    simple_map
    runtime_context
    kernel_context
    runtime_parameters
    run_nodes
    runtime_session
    kernel_dispatch_table
    cpu_executor
    cpu_execution_policy
    kernel_tuning
    kernel_tuning_cache
    parallel_for
    tensor_compare
    node_helpers
    temporary_buffer
    random
    cast_float8
    cast_helper
    cast_sub_byte
    elementwise_helpers
    float16_promote
    controlflow/index
