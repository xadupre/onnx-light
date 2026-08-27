.. _l-design-runtime:

Runtime Design
==============

Execution flow
--------------

The runnable :ref:`backend-test walkthrough <l-howto-run-backend-test-case>`
starts at :class:`onnx_light.onnx.reference.ReferenceEvaluator`, the Python
compatibility entry point.  It converts the supplied NumPy values and delegates
to a reusable C++ :cpp:class:`RuntimeSession`.  The session builds an execution
plan, resolves each node's kernel on its first run, and retains those kernel
instances for repeated runs.

Kernel resolution first checks model- and context-local registrations, then
the built-in :cpp:func:`KernelDispatchTable`.  Each run stores inputs,
initializers, intermediates, and outputs in a :cpp:class:`RuntimeContext`.
That context also owns the execution and output allocator routes, records
events when requested, and releases last-use intermediates when enabled.

Before dispatch, the session leases its resolved :cpp:class:`CpuExecutor` and
installs it on the runtime context.  Kernels then use the same session CPU
executor for serial or parallel work.  The backend-test catalog sits above
this flow: it supplies a model, inputs, expected outputs, and tolerances, while
the how-to validates the public Python path.  The lower-level C++ APIs expose
the same ``RuntimeSession`` / ``RuntimeContext`` / kernel-registry path for
native callers.

.. toctree::
    :maxdepth: 1

    kernel_tuning
    calibration_profile_store
    arenas
    test_coverage
    runtime_coverage
