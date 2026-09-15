.. _l-design-runtime:

Runtime
========

Execution flow
--------------

The runnable :ref:`backend-test walkthrough <l-howto-run-backend-test-case>`
starts at :class:`onnx_light.onnx.reference.ReferenceEvaluator`, the Python
compatibility entry point. It converts NumPy values and delegates to a reusable
C++ :cpp:class:`RuntimeSession`. The session builds an execution plan, captures
an immutable tuning-registry generation, resolves kernels, and retains prepared
kernel instances for repeated runs.

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

Preparation and execution
-------------------------

``PreparedExecutionPlan`` may prepare synchronously or submit work through a
shared execution pool. Its tasks cover payload reads, kernel creation, weight
prepacking, device copies, and publication; dependencies and memory admission
prevent a consumer from observing a partially prepared value. ``RunAsync``
uses the same task graph, while ``RunSequential`` remains the synchronous
reference.

Prepared tensors are keyed by source identity, processor/device, layout,
kernel ABI, and format version. Compatible entries bypass portable-weight
loading and prepacking. Residency is bounded, active consumers pin their
objects, and evicted entries retain enough information for a later reload.

``RuntimeSessionOptions`` selects the CPU execution policy.
``PreparedExecutionState`` owns preparation arenas, residency, and resource
limits. Sessions and prepared plans lease executors rather than creating one
thread pool per kernel, so independent sessions can share workers without
nested oversubscription. See :ref:`l-next-steps-prepared-execution`,
:ref:`l-next-steps-native-fast-loading-completion`, and
:ref:`l-next-steps-session-execution-pools` for the implemented contracts.

.. toctree::
    :maxdepth: 1

    arenas
    test_coverage
    runtime_coverage
