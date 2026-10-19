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

Kernel resolution first checks the session's model-local functions, then
explicit context-local functions, built-in control-flow operators,
context-local custom kernels, process-wide custom kernels, and finally the
built-in :cpp:func:`KernelDispatchTable`. The selected factory creates a
session-owned kernel instance that is cached for repeated runs. Constructing,
preparing, running, resetting, or destroying a session never modifies any
process-wide function or kernel registry; global registration remains an
explicit application setup operation. Each run stores inputs, initializers,
intermediates, and outputs in a :cpp:class:`RuntimeContext`. That context also
owns the execution and output allocator routes, records events when requested,
and releases last-use intermediates when enabled.

Before kernel resolution, the execution plan registers every value name from
the root graph, nested graph attributes, and model-local functions in one
invocation-wide directory. Each unique name keeps one stable ``uint32_t`` slot.
Tensor payloads live in a vector indexed by those slots. ``RuntimeSession``
resolves each kernel's input and output slots once for a context, so built-in
kernels do not hash ONNX names while replaying the plan. The name-keyed tensor
map remains a synchronized compatibility view for external callers and custom
kernels.
The same ``RuntimeContext`` is reused by control-flow bodies and local
functions, so nested execution neither copies the context nor deep-copies
loop-carried values merely to cross a context boundary.
Operator-owned nested sessions, such as the ``score_mod`` and ``prob_mod``
graphs of ``FlexAttention``, follow the same rule: they reuse the active
context, allocator, value store, and executor while explicitly borrowing the
enclosing model-function scope.

When the caller supplies no execution allocator, the context creates a
fixed-capacity :cpp:class:`ExecutionArena` from
``ExecutionPlan::arena_capacity()``. Stable value slots do not pin physical
storage: deleting a last-use value releases its allocation handle and the arena
may reuse that ``RawBuffer`` for any later value.

Before dispatch, the session leases its resolved :cpp:class:`CpuExecutor` and
installs it on the runtime context.  Kernels then use the same session CPU
executor for serial or parallel work.  The backend-test catalog sits above
this flow: it supplies a model, inputs, expected outputs, and tolerances, while
the how-to validates the public Python path.  The lower-level C++ APIs expose
the same ``RuntimeSession`` / ``RuntimeContext`` / kernel-registry path for
native callers.

Preparation and execution
-------------------------

Direct ``ExecutionPlan`` replay prepares immutable kernel inputs synchronously
during the first ``RuntimeSession::Run``. The root session and all nested
graph/function sessions share one ``ValueStore``. Each kernel
retains a stable integer slot, so later executions read the prepared value by
direct vector indexing without a mutex, map lookup, readiness wait, or
scheduler dispatch. The store owns its prepared allocations for the complete
session-hierarchy lifetime, or longer if a context retains its shared owner.
The root owns a non-null store from construction; nested sessions explicitly
inherit that owner. Independent sessions have independent stores even when
they reuse a context. Intentional cross-session sharing uses
``RuntimeSessionOptions::value_store`` and requires serialized runs across
all sharing sessions.

The preparation ``ValueStore`` is separate from the invocation value
directory. The former owns immutable prepared kernel objects; the latter maps
ONNX value names to stable runtime slots whose physical tensor buffers remain
recyclable by the execution arena.

``RuntimeSessionOptions`` selects the CPU execution policy.
Sessions lease executors rather than creating one thread pool per kernel, so
independent sessions can share workers without nested oversubscription. See
:ref:`l-next-steps-session-execution-pools` for the executor contract.

The executor, worker lifecycle, CPU topology, affinity and threading diagnostics
are implemented by the standalone ``onnx_threads`` library. Existing runtime
headers forward to it without changing callers. See :ref:`l-design-threading`
for the dependency boundary and the distinction between sharing implementation
and sharing worker instances.

``RuntimeSession::prepared_bytes()`` sums live static-store buffer sizes,
including unpublished allocations and explicitly shared sessions; it excludes
retained free capacity and execution/I/O arenas. Prepared allocations remain
available until the last shared owner of the store is released.

.. toctree::
    :maxdepth: 1

    arenas
    test_coverage
    runtime_coverage
