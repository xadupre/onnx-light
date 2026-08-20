.. _l-next-steps-prepared-execution:

Prepared and asynchronous execution
===================================

:Date: 2026-08

**blocked by model-loading bug fixes**

Objective
+++++++++

This is step 2 of the fast-loading sequence, after
:ref:`l-next-steps-model-loading-bug-fixes` and before the explicit
:ref:`l-next-steps-model-loading` integration with onnxruntime.

The first objective is to reduce model loading time. This plan consumes the
immutable ``ResolvedModel`` contract defined by
:ref:`l-next-steps-model-resolution`. Its transformed graph, kernel choices,
prepared-object bindings, and minimal payload manifest are frozen before this
plan creates or submits any weight-read task. The first implementation may use
a deterministic resolved-model fixture; the production resolver and selective
payload integration are completed in the fourth roadmap step.

The execution plan may be submitted as soon as its structure and prepared-object
requirements are known. It does not wait for every initialization task: each
inference task becomes runnable when its ordinary inference dependencies and
its own prepared-object dependencies are ready. The task graph and scheduler
also define the asynchronous execution contract requested by
`issue #4299 <https://github.com/xadupre/onnx-light/issues/4299>`_. Ordinary
model execution remains sequential in the first implementation, but the public
API, dependency representation, errors, and cancellation must not require a
later incompatible ``run``/``run_wait`` split.

Step 2 implementation boundary
++++++++++++++++++++++++++++++

This document implements the reusable execution substrate:

* one immutable scope-aware plan and separate session/invocation state;
* dependency events, completion handles, cancellation, failure propagation,
  priorities, and memory admission;
* persistent bounded I/O resources and the existing session CPU executor;
* a sequential session-task reference path, a delayed-weight overlap test, and
  the fully prepared hot path;
* ``PrepareAsync``/``RunAsync`` with synchronous wrappers.

The production model resolver, adaptive loader policy, persisted prepared
cache, eviction/offloading policy, and device-placement variants are step 4,
:ref:`l-next-steps-native-fast-loading-completion`. They remain described here
where they constrain the core interfaces, but are not merge criteria for the
step 3 implementation.

Plan architecture
+++++++++++++++++

Decision: one scoped execution plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After ``ResolvedModel`` is frozen, initialization and inference must be
represented in **one dependency graph**. Every task declares its lifetime
scope:

.. code-block:: cpp

    enum class TaskScope {
      kSession,     // load, prepack, persist, device residency
      kInvocation,  // node execution, outputs, invocation-local scratch
    };

    struct PreparedExecutionPlan {
      std::vector<TaskDescriptor> tasks;
    };

The immutable plan contains both kinds of task. Mutable residency, completion
events, failures, and allocations belong to ``PreparedExecutionState``, not to
another plan stored beside it.

.. code-block:: text

    read B [session]
      -> prepack B [session]
        -> execute Gemm 4 [invocation]
          -> execute Add 5 [invocation]

    read W20 [session] ----------------> execute Gemm 20 [invocation]

``RunAsync`` instantiates the ``kInvocation`` tasks and binds their dependency
edges to the current generation of the ``kSession`` tasks. The first ``Gemm``
may therefore execute while ``W20`` is still loading. ``PrepareAsync`` submits
all session-scoped tasks eagerly, whereas ``RunAsync`` may submit only the
session-task closure required by that invocation.

The existing ``ExecutionPlan`` action sequence is migrated into
invocation-scoped task descriptors; it is not retained as a second plan.
Its allocation, execution, and release actions remain reusable templates, but
each inference receives its own task state and invocation-owned values.

Publishing a session object is atomic: the producing task completes only after
the immutable object and its allocation handle are visible. An inference task
depends on that completion event directly, rather than looking up a value after
a separate initialization plan has finished.

Why a scoped plan?
^^^^^^^^^^^^^^^^^^

The alternatives have the following trade-offs:

.. list-table::
    :header-rows: 1
    :widths: 18 36 46

    * - Design
      - Advantages
      - Disadvantages
    * - One unscoped combined plan
      - One end-to-end dependency graph; initialization and the first inference
        could overlap at individual-kernel granularity; one object to inspect.
      - Initialization tasks must be marked and skipped on every later run;
        session-owned and invocation-owned values coexist in one graph;
        cancellation and retry semantics become ambiguous; concurrent
        inferences need cloning or partitioning of the same graph; rebuilding
        or evicting one prepared object mutates a supposedly reusable plan.
    * - Two plans
      - Lifetimes and replay rules are simple; inference remains small and
        reusable.
      - Cross-plan events recreate a combined graph indirectly; inspection,
        cancellation, priority propagation, and critical-path scheduling must
        cross an artificial boundary.
    * - One scope-aware plan
      - Keeps the complete dependency graph, permits inference/loading overlap,
        and propagates priorities normally; task scope preserves session and
        invocation lifetimes; concurrent inferences instantiate only their
        invocation state.
      - Requires the scheduler to manage shared session-task generations,
        consumer pinning, and cancellation independently from invocation tasks.

The third design is required. The plan is one immutable graph; the distinction
between initialization and inference remains only as task scope and mutable
execution state.

Comparison with ONNX Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ONNX Runtime uses a session-owned ``SequentialExecutionPlan`` attached to
``SessionState`` and executes a run through ``utils::ExecuteGraph`` with
run-specific feeds/fetches management. In that design, the execution plan
primarily describes kernel launch order, allocation/reuse policy, release
actions, stream notifications, and synchronization.

The scoped ``PreparedExecutionPlan`` proposed here stays aligned with ONNX
Runtime on static planning and session-owned immutable metadata, while extending
the model with explicit session-scoped and invocation-scoped task descriptors
in one dependency graph. This is the main semantic difference:

* ONNX Runtime keeps initialization/finalization concerns largely around
  ``SessionState`` construction and run-time feed/fetch orchestration.
* This design represents deferred load/prepack work and regular inference tasks
  uniformly as schedulable nodes with scope-aware state.

The practical goal is compatibility of planning principles (static descriptors,
allocator-aware execution order) while enabling first-inference overlap with
background preparation without introducing a second ad hoc plan format.

Construction and execution lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prepared session is built and used in this order:

1. obtain the frozen ``ResolvedModel`` from
   :ref:`l-next-steps-model-resolution`;
2. build one ``PreparedExecutionPlan`` containing
   session- and invocation-scoped tasks and all dependencies between them;
3. optionally submit session-scoped tasks eagerly;
4. create a fresh submission state for every inference and attach each
   inference task to the completion events of only the prepared objects it
   consumes;
5. run the ready inference prefix while later selected payloads are still
    loading or being prepacked.

Plan construction must reject a mutable or incomplete resolved model. It may
create read tasks only for active entries in the frozen payload manifest.

``RunAsync`` never blocks while creating the submission and does not reject a
session merely because preparation is still running. An inference task waiting
for a prepared object stays pending while independent ready tasks continue.
When inference reaches an unavailable weight, its load/prepack chain inherits
the inference priority so background preparation or cache writes cannot starve
the critical path.

An initialization failure fails only the inference tasks that transitively
depend on the failed object, then follows ordinary submission failure
propagation. ``PrepareAsync().Wait()`` remains useful for callers that require
fully eager preparation, but it is not a prerequisite for ``RunAsync``.

Re-preparing an evicted or device-specific object creates a new initialization
generation for the corresponding session-scoped task in the same plan. Active
inference submissions bind to that generation's completion event. Concurrent
inference submissions therefore share immutable resident prepared objects and
own separate inputs, outputs, temporary buffers, cancellation state, and
errors.

Plan template and inference submissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The unified plan is a template, not mutable execution state. Every descriptor
has a stable ``TaskId`` and a scope. Runtime state is partitioned as follows:

.. code-block:: cpp

    struct PreparedExecutionState {
      std::unordered_map<TaskId, SessionTaskState> session_tasks;
    };

    struct ExecutionSubmission {
      uint64_t invocation_id;
      RuntimeContext &context;
      std::unordered_map<TaskId, InvocationTaskState> invocation_tasks;
    };

For:

.. code-block:: text

    S0: read/prepare W0 [session]
    I0: execute Gemm 0  [invocation], depends on S0
    I1: execute Add 1   [invocation], depends on I0

two calls to ``RunAsync`` produce:

.. code-block:: text

    shared:       S0@generation-1
                    |             |
    inference 1:  I0@run-1  ->  I1@run-1
    inference 2:  I0@run-2  ->  I1@run-2

``RunAsync`` performs these steps:

1. allocates a new ``invocation_id`` and invocation-local task-state table;
2. instantiates every required ``kInvocation`` descriptor for that ID;
3. rewrites invocation-to-invocation edges to tasks with the same ID;
4. binds invocation-to-session edges to the current session-task generation;
5. treats an already-resident session dependency as complete, shares an
   in-flight generation, or schedules a new generation when it was evicted;
6. submits only the newly ready invocation and session tasks.

Inference 2 therefore never reuses completion, errors, cancellation, outputs,
or temporary buffers from inference 1. It only shares immutable prepared
objects. Each submission pins a prepared object from the time its consumer
becomes dispatchable until the consuming task completes, preventing eviction
during kernel execution.

Cancelling inference 1 removes only its invocation tasks and pins. It does not
cancel ``S0`` while inference 2, an eager preparation handle, or another
consumer still references that session generation. Mutable caller state such
as a KV-cache is bound through each submission's ``RuntimeContext`` and is not
stored in the shared session-task state.

Cold and hot inference paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Streaming preparation has unavoidable cold-path costs: dependency checks,
ready-queue operations, completion events, pinning, suspension/resumption, and
possible CPU or memory-bandwidth contention with prepacking. These costs buy a
shorter time to first execution; they must not become permanent overhead after
all required objects are resident.

The first implementation keeps sequential inference as one continuation over
the existing ordered ``ExecuteAction`` sequence. It does not allocate and
enqueue one scheduler task per node:

.. code-block:: text

    run continuation:
      replay ready actions
      next node dependency missing -> register continuation and yield
      dependency completes         -> enqueue continuation once and resume

When every required session object is resident, ``PreparedExecutionState``
publishes a readiness epoch. ``RunAsync`` compares that epoch once at
submission:

* on the **cold path**, the continuation checks prepared dependencies at the
  next execution boundary and may yield;
* on the **hot path**, it pins the required resident generation and replays the
  compact action sequence directly, with no per-node event lookup, task
  allocation, or ready-queue dispatch.

The hot path must therefore have essentially the same dispatch and allocation
cost as executing the current ``ExecutionPlan`` after eager preparation. The
remaining incremental cost is one readiness-epoch check and one run-level
pin/lease operation. Benchmarks must measure and cap this steady-state overhead
separately from cold-start latency.

``Prepare().Wait()`` explicitly selects eager behavior: it completes every
required session task before the first inference, which then starts directly on
the hot path. Offloading or eviction invalidates the readiness epoch and returns
later submissions to the cold path until the required generation is resident
again.

Prepared-object residency
^^^^^^^^^^^^^^^^^^^^^^^^^

``PreparedObjectStore`` is a registry and state machine, not merely a map of
resident pointers. Each key is in one of these states:

.. code-block:: text

    absent -> loading -> preparing -> resident
       ^          |           |          |
       |          +-------- failed       +-> evicting -> persisted/absent
       +---------------------------------+

The entry owns a completion event, immutable identity, source fallback,
resident allocation handle, device, byte size, last-use information, and
optional persisted-cache location. Requests for the same key share the same
in-flight load or prepack. Eviction removes residency, not identity: the next
consumer schedules a cache read or rebuild and waits on the new completion
event.

Allocation domains
^^^^^^^^^^^^^^^^^^

Prepared execution uses the existing runtime arenas and adds two explicit
initialization domains:

* ``PreparationArena`` owns source buffers and scratch used only while reading,
  transposing, packing, hashing, or writing a cache entry; allocations return
  after the producing task completes;
* ``PreparedArena`` owns resident CPU prepared tensors for the session and
  supports retention limits and eviction;
* device allocators own resident accelerator objects and expose the same
  movable allocation-handle contract;
* ``ExecutionArena`` owns invocation-local intermediates and workspaces;
* ``IOArena`` owns final outputs and copied/staged inputs that cross the runtime
  boundary.

``PreparedObjectStore`` does not implement allocation. It owns the resulting
allocation handle and returns it to its original arena or device allocator on
eviction. The scheduler applies one global memory budget across preparation
scratch, prepared residency, execution, and I/O retention, with per-domain
limits so speculative loading cannot consume memory required by an active
inference. Disk persistence is a cache tier, not an arena.

Building from a resolved model
++++++++++++++++++++++++++++++

All kernel preparation requests and inference actions are merged into one
dependency graph. Identical session-scoped requests share one task:

.. code-block:: text

    read W0 ----> prepack W0 ----> execute kernel 0
       |
       +--------> prepack W0' ---> execute kernel 7

    read W1 ----> copy to CUDA ---> execute kernel 1

    compute small constant -------> execute kernel 2

The scheduler executes ready tasks in parallel. It uses separate queues for:

* file reads;
* CPU transformations;
* each accelerator.

It also enforces a global in-flight memory budget. Without this budget,
parallel reads and prepacks may temporarily allocate both the source and
prepared forms of every weight.

Loading and prepack overlap naturally: ``prepack W0`` starts when ``W0`` is
available while another I/O worker reads ``W1``. A task releases its source
buffer when no later task needs it. External-data mappings may instead remain
available as the portable backing store.

Asynchronous session contract
+++++++++++++++++++++++++++++

Preparation and inference both return the same moveable handle type, but select
different task scopes from the same plan:

.. code-block:: cpp

    ExecutionHandle RuntimeSession::PrepareAsync();
    ExecutionHandle RuntimeSession::RunAsync(RuntimeContext &rt);

    void RuntimeSession::Prepare() {
      PrepareAsync().Wait();
    }

    void RuntimeSession::Run(RuntimeContext &rt) {
      RunAsync(rt).Wait();
    }

The handle owns the submission state, keeps every borrowed context/resource
alive for the submission, and exposes at least:

* ``Wait()``, which returns only after every required task finishes and
  rethrows the first execution error;
* ``IsReady()``, for non-blocking completion checks;
* cooperative ``Cancel()``, which prevents tasks that have not started from
  running and requests cancellation from operations that support it;
* final status and diagnostics, including the failing task and suppressed
  downstream tasks.

Destroying a live handle must have one documented behavior. The initial
contract should wait rather than detach silently, because ``RuntimeContext`` and
caller-owned inputs may otherwise be destroyed while tasks still reference
them. A later detached API requires owned inputs and a session-owned result
object.

``Run`` stays the compatibility entry point and is exactly synchronous
``RunAsync`` followed by ``Wait``. A separate ``run_wait`` method would expose
two partially overlapping execution paths and leave ownership between the calls
undefined.

Task resources and scheduling
+++++++++++++++++++++++++++++

Both task scopes use explicit resource classes:

* bounded I/O workers for external-data reads and persisted-prepack reads;
* CPU workers for transforms, packing, and CPU kernels;
* one queue per accelerator or asynchronous device stream;
* an inline resource for trivial tasks whose dispatch cost exceeds their work.

Ready tasks may run when every dependency has completed successfully and the
resource and in-flight-memory budgets admit them. Failure cancels dependent
tasks but does not abandon already-running independent work without joining it.
Cancellation and failure therefore reach one terminal submission state before
``Wait`` returns.

Initialization and inference use the same plan and scheduler, but have different
task scopes and execution state. Session-scoped tasks produce prepared objects.
Invocation-scoped tasks consume caller inputs and produce invocation-owned
outputs. The scope prevents a loading task from being replayed on every
inference and prevents one invocation from mutating another invocation's state.
A cancelled inference releases its interest in shared session tasks; it cancels
one of those tasks only when no eager preparation handle or other inference
still consumes it.

CPU executor ownership
^^^^^^^^^^^^^^^^^^^^^^

The runtime scheduler must lease the session's resolved ``CpuExecutor`` rather
than create a second CPU worker pool. CPU preparation tasks, invocation tasks,
and kernel parallel regions all execute through that lease. A task already
running on an executor participant installs the same executor view, so nested
kernel work runs inline instead of waking unrelated workers.

Bounded I/O workers remain a separate resource because blocking reads must not
occupy CPU compute participants. Diagnostics carry the process-local executor
instance identifier exposed by runtime events, while persisted prepared objects
and tuning caches use the behavior-only executor key; the instance identifier
is never persistent compatibility metadata.

Relationship with the ``onnx_proto`` thread pool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The existing :cpp:class:`utils::ThreadPool` must not be used directly as the
runtime scheduler. It is designed for one parse or serialization batch:
``Start`` creates a FIFO worker set, ``Wait`` drains it and stops the workers,
and the pool has no dependency events, priorities, resource classes,
cancellation, or persistent submissions. Sharing that object would force
inference to wait for parsing and would permit background cache writes to
starve kernel work.

The reusable part should instead be extracted below both users:

.. code-block:: text

    WorkerPool
      persistent workers, enqueue, shutdown
          |
          +-- onnx_proto ThreadPool adapter
          |     batch Start/SubmitTask/Wait semantics
          |
          `-- RuntimeScheduler
                dependency events, priorities, cancellation,
                I/O/CPU/device queues, memory admission

``onnx_proto`` keeps its current public API. The runtime owns a persistent
``RuntimeScheduler`` for the prepared session or receives one from the caller.
It uses bounded I/O workers separately from CPU workers and device queues.

When an inference task waits for a prepared object, the object's producer chain
inherits its priority. At least one CPU worker and one I/O admission slot must
remain available to critical-path work; otherwise speculative prepacking could
occupy every worker or all in-flight memory and deadlock the inference that is
supposed to release memory.

Pipelining loading and computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no initialization barrier before inference. The scheduler continuously
dispatches ready tasks from separate resource queues:

.. code-block:: text

    I/O workers:       read W0 | read W1 | read W2 | ... | persist packed W0
                           |         |         |
    CPU preparation:  prepack W0  prepack W1  prepack W2
                           |         |         |
    inference:        execute L0  execute L1  execute L2

As soon as ``prepack W0`` publishes its object, ``execute L0`` becomes ready.
While ``L0`` computes, the I/O queue continues reading ``W1`` and ``W2``.
There is no ``Wait`` between those operations; only the dependency from one
layer to the exact prepared objects it consumes.

The scheduler uses three priority classes:

* ``critical``: inference tasks and the load/prepack/upload chain currently
  blocking an inference task;
* ``prefetch``: preparation inside a bounded look-ahead window on the inference
  critical path;
* ``background``: preparation outside that window, cache persistence, and
  optional variants.

Priority inheritance promotes a prefetched producer to ``critical`` when an
inference reaches it. Cache writes never delay reads required by inference.

The look-ahead window is bounded by both graph distance and bytes in flight.
For a sequential transformer it normally prepares the next few layers, not the
complete model. It grows while I/O is slower than computation and shrinks when
prepared residency approaches its budget. Branches prefetch the union of the
nearest reachable requirements; unsupported dynamic control flow loads on
demand.

CPU prepacking and CPU inference compete for cores even though I/O does not.
The runtime therefore reserves inference capacity and lets preparation consume
only the remaining CPU slots. A practical default is one latency-critical CPU
queue plus a bounded preparation queue; throughput-oriented callers may allow
both to share the full worker set. Device copies and kernels use device events
and streams, so a copy may overlap a kernel only when the backend and device
support it.

Memory admission happens before dispatch. A read is delayed when its source
buffer plus expected prepared output would exceed the preparation/residency
budget. This prevents aggressive prefetch from evicting the weight currently
used by inference or from exhausting the ``ExecutionArena`` and ``IOArena``
budgets.

Prepacking
++++++++++

A prepack task turns a source weight into the physical representation a kernel
consumes at run time. Source bytes are never modified: the graph continues to
reference the transformed model's logical value, while its portable or derived
recipe remains the fallback. The prepacked object is a derived, kernel- and
device-specific value.

Prepacked object contract
^^^^^^^^^^^^^^^^^^^^^^^^^^

A prepack task produces a prepared object described by:

.. code-block:: cpp

    struct PrepackRequest {
      std::string logical_value;
      std::vector<SourceIdentity> source_lineage;
      std::string layout;        // requested packed layout key
      int32_t device;            // index into ModelProto.devices
    };

    struct PrepackedWeight {
      PrepackRequest request;    // identity of the prepared object
      // opaque, kernel-owned packed representation
    };

The ``layout`` key captures every parameter that changes the packed bytes, such
as the ``transB`` flag of ``Gemm``, a block size, or a tiling shape. Two kernels
that request the same ``(source_lineage, layout, device)`` share one prepacked
object; two kernels that disagree on any of these fields receive distinct
objects. This is the deduplication rule illustrated by the
``read W0 -> prepack W0`` / ``prepack W0'`` fan-out of Step 6.

Because a ``PrepackRequest`` is a pure function of its ordered source lineage
and layout key, it also defines the cache identity used for persistence.

Reusing a persisted prepack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prepacking is often linear but not free, and it repeats on every load. The
first implementation does **not** depend on the proposed
:ref:`l-next-steps-compiled-tensor` format: ``CompiledTensorProto`` is not
implemented and is not required for prepared execution.

A companion ONNX model stores each packed representation as a standard
``TensorProto``:

.. code-block:: text

    TensorProto {
      name: "prepared/<prepared-key>"
      data_type: UINT8
      dims: [packed_byte_size]
      data_location: EXTERNAL
      external_data: { location, offset, length, checksum }
      metadata_props: {
        "onnx_light.prepack.source_lineage_digest": ...
        "onnx_light.prepack.source_lineage": ...
        "onnx_light.prepack.layout": ...
        "onnx_light.prepack.device": ...
        "onnx_light.prepack.kernel_abi": ...
        "onnx_light.prepack.format_version": ...
      }
    }

The packed bytes are opaque to generic ONNX execution. ``UINT8`` only provides
a standard byte container; the metadata identifies the kernel-specific physical
format. A format requiring several buffers may use one small container format
inside the byte payload or several initializers sharing one prepared-entry ID.

Prepared execution uses these tensors through a sidecar cache rather than
rewriting the portable model during inference:

.. code-block:: text

    cache root /
      model identity /
        source digest /
          layout + device ABI + runtime version -> TensorProto payload

The model identity only partitions lookup; correctness comes from
``source_lineage_digest``, layout, device compatibility, packed-format version,
payload checksum, and kernel/runtime ABI.
The cache may be a companion ONNX ``ModelProto`` or be embedded in an
explicitly compiled copy of the source model. Ordinary loading and offloading
never mutate the portable source model.

Prepacking reuses that cache instead of introducing a second format:

* the cache key is ``(source_lineage_digest, layout, device)``, where
  ``source_lineage_digest`` covers the ordered identities, element types,
  dimensions, and canonical content of every source used to derive the
  prepared object, while ``layout``/``device`` come from the
  ``PrepackRequest``;
* on a cache hit with compatible device, format, and kernel ABI metadata, the
  scheduler replaces the ``prepack`` task with a read or memory map of the
  cached bytes, followed by a device upload when required, and skips the
  packing work;
* on a cache miss, or when the persisted ``source_lineage_digest`` no longer
  matches the current transformed sources, the prepack task runs normally,
  publishes the resident object immediately, and schedules a lower-priority
  cache write;
* an incompatible runtime or device ignores the cached value and falls back to
  the portable initializer.

``CompiledTensorProto`` may later replace this metadata convention if a stable,
shared structured-physical-value format is adopted. That migration changes the
cache serialization only; it does not change plan tasks, scheduling, residency,
or allocator ownership.

Cache writes use a temporary file, flush and validate the complete record, then
publish it with an atomic rename. Cancellation before publication leaves no
cache hit, and a corrupt or partial entry is removed or ignored. Concurrent
sessions racing on the same key may duplicate computation, but only a complete
compatible record becomes visible.

Demand-driven reuse in a later session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sidecar separates a small index from the packed payloads:

.. code-block:: text

    manifest
      model identity
      source name -> source digest, type, dimensions
      prepared key -> source lineage and digest, payload path, layout,
                      device ABI, size, checksum

    payload files
      packed bytes, loaded only when their consumer approaches execution

At the next model load, the runtime reads only model metadata and this manifest.
It first builds ``WeightDescriptor`` objects, applies the transformation
pipeline, and freezes ``ResolvedModel`` without materializing initializer
payloads. It then builds plan tasks from the resolved payload manifest. For
every prepared-object requirement:

1. if the manifest has a compatible entry, the plan uses ``read packed
   payload -> optional device upload`` and never reads the portable source
   tensor;
2. if no entry exists, it keeps ``read source -> prepack -> publish`` as a
   demand-driven task;
3. if no reachable node requests an initializer, no task reads either its
   source bytes or any cached payload;
4. prefetch submits only entries inside the bounded look-ahead window, so later
   layers remain metadata-only until inference approaches them.

Every component source digest, and therefore the aggregate lineage digest, must
be available without rereading source payloads. Preferred sources are checksums
carried by external-data metadata or a trusted content-addressed model-package
manifest. The first successful source read may compute a missing digest and
persist it in the sidecar for that immutable model identity. File path,
modification time, offset, and length may accelerate change detection but are
not sufficient cache identity for correctness.

When any trusted source digest is unavailable, preliminary resolution treats
the cache entry as unverified and selects the portable source/prepack recipe.
It does not read and hash a weight merely to decide the manifest. The later
source-read task computes and persists the missing digest while rebuilding the
prepared object. File timestamps must not be trusted as cache identity.
Packaging models with external-weight digests is therefore required to obtain
both strict validation and zero source-weight reads on the next session.

After a cache miss, publishing the resident object unblocks inference
immediately. A background task then serializes the packed representation and
atomically adds the manifest entry. The next inference in the same session uses
the resident allocation; after eviction or process restart, it loads the
packed payload directly.

Companion prepared ONNX model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sidecar index may itself be a second ONNX ``ModelProto`` supplied alongside
the portable model. This *prepared model* is a cache container, not another
graph to execute. It contains:

* the identity or digest of the portable source model;
* model metadata declaring the prepared-cache format version;
* one ``UINT8`` graph initializer per persisted prepared representation, with
  source, layout, device, format, and ABI information in ``metadata_props``;
* external-data descriptors for large packed payloads.

Packed payloads must normally remain external. The prepared model is parsed
with ``skip_raw_data=true`` so passing it to the plan builder does not eagerly
load every cached tensor.

The API makes cache behavior explicit:

.. code-block:: cpp

    enum class PreparedModelPolicy {
      kDisabled,        // ignores any prepared model and never writes one
      kReadOnly,        // uses compatible entries; builds misses only in memory
      kBuildIfMissing,  // creates/updates a companion model and persists misses
      kRequireComplete, // fails when a required compatible entry is absent
    };

    struct PreparedModelOptions {
      const ModelProto *model = nullptr;
      PreparedModelPolicy policy = PreparedModelPolicy::kDisabled;
      std::string output_path;
    };

    PreparedExecutionPlan BuildPreparedExecutionPlan(
        ResolvedModel resolved);

``ResolveModel`` and ``ModelResolutionOptions`` are defined by
:ref:`l-next-steps-model-resolution`; this plan only consumes their frozen
result.

``kBuildIfMissing`` is the explicit answer to whether the second model should
be constructed. When ``model`` is null, it creates a new companion model; when
it is present, it preserves valid entries and adds or replaces missing,
incompatible, or stale entries. ``kReadOnly`` never modifies or creates the
companion model. ``kRequireComplete`` is intended for deployments where
runtime prepacking is forbidden; it validates requirements while building the
plan and fails before inference when the prepared model is incomplete.

The plan builder consumes only cache metadata and creates demand-driven
``read packed payload`` tasks. ``RuntimeSession`` owns or receives a shared
``PreparedModelStore`` that keeps the companion model and external-data reader
alive, resolves payload descriptors, and accepts completed prepack writes under
``kBuildIfMissing``. ``ResolvedModel`` owns or shares the source and prepared
stores needed by its descriptors so they outlive the plan. The prepared model
should not be passed independently to resolution, plan construction, and the
runtime as unrelated objects, because they could observe different cache
generations.

Updating a prepared model never rewrites it in place. Packed payloads are
written to temporary external-data files and atomically published first; a new
prepared ``ModelProto`` manifest is then serialized and atomically replaces the
old manifest. An interrupted update therefore leaves either the previous
complete generation or the new complete generation visible.

The source-model digest, complete source lineage and aggregate digest, layout,
packed-format version, device description, kernel/runtime ABI, payload size,
and payload checksum are validated before an entry is accepted. Structural
corruption is an error. Ordinary incompatibility is a cache miss under
``kReadOnly`` or ``kBuildIfMissing`` and an error under
``kRequireComplete``.

For offloading, eviction first checks whether a valid persisted entry exists.
If it does, resident CPU or device storage can be released immediately. If it
does not, policy chooses between:

* asynchronously persisting the prepared bytes before eviction;
* dropping the object and rebuilding it later from the portable initializer;
* refusing eviction while an active inference still pins the object.

An inference that needs an evicted object schedules ``cache read -> optional
device upload -> publish`` and waits only at the consuming node. Cache reads
needed by active inference have higher priority than speculative reads and
background writes.

The scheduler therefore treats a cached prepack as an I/O task followed
optionally by a device task, and an uncached prepack as an I/O task followed by
a CPU or accelerator task and an independent background write. The dependency
graph, memory budgets, and deduplication rule are unchanged; only the source and
residency of the packed bytes differ.

Kernel and device
+++++++++++++++++

A kernel implementation should remain attached to one **execution device**.
A CUDA kernel produces CUDA outputs and participates in a CUDA execution
schedule. Treating it simultaneously as a CPU and CUDA kernel would make
placement, output ownership, and transfer insertion ambiguous.

However, its **initialization tasks** may use several resources. For example:

.. code-block:: text

    CUDA Gemm kernel
      execution device: CUDA

      initialization:
        read B                 -> I/O
        transpose/pack B       -> CPU
        upload packed B        -> CUDA

or:

.. code-block:: text

    CUDA Gemm kernel
      execution device: CUDA

      initialization:
        read B                 -> I/O
        upload B               -> CUDA
        pack B                 -> CUDA

The CUDA kernel may therefore use the CPU during initialization without
becoming a CPU execution kernel. The kernel chooses between CPU packing and
CUDA packing according to the available implementation and reports the
corresponding task graph.

If an operator can execute entirely on either CPU or CUDA, it has two kernel
implementations. The session selects one execution kernel for the current
placement:

.. code-block:: text

    GemmCpu   -> execution device CPU  -> CPU packed weight
    GemmCuda  -> execution device CUDA -> CUDA packed weight

They may share the same source ``WeightDescriptor`` and the same read task, but
their prepared objects are distinct.

Multiple devices
++++++++++++++++

The first implementation should assign one execution device to every node
before kernel initialization. Kernel selection then produces the initialization
tasks for that fixed placement. This keeps the first version simple while
already allowing all I/O, CPU preparation, and accelerator preparation to
overlap.

Supporting a placement that changes later requires retaining one prepared
variant per ``(node, execution device)``. Changing placement selects another
kernel and may trigger its missing initialization tasks. This is an extension
of the same plan, not a reason to make one kernel multi-device.

Offloading between inference iterations is outside the initial loading plan.
It should be implemented as a residency policy over already defined CPU and
accelerator kernel variants. The portable source weight remains addressable,
and prepared forms may be cached or evicted independently. This can be added
after fixed-placement parallel initialization works.

Tiny INT4 LLM example
+++++++++++++++++++++

Consider a small decoder-only model with four transformer blocks:

.. code-block:: text

    hidden size:       512
    intermediate size: 2048
    vocabulary:        8192
    blocks:            4
    weight format:     grouped INT4, group size 32
    execution:         CPU AVX2 MatMulNBits kernels

One block contains QKV, attention output, gate/up, and down projections. Its
INT4 weights plus FP16 group scales occupy approximately 2.3 MB; the exact
packed size depends on the kernel layout. The portable model stores those
weights as external data:

.. code-block:: text

    tiny-llm.onnx
    tiny-llm.data
      layer0.qkv.int4
      layer0.attention_out.int4
      layer0.gate_up.int4
      layer0.down.int4
      ...
      layer3.down.int4

The scoped plan contains session and invocation dependencies such as:

.. code-block:: text

    read layer0.qkv [session, I/O]
      -> prepack layer0.qkv [session, CPU]
        -> execute layer0 attention [invocation, CPU]

    read layer0.gate_up [session, I/O]
      -> prepack layer0.gate_up [session, CPU]
        -> execute layer0 MLP [invocation, CPU]

    execute layer0 MLP
      -> execute layer1 attention

Concrete execution-plan representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The plan stores immutable descriptors. A descriptor contains enough
information for dependency scheduling, resource admission, memory accounting,
and direct hot-path replay:

.. code-block:: cpp

    struct TaskDescriptor {
      TaskId id;
      TaskScope scope;
      TaskKind kind;
      ResourceClass resource;
      std::vector<TaskId> dependencies;
      std::vector<PreparedKey> prepared_requirements;
      std::optional<PreparedKey> publishes;
      size_t estimated_input_bytes;
      size_t estimated_output_bytes;
      size_t peak_temporary_bytes;
      std::optional<ActionRange> actions;
    };

``actions`` identifies a range in the existing compact ``ExecuteAction`` array
for an invocation step. It is absent for session-scoped tasks. Session
descriptors instead publish a ``PreparedKey``.

For one packed weight, the plan also records alternative materialization
recipes:

.. code-block:: cpp

    struct PreparedRequirementDescriptor {
      PreparedKey key;
      std::vector<MaterializationRecipe> recipes;
    };

    // Ordered from cheapest to most expensive.
    recipes = {
      ReadPackedPayload{prepared_model_descriptor},
      ReadSourceAndPrepack{weight_descriptor, kernel_layout},
    };

Only a compatible recipe is eligible. A cache hit selects
``ReadPackedPayload``. A miss selects ``ReadSourceAndPrepack`` and may append a
background persistence task. Every recipe completes by publishing the same
``PreparedKey``, so inference dependencies do not change when cache residency
changes.

For block 0 with no compatible cache entries, the selected recipes expand to
this illustrative plan dump. ``W`` denotes each of ``qkv``, ``attn_out``,
``gate_up``, and ``down``; expansion creates one concrete task per value of
``W``:

.. list-table::
    :header-rows: 1
    :widths: 19 12 13 26 30

    * - Task
      - Scope
      - Resource
      - Dependencies
      - Result
    * - ``S.emb.read_source``
      - session
      - I/O
      - weight descriptor
      - source embedding bytes
    * - ``S.emb.prepack``
      - session
      - CPU
      - ``S.emb.read_source``
      - ``P.emb``
    * - ``S.l0.W.read_source``
      - session
      - I/O
      - weight descriptor for ``W``
      - source INT4 bytes for ``W``
    * - ``S.l0.W.prepack``
      - session
      - CPU
      - ``S.l0.W.read_source``
      - ``P.l0.W.avx2``
    * - ``S.l0.W.persist``
      - session
      - I/O
      - ``S.l0.W.prepack``
      - companion-model payload; not required by inference
    * - ``I.embedding``
      - invocation
      - CPU
      - input tokens, ``P.emb``
      - hidden state 0
    * - ``I.l0.attention``
      - invocation
      - CPU
      - hidden state 0, ``P.l0.qkv.avx2``,
        ``P.l0.attn_out.avx2``, KV-cache
      - attention state 0
    * - ``I.l0.mlp``
      - invocation
      - CPU
      - attention state 0, ``P.l0.gate_up.avx2``,
        ``P.l0.down.avx2``
      - hidden state 1

On a cache hit, the three ``read_source -> prepack -> persist`` descriptors for
one ``W`` are replaced by one ``S.l0.W.read_packed`` I/O descriptor that
publishes the same ``P.l0.W.avx2`` key.

Blocks 1--3 repeat the four prepared requirements and two invocation steps. The
tail contains final normalization and LM-head requirements:

.. code-block:: text

    I.embedding
      -> I.l0.attention -> I.l0.mlp
      -> I.l1.attention -> I.l1.mlp
      -> I.l2.attention -> I.l2.mlp
      -> I.l3.attention -> I.l3.mlp
      -> I.final_norm -> I.lm_head -> I.logits

Each invocation step also depends on its own prepared requirements. For
example, the complete dependency expression for layer 1 attention is:

.. code-block:: text

    ready(I.l1.attention@run-N) =
        complete(I.l0.mlp@run-N)
        and resident(P.l1.qkv.avx2@generation-G)
        and resident(P.l1.attn_out.avx2@generation-G)
        and bound(KV-cache@run-N)

The plan precomputes:

* the invocation order above;
* one bit index for every prepared requirement;
* the requirement bitset needed by every invocation step;
* last-use positions for activations and prepared-object pins;
* the prefetch distance and estimated bytes for every session requirement;
* direct ``ExecuteAction`` ranges used by the hot path.

The mutable session state may then look like:

.. code-block:: text

    P.emb                    resident generation 1
    P.l0.qkv.avx2            resident generation 1
    P.l0.attn_out.avx2       preparing generation 1
    P.l0.gate_up.avx2        loading generation 1
    P.l0.down.avx2           absent
    P.l1.*                   absent
    ...

``RunAsync`` creates only invocation state:

.. code-block:: text

    ExecutionSubmission run-1
      program_counter: I.embedding
      ready_requirements: P.emb, P.l0.qkv.avx2
      RuntimeContext: token input, KV-cache binding, arenas, outputs
      status: runnable

The continuation executes ``I.embedding`` and then yields before
``I.l0.attention`` until ``P.l0.attn_out.avx2`` is published. Meanwhile the I/O
and preparation pools continue materializing the later requirements.

After every requirement becomes resident, a second submission uses the same
plan without session work:

.. code-block:: text

    ExecutionSubmission run-2
      readiness_epoch: matches session epoch
      program_counter: I.embedding
      mode: hot

    one run-level pin
      -> replay all precomputed ExecuteAction ranges
      -> release pin

If ``P.l2.qkv.avx2`` was evicted before ``run-3``, only its readiness bit is
clear. The continuation runs through block 1, while the selected
``ReadPackedPayload`` recipe restores that object, and yields at block 2 only
if the read has not completed.

The descriptor's memory fields also determine allocator admission:

.. code-block:: text

    S.* source/prepack temporary  -> PreparationArena
    S.* published P.*             -> PreparedArena
    I.* intermediate/action       -> ExecutionArena
    I.logits                      -> IOArena
    KV-cache                      -> caller-owned mutable binding

Arena memory along the concrete plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following accounting is illustrative, not a benchmark. It uses these
explicit assumptions:

.. code-block:: text

    one block, portable INT4 + FP16 scales:       2.30 MiB
    one block, packed AVX2 representation:         2.50 MiB
    maximum packing scratch for one block:         0.50 MiB
    tied INT4 embedding / LM-head source:           2.25 MiB
    tied embedding / LM-head packed representation: 2.50 MiB
    hidden activation [1, 128, 512] FP16:           0.125 MiB
    maximum attention/MLP execution workspace:      0.625 MiB
    final last-token logits [1, 8192] FP16:          0.016 MiB
    PreparedArena residency limit:                  5.00 MiB

The table reports **live bytes**, with preparation values being the peak during
the row. Each ``prepare Ln`` row aggregates the four packed requirements of one
block; the actual scheduler can release each source projection after its own
prepack.

.. list-table::
    :header-rows: 1
    :widths: 9 25 14 14 14 12 12

    * - Step
      - Plan work
      - ``PreparationArena``
      - ``PreparedArena``
      - ``ExecutionArena``
      - ``IOArena``
      - Residency action
    * - 0
      - Parse both model manifests and build descriptors
      - 0
      - 0
      - 0
      - 0
      - None
    * - 1
      - ``S.emb.read_source -> S.emb.prepack``
      - 2.75 MiB
      - 2.50 MiB
      - 0
      - 0
      - Publish ``P.emb``
    * - 2
      - ``S.l0.*.read_source -> S.l0.*.prepack``
      - 2.80 MiB
      - 5.00 MiB
      - 0
      - 0
      - Publish ``P.l0.*``
    * - 3
      - ``I.embedding@run-1``
      - 0
      - 5.00 MiB
      - 0.125 MiB
      - 0
      - Unpin and evict ``P.emb`` after its last use
    * - 4
      - Execute ``I.l0.*`` while preparing ``S.l1.*``
      - 2.80 MiB
      - 5.00 MiB
      - 0.750 MiB
      - 0
      - Keep L0 pinned; publish L1, then evict L0
    * - 5
      - Execute ``I.l1.*`` while preparing ``S.l2.*``
      - 2.80 MiB
      - 5.00 MiB
      - 0.750 MiB
      - 0
      - Keep L1 pinned; publish L2, then evict L1
    * - 6
      - Execute ``I.l2.*`` while preparing ``S.l3.*``
      - 2.80 MiB
      - 5.00 MiB
      - 0.750 MiB
      - 0
      - Keep L2 pinned; publish L3, then evict L2
    * - 7
      - Execute ``I.l3.*`` while restoring tied LM-head prepack
      - 0
      - 5.00 MiB
      - 0.750 MiB
      - 0
      - Read packed LM-head directly into ``PreparedArena``
    * - 8
      - Evict L3, then ``I.lm_head -> I.logits``
      - 0
      - 2.50 MiB
      - 0.125 MiB
      - 0.016 MiB
      - Keep LM-head resident for output computation
    * - 9
      - Complete ``run-1`` and prefetch L0 for the next token
      - 0
      - 5.00 MiB
      - 0
      - 0.016 MiB
      - LM-head + L0 resident

At step 4, for example, the approximate live total managed by these arenas is:

.. code-block:: text

    PreparationArena  2.80 MiB
    PreparedArena     5.00 MiB
    ExecutionArena    0.75 MiB
    IOArena           0.00 MiB
                       --------
    total             8.55 MiB

The mutable KV-cache is deliberately outside those four arenas. With 4 layers,
8 heads, head dimension 64, 128 tokens, FP16 keys and values, its live
caller-owned size is:

.. code-block:: text

    4 layers * 2 (K,V) * 128 tokens * 8 heads * 64 * 2 bytes = 1.00 MiB

Live bytes are not the complete process footprint because arenas retain freed
capacity. After step 9, one possible accounting is:

.. list-table::
    :header-rows: 1
    :widths: 25 18 18 39

    * - Arena
      - Live
      - Retained free
      - Explanation
    * - ``PreparationArena``
      - 0
      - up to 2.80 MiB
      - Packing scratch/source capacity retained unless trimmed or capped
    * - ``PreparedArena``
      - 5.00 MiB
      - 0
      - LM-head and prefetched block 0
    * - ``ExecutionArena``
      - 0
      - up to 0.75 MiB
      - Activation/workspace capacity retained for the next invocation
    * - ``IOArena``
      - 0.016 MiB
      - 0
      - Logits remain leased by the caller

The corresponding arena-reserved total may therefore be 8.566 MiB, plus the
1.00 MiB caller-owned KV-cache, scheduler metadata, model metadata, thread
stacks, mapped files, and allocator alignment. Per-arena retention caps may
lower the retained 3.55 MiB without changing live execution requirements.

First inference without a prepared model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With a two-block look-ahead window, loading, prepacking, and inference overlap:

.. code-block:: text

    I/O workers       read L0 ---- read L1 ---- read L2 ---- read L3
                          |            |            |            |
    CPU preparation  prepack L0 -- prepack L1 -- prepack L2 -- prepack L3
                          |            |            |            |
    CPU inference     execute L0 -- execute L1 -- execute L2 -- execute L3

Inference starts when the embedding and first block requirements are resident;
it does not wait for blocks 1--3. While block 0 executes, I/O workers read block
1 and block 2. CPU preparation uses worker capacity not reserved for inference.

Each completed prepack first publishes its resident object, unblocking its
consumer. A background task then writes an external payload and adds a standard
``UINT8`` initializer to the companion model:

.. code-block:: text

    TensorProto {
      name: "prepared/layer0/qkv/avx2-block32"
      data_type: UINT8
      dims: [packed_byte_size]
      data_location: EXTERNAL
      external_data: {
        location: "tiny-llm.prepared.data"
        offset: ...
        length: ...
        checksum: ...
      }
      metadata_props: {
        "onnx_light.prepack.source_lineage_digest": "..."
        "onnx_light.prepack.source_lineage": "layer0.qkv.weight"
        "onnx_light.prepack.layout": "matmul-nbits-avx2-block32"
        "onnx_light.prepack.device": "cpu:x86_64:avx2"
        "onnx_light.prepack.kernel_abi": "onnx-light-cpu-v1"
        "onnx_light.prepack.format_version": "1"
      }
    }

The allocation domains are:

.. list-table::
    :header-rows: 1
    :widths: 35 35 30

    * - Value
      - Owner
      - Lifetime
    * - Source INT4 bytes and packing scratch
      - ``PreparationArena``
      - Read/prepack task
    * - AVX2 packed weights
      - ``PreparedArena``
      - Resident session generation
    * - Activations and kernel workspace
      - ``ExecutionArena``
      - Inference/last use
    * - Logits
      - ``IOArena``
      - External output owner
    * - KV-cache
      - Mutable caller binding
      - Across token invocations

Second inference in the same session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If all four packed blocks remain resident, the readiness epoch is valid. The
second ``RunAsync`` takes the hot path, pins that resident generation once, and
replays the compact inference action sequence:

.. code-block:: text

    epoch check -> execute L0 -> execute L1 -> execute L2 -> execute L3

It performs no source-weight read, prepack, per-node scheduler dispatch, or
packed-cache read. It creates new invocation-local activations and logits while
sharing the immutable packed weights. The caller binds the updated KV-cache
separately for this token.

New process with the companion model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next process supplies both ONNX models:

.. code-block:: cpp

    PreparedModelOptions prepared{
        &prepared_model,
        PreparedModelPolicy::kBuildIfMissing,
        "tiny-llm.prepared.onnx",
    };
    ModelResolutionOptions resolution{
        StandardRuntimeTransformations(),
        prepared,
    };
    ResolvedModel resolved = ResolveModel(portable_model, resolution);
    PreparedExecutionPlan plan =
        BuildPreparedExecutionPlan(std::move(resolved));

Both models are parsed metadata-only, transformations and cleanup run, and the
minimal payload manifest is frozen. For a compatible entry, the plan uses:

.. code-block:: text

    read tiny-llm.prepared.data -> publish packed object -> execute layer

It does not read the corresponding INT4 source bytes and does not run the AVX2
prepack. Payloads for later blocks remain on disk until they enter the prefetch
window.

If the block 2 cache entry is absent or incompatible, only block 2 uses:

.. code-block:: text

    read layer2 INT4 source -> AVX2 prepack -> publish -> background persist

Blocks 0, 1, and 3 still use their persisted packed payloads. An initializer
not consumed by any reachable node has neither its portable bytes nor a cached
payload loaded.

Offloading example
^^^^^^^^^^^^^^^^^^

Assume ``PreparedArena`` has a 5 MB residency limit, enough for approximately
two packed blocks. As inference advances, the scheduler pins the current block,
prefetches the next block, and evicts an unpinned older block whose packed
payload is already persisted:

.. code-block:: text

    resident before L1: L0, L1
    execute L1 / load L2
    evict unpinned L0
    resident before L2: L1, L2

On a later token, block 0 is restored directly from
``tiny-llm.prepared.data``. The runtime does not read ``layer0.*.int4`` and does
not repeat prepacking. Cache reads needed by the active token have critical
priority; background cache writes cannot delay them.

Benchmark
+++++++++

The benchmark must start from a valid serialized model and must not modify its
graph or synthesize weights outside the configured preliminary transformation
pipeline. The current Qwen3-like backend fixture contains metadata-only
initializers, so materializing random weights, inlining functions, and deleting
``value_info`` before resolution changes the workload. It may remain a session
microbenchmark, but it is not the loading benchmark for this work.

Resolution is benchmarked separately by
:ref:`l-next-steps-model-resolution`. The prepared-execution benchmark starts
from its frozen output and uses a deterministic model with real external
weights. It measures:

* execution-plan and task-graph construction;
* weight reads;
* prepack and device transfers;
* time to first runnable inference node and time to first output;
* total time until every initialization task is ready;
* peak memory and maximum bytes in flight.

It should compare sequential and parallel execution of exactly the same set of
session-scoped tasks.

The asynchronous execution benchmark additionally measures submission latency,
per-node time spent waiting for prepared objects, overlap between loading,
prepacking and inference, ready-queue delay, cancellation latency, and
end-to-end completion. It must verify that
``RunAsync(...).Wait()`` and ``Run(...)`` produce identical values, errors,
release order, and allocator ownership.

Implementation order
++++++++++++++++++++

#. Add a valid external-data model benchmark that is never rewritten by the
   benchmark.
#. Consume a deterministic frozen ``ResolvedModel`` fixture conforming to
   :ref:`l-next-steps-model-resolution` and reject every read not present in its
   active payload manifest.
#. Extract a persistent ``WorkerPool`` below the current ``onnx_proto``
   ``ThreadPool`` API, then add the common task, dependency, resource,
   completion, scope, and diagnostic types used by the unified runtime plan;
   keep the existing ``ExecutionPlan`` behavior unchanged during migration.
#. Add ``PreparedExecution``, the residency-state ``PreparedObjectStore``,
   ``PreparationArena``, ``PreparedArena``, and the explicit
   prepared-object requirement/publish contract.
#. Expand selected materialization recipes into load, prepack, publish, and
   dormant-fallback task descriptors.
#. Implement initialization tasks for one CPU ``Gemm`` with a constant ``B``,
   including both ``transB`` values.
#. Merge kernel preparation and node execution into one
   ``PreparedExecutionPlan`` and first execute its session-scoped tasks
   sequentially through completion events.
#. Add two dependent inference nodes with independently delayed weights and
   verify that the first node executes before the second weight finishes
   loading, while later inferences reuse resident prepared objects.
#. Add persistent bounded I/O and CPU queues, inference priority inheritance,
   reserved critical-path admission, and global/per-arena memory budgets; then
   compare the same plans in sequential and parallel modes.
#. Execute sequential inference as a suspendable continuation and add the
   readiness-epoch hot path; verify that fully prepared inference does not
   allocate or enqueue one task per node and benchmark its overhead against the
   current direct ``ExecutionPlan`` replay.
#. Add ``ExecutionHandle`` for both ``PrepareAsync`` and ``RunAsync``; implement
   synchronous ``Prepare`` and ``Run`` as asynchronous submission followed by
   ``Wait`` over the same scheduler.
#. Add task failure propagation, dependency cancellation, and deterministic
   handle-destruction semantics; test initialization failure separately from
   per-inference failure.
The companion prepared cache, eviction/offloading, CUDA preparation, and
alternative placement variants continue in the fourth roadmap document after
this core is stable.
