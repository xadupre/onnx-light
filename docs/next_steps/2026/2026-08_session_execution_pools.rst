.. _l-next-steps-session-execution-pools:

Session execution policies and shared CPU pools
===============================================

:Date: 2026-08

**implementation complete**

Progress
++++++++

The roadmap is delivered incrementally. Steps that are merged are recorded
here; the full sequence is tracked in the `Implementation sequence`_ table.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Step
     - Scope
     - Result
   * - Pool PR01
     - Requested and resolved CPU policy
     - Added ``CpuExecutionPolicy`` and ``ResolvedCpuExecutionPolicy`` with
       typed thread, spin, affinity, CPU-set, and nesting options.
       ``ResolveCpuExecutionPolicy`` validates the request deterministically,
       derives the effective participant count from the process-visible
       topology, and records fallback diagnostics.
   * - Pool PR02
     - Executor and compatible-pool registry
     - Added ``CpuExecutor`` and the bounded ``CpuExecutorRegistry``. Compatible
       resolved policies share a persistent pool; incompatible policies remain
       isolated, the last lease stops workers immediately, serial policies
       create no workers, and inherited executors are rejected after ``fork``.
   * - Pool PR03
     - Session and runtime wiring
     - ``RuntimeSessionOptions`` accepts a ``CpuExecutionPolicy``, defaulting to
       the policy derived from ``RuntimeParameters::num_threads``.
       ``RuntimeSession`` leases a shared executor before preparing kernels,
       installs it on the running thread and on ``RuntimeContext`` for the whole
       run, and reports it through ``cpu_executor()``. Worker startup now
       follows the lease, so reusing a session amortizes it while rebuilding a
       session for every inference pays it again. Portable ``ParallelFor``
       and ``ParallelForThreadCount`` dispatch through the installed executor,
       so a session's requested participants are the participants its kernels
       observe and the process-wide pool is no longer reached from a session
       run. A nested session without an explicit policy (a subgraph body, a
       model-local function) keeps the enclosing executor instead of leasing a
       second pool. Kernel tuning descriptors report the effective threads of
       the executor installed for the run.
   * - Pool PR04
     - Tuning, Python API, and inspection
     - Python exposes typed CPU spin, affinity, processor, request, resolution,
       sharing-key, and optional-counter objects. ``ReferenceEvaluator`` accepts
       a validated ``cpu_execution`` mapping and reports its requested and
       resolved policy. Calibration may lease an explicit compatible executor
       and rejects an execution descriptor whose participants differ from the
       active executor before invoking a callback. Optional executor counters
       allocate no state until enabled and report dispatches, limited inline
       calls, and nested inline calls.
   * - Pool PR05
     - Registered-kernel executor adapter
     - ``onnx-light-cpu`` registered kernels install a non-owning view of the
       exact session ``CpuExecutor`` and dispatch every parallel range through
       it.
   * - Pool PR06
     - Runtime-event executor identity
     - Every node-run event records the process-local identity and effective
       participant count of the executor that performed the dispatch.
   * - Pool PR07
     - Scheduler ownership and compatibility gate
     - Removed the private ``onnx-light-cpu`` scheduler and its controls.
       Standalone calls are serial, registered calls use the session executor,
       and source integration against ``onnx-light/main`` passes on Linux,
       macOS, and Windows.

Objective
+++++++++

The objective is to make one truthful CPU execution policy control every
parallel region launched for a runtime session. Thread count, affinity,
spin-before-park, nesting, kernel-specific participant limits, calibration,
and diagnostics must describe the workers that actually execute the graph.

``onnx-light`` should own the policy and executor used by registered kernels.
It should not create one independent set of threads per session. Sessions with
the same resolved policy should lease a compatible shared pool from a registry.
Sessions with incompatible policies must not silently share one.

Standalone kernel libraries must either accept an explicitly supplied executor
or run serially outside ``onnx-light``. When a kernel is registered with
``onnx-light``, it uses the session executor rather than waking a second private
pool.

This plan is the ``onnx-light`` counterpart of the
`onnx-light-cpu Runtime Execution Controls roadmap
<https://github.com/xadupre/onnx-light-cpu/blob/docs/benchmark-runtime-tuning/docs/next_steps/2026_08_runtime_execution_controls.rst>`_.

Problem addressed
+++++++++++++++++

Before this roadmap, ``RuntimeParameters`` stored ``num_threads`` in every
``RuntimeSession``, but the portable ``ParallelFor`` implementation used one
process-wide ``GlobalThreadPool``. Its participant count was resolved from a
default-constructed ``RuntimeParameters`` and captured on first use. A session
could therefore report one requested thread count while its kernels executed
with another.

The old global pool also had a compiled spin count and no public affinity
policy. Registered ``onnx-light-cpu`` kernels could additionally wake their own
process-wide pool, allowing two persistent worker sets to interfere or nest.
Pool PR01--PR07 replaced that behavior with an inspectable session policy, a
compatible-executor registry, and one scheduler for registered execution.

Calibration now validates its execution descriptor against the installed
executor, so profiles and inference use the same resolved participant count.

Ownership decision
++++++++++++++++++

``onnx-light`` owns:

* the public per-session CPU policy;
* validation and topology-aware policy resolution;
* the registry of compatible shared pools;
* pool lifetime and worker shutdown;
* the executor passed to prepared kernels;
* nesting and concurrent-session behavior;
* resolved-policy inspection and optional scheduler diagnostics;
* the execution descriptor used by tuning and calibration.

Kernel libraries own:

* serial or range-based compute functions;
* safe portable thresholds;
* kernel-specific maximum useful participants;
* algorithm and packing parameters;
* calibration callbacks and correctness checks;
* standalone execution policy for callers that do not use ``onnx-light``.

The runtime must not know an accelerated kernel's algorithm. The kernel must
not reinterpret or override the session's thread, spin, or affinity policy.

Requested and resolved policy
+++++++++++++++++++++++++++++

Replace the single effective-thread calculation with two explicit types. Names
are illustrative and may change during API review.

.. code-block:: cpp

    enum class CpuSpinPolicy {
      kAdaptive,
      kFixedIterations,
      kFixedDuration,
      kParkImmediately,
    };

    enum class CpuAffinityPolicy {
      kNone,
      kPhysicalCores,
      kPerformanceCores,
      kPhysicalThenSmt,
      kExplicit,
    };

    struct CpuExecutionPolicy {
      int32_t num_threads = 0;
      CpuSpinPolicy spin_policy = CpuSpinPolicy::kAdaptive;
      uint64_t spin_budget = 0;
      CpuAffinityPolicy affinity_policy = CpuAffinityPolicy::kPhysicalCores;
      std::vector<CpuLogicalProcessor> cpu_set;
      bool allow_nested_parallelism = false;
    };

    struct ResolvedCpuExecutionPolicy {
      CpuExecutionPolicy request;
      uint32_t effective_threads;
      std::optional<CpuLogicalProcessor> caller_processor;
      std::vector<CpuLogicalProcessor> worker_processors;
      bool uses_smt;
      bool uses_efficiency_cores;
      ResolvedSpinPolicy spin;
      std::vector<std::string> diagnostics;
    };

``num_threads == 0`` selects a topology-derived default, ``1`` is serial, and
values above one request that many participants including the caller. Invalid
explicit CPU sets, impossible affinity requests, negative values, and
unsupported combinations fail explicitly. A fallback from an unavailable
topology feature is recorded in diagnostics.

An explicit CPU set lists all participants: its first processor belongs to the
caller and the remaining processors belong to workers. This keeps
``worker_processors`` consistent with the participant count and allows an
explicit serial policy to retain its caller placement.

The process-visible CPU set is authoritative. Resolution must respect Linux
cpusets and containers, Windows processor groups, hybrid P/E cores, SMT
siblings, and platforms where pinning is unsupported. Processor identifiers
must not be inferred from adjacency.

Pool registry
+++++++++++++

Pool PR02 introduces a process-owned ``CpuExecutorRegistry``. A session resolves its
policy, obtains an immutable key, and leases a ``CpuExecutor``:

.. code-block:: text

    RuntimeSession
      -> resolve CpuExecutionPolicy
      -> CpuExecutorRegistry::Acquire(resolved_policy)
      -> shared CpuExecutor lease
      -> prepared kernels receive executor view

The registry key includes every property that changes worker behavior:

* effective participant count;
* exact worker processor assignment or explicit no-affinity policy;
* resolved spin and park policy;
* nesting policy;
* any worker-lifetime policy.

The key does not include diagnostics, counters, session identifiers, or
kernel-specific thresholds. Two sessions share only when their immutable keys
are equal.

Leases are reference-counted. The registry retains weak references, so
releasing the last lease stops the pool immediately. A registry accepts at
most its configured number of simultaneously live incompatible pools; the
process-owned registry defaults to eight. Capacity exhaustion fails explicitly,
and expired entries never consume capacity.

A serial policy does not create worker threads. An executor records its
creating process and rejects dispatch after ``fork``; inherited worker state is
never treated as usable.

Executor interface
++++++++++++++++++

Kernels need a small non-owning interface, not the concrete pool:

.. code-block:: cpp

    class CpuExecutor {
    public:
      uint32_t effective_threads() const noexcept;
      const ResolvedCpuExecutionPolicy &policy() const noexcept;

      void ParallelFor(int64_t total, int64_t grain,
                       void *context, ParallelRangeFn function,
                       uint32_t maximum_participants = 0);
    };

``maximum_participants == 0`` means the session limit. A prepared kernel may
request a lower positive limit resolved by processor-aware tuning. It can never
exceed the session limit.

Worker affinity is applied before the executor becomes available. Failure to
apply a resolved worker assignment fails acquisition. Explicit caller affinity
is applied at dispatch and likewise fails explicitly if the process CPU set
changed after resolution.

``RuntimeContext`` or the prepared-kernel context carries a non-owning executor
view. Portable kernel helpers dispatch through that view. Standalone helpers
may use an explicitly supplied executor or a documented standalone default.

The existing free ``ParallelFor`` API can remain as a compatibility wrapper,
but runtime kernels must migrate to the context executor. A hidden global
fallback must not remain in a path that claims to obey session parameters.

Pool PR03 implements this with a thread-scoped view. ``RuntimeSession::Run``
installs its leased executor through ``CpuExecutorScope`` and on
``RuntimeContext::cpu_executor`` for the duration of the run, and every
participant of a parallel region keeps that view installed, so a nested region
runs inline on the same executor. The free ``ParallelFor`` dispatches through
the installed executor and only falls back to the process-wide pool for
standalone callers running outside any session. A nested session that did not
receive an explicit policy inherits the executor already installed on the
context, so subgraphs and model-local functions run with the policy of the
session that started the run.

Session wiring derives its default policy from
``RuntimeParameters::num_threads`` and requests ``CpuAffinityPolicy::kNone``,
which keeps worker placement identical to the pre-policy behavior. Pinned
defaults are a placement change that belongs with the measurements of Pool
PR07; a caller that wants them supplies an explicit policy today.

Nesting and concurrency
+++++++++++++++++++++++

Nested parallel regions run inline by default. This includes:

* a kernel calling another parallel helper;
* a registered ``onnx-light-cpu`` kernel running inside a session worker;
* application code invoking a session from its own pool;
* callbacks that enter BLAS, OpenMP, or another runtime pool.

An executor marks its workers and caller-owned active regions. Pool PR02 always
runs nested calls inline, including when the nesting flag is present in the
sharing key. A nested call may reuse the current participants only if a later
explicit composition design proves it deadlock-free and bounded; it must never
wake an unrelated pool.

Concurrent calls sharing one executor serialize only the parallel-region
dispatch metadata, not complete inference runs. The design must measure and
document whether one active region at a time is acceptable or whether the pool
needs a bounded multi-region scheduler.

Spinning and parking
++++++++++++++++++++

Spinning applies both to workers waiting for a new generation and to the caller
waiting for worker completion. The initial public policy may control both with
one setting. Split controls are justified only by measurements.

The default is bounded and eventually parks. A fixed duration is more portable
than a raw pause count, while a fixed-iteration mode is useful for compatibility
and low-level experiments. Adaptive policy may consider call cadence,
oversubscription, power mode, and observed wakeup latency, but it must resolve
to inspectable behavior rather than silently changing calibration conditions.

Do not expose a magic compile-time spin constant as the only production
control. Do not let loading ``onnx-light-cpu`` implicitly change an
``onnx-light`` session through ``ONNX_LIGHT_CPU_*`` environment variables.

Affinity
++++++++

Default affinity uses one logical processor per physical core and prefers
performance cores when topology can identify them. SMT siblings are added only
for an explicit policy that requests them.

Affinity resolution must:

* preserve an externally pinned calling thread unless explicitly changed;
* avoid placing a worker on the caller's physical core when alternatives exist;
* report every failed pin;
* reject unavailable explicit processor identifiers;
* distinguish unsupported affinity from a successful no-affinity policy;
* define behavior if the process CPU set changes after pool creation.

Applications that own placement need ``kNone`` and an explicit executor
injection path.

Integration with runtime preparation
++++++++++++++++++++++++++++++++++++

``RuntimeSession`` resolves and leases its executor before preparing kernels.
The resolved execution descriptor is then available while a kernel captures
immutable tuning parameters. Dynamic-shape re-preparation does not change the
executor unless the caller creates a new session policy.

The future :ref:`l-next-steps-prepared-execution` task graph must submit
invocation tasks through the same executor or through a scheduler that owns it.
Prepared execution must not introduce another worker pool beside the kernel
pool. Session-scoped preparation tasks and invocation-scoped compute tasks need
an explicit resource class when they can overlap.

Registered kernel libraries
+++++++++++++++++++++++++++

``onnx-light-cpu`` now installs an adapter that accepts the ``CpuExecutor`` view
and invokes serial SIMD range functions. Standalone C++ entry points run
serially unless a caller supplies execution through a registered
``onnx-light`` session; the private pool no longer exists.

Registration should advertise executor support as a capability. A kernel that
requires an executor but receives none fails preparation rather than silently
using a global pool. Compatibility adapters may run serially while a library is
migrated.

The integration must prove:

* no ``onnx-light-cpu`` worker is created by registered-kernel execution;
* the session's effective threads equal observed participants;
* kernel-specific maximum participants are respected;
* nested calls remain inline;
* standalone kernels retain their documented behavior.

Tuning and cache identity
++++++++++++++++++++++++

Calibration executes on the same executor as inference. A request whose
execution descriptor differs from the active executor fails before allocating
benchmark tensors.

Persistent profile compatibility includes effective threads and stable
execution properties that can change the winning parameter: affinity class,
SMT use, and spin-policy class where measured. Exact transient worker IDs,
session IDs, and scheduler counters do not belong in cache keys.

The completed :ref:`l-next-steps-processor-aware-kernel-tuning` infrastructure
remains the owner of schema validation, calibration, and persistence. This
roadmap replaces only the hidden global executor assumptions beneath it.

Python API and inspection
++++++++++++++++++++++++

``ReferenceEvaluator`` should accept the same typed policy as
``RuntimeSessionOptions`` and expose its immutable resolution:

.. code-block:: python

    evaluator = ReferenceEvaluator(
        model,
        cpu_execution={
            "num_threads": 0,
            "spin_policy": "adaptive",
            "affinity_policy": "physical_cores",
        },
    )
    print(evaluator.cpu_execution_policy)

Unknown keys or invalid values raise. Inspection reports requested and
effective threads, worker processors, SMT/P/E use, spin/park policy, registry
sharing identity, and fallback diagnostics.

Optional counters include dispatches, spins, parks, wakeups, caller waits,
worker-active time, and nested-inline calls. They are disabled by default with
no allocation, lock, or clock read in the hot path. Detailed region telemetry
belongs to :ref:`l-next-steps-parallel-for-profiling`.

Pool PR04 exposes the policy as native Python objects and accepts either a
``CpuExecutionPolicy`` or the mapping shown above. Mapping keys and string enum
values are validated eagerly. ``cpu_execution_resolution`` reports the
immutable topology resolution, ``cpu_execution_identity`` reports the
behavior-only executor sharing key, and ``cpu_execution_counters`` reports a
snapshot of optional dispatch counters. Counters are disabled by default; the
hot path performs no counter allocation, lock, or clock read in that state.
Counters belong to the shared executor rather than one session: enabling them
through any compatible lease enables cumulative counting for every leaseholder
until that executor is destroyed.

Pool PR06 gives every ``kRunNode`` runtime event the process-local identity of
the exact ``CpuExecutor`` installed for that dispatch and its effective
participant count. Compatible sessions therefore emit the same identifier,
incompatible executors emit different identifiers, and nested sessions retain
their enclosing executor identity. The identifier is diagnostic and
non-persistent; tuning and prepared-object compatibility continue to use the
behavior-only executor key.

Validation
++++++++++

Correctness tests cover:

* serial and multiple participant counts;
* two sessions sharing one compatible pool;
* incompatible sessions receiving distinct pools;
* concurrent runs, nested calls, and session destruction;
* failed affinity and changed process CPU sets;
* pool-registry bounds and idle eviction;
* exceptions during policy resolution and kernel preparation;
* registered and standalone ``onnx-light-cpu`` paths;
* thread-sanitizer runs and fork handling where supported.

Performance tests cover tiny latency, bursty inference, sustained throughput,
memory-bound elementwise kernels, GEMM, idle power, and concurrent sessions.
They record complete policy metadata and follow the
`onnx-light-cpu benchmark methodology
<https://github.com/xadupre/onnx-light-cpu/blob/docs/benchmark-runtime-tuning/docs/design/benchmark_methodology.rst>`_.

Implementation sequence
+++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 9 22 43 14 12

   * - PR
     - Repository and scope
     - Merge criterion
     - Depends on
     - Status
   * - Pool PR01
     - ``onnx-light``: requested and resolved CPU policy.
     - Typed thread, spin, affinity, CPU-set, and nesting options validate
       deterministically; topology and fallback diagnostics are tested.
     - None
     - Done
   * - Pool PR02
     - ``onnx-light``: executor and compatible-pool registry.
     - Compatible sessions share a bounded pool; incompatible and serial
       policies behave correctly; lifecycle and thread-sanitizer tests pass.
     - PR01
     - Done
   * - Pool PR03
     - ``onnx-light``: session/runtime wiring.
     - ``RuntimeSession`` and ``ReferenceEvaluator`` use the leased executor;
       requested thread counts equal observed participants; global fallback is
       absent from runtime kernels.
     - PR02
     - Done
   * - Pool PR04
     - ``onnx-light``: tuning, Python, and inspection.
     - Calibration uses the active executor; Python exposes policy and
       resolution; disabled counters have no measurable overhead.
     - PR03
     - Done
   * - Pool PR05
     - ``onnx-light-cpu``: registered-kernel executor adapter.
     - Registered kernels use only the session executor, never wake the private
       CPU pool, and respect kernel participant limits; standalone behavior is
       unchanged.
     - PR03
     - Done (`onnx-light-cpu#270
       <https://github.com/xadupre/onnx-light-cpu/pull/270>`_)
   * - Pool PR06
     - ``onnx-light``: profiling and prepared-execution integration.
     - Region events identify the resolved executor; future prepared tasks do
       not introduce an incompatible pool or nested oversubscription.
     - PR04, PR05
     - Done (`#4594
       <https://github.com/xadupre/onnx-light/pull/4594>`_)
   * - Pool PR07
     - Both repositories: compatibility and performance gate.
     - The private ``onnx-light-cpu`` scheduler is absent; standalone kernels
       are serial; registered kernels use only the session executor; the
       cross-repository policy and compatibility gates pass.
     - PR06
     - Done (`onnx-light-cpu#271
       <https://github.com/xadupre/onnx-light-cpu/pull/271>`_)

Pool PR07 completed the roadmap. Its cross-repository gate builds
``onnx-light`` from ``main`` and validates ``onnx-light-cpu`` against that exact
runtime on Linux, macOS, and Windows. Documentation remains a Linux-only build;
the integration matrix varies only the operating system, not stable external
dependency versions. It also removes ``onnx-light-cpu``'s private
``parallel_for`` implementation so runtime policy has a single scheduler.
The follow-up `onnx-light-cpu#272
<https://github.com/xadupre/onnx-light-cpu/pull/272>`_ fixes source-package
discovery in that Linux documentation job without changing scheduler
ownership.
