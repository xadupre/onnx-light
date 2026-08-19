.. _l-next-steps-session-execution-pools:

Session execution policies and shared CPU pools
===============================================

:Date: 2026-08

**implementation in progress**

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

Standalone kernel libraries may retain a private executor for calls made
outside ``onnx-light``. When a kernel is registered with ``onnx-light``, it must
use the session executor rather than waking a second private pool.

This plan is the ``onnx-light`` counterpart of the
`onnx-light-cpu Runtime Execution Controls roadmap
<https://github.com/xadupre/onnx-light-cpu/blob/docs/benchmark-runtime-tuning/docs/next_steps/2026_08_runtime_execution_controls.rst>`_.

Current problem
+++++++++++++++

``RuntimeParameters`` currently stores ``num_threads`` in every
``RuntimeSession``, but the portable ``ParallelFor`` implementation uses one
process-wide ``GlobalThreadPool``. Its participant count is resolved from a
default-constructed ``RuntimeParameters`` and captured on first use. Therefore
a session can report one requested thread count while its kernels execute with
another.

The current global pool also has a compiled spin count and no public affinity
policy. It cannot support two sessions that request different execution
policies. Registered ``onnx-light-cpu`` kernels may additionally wake their own
process-wide pool, so two sets of persistent workers can coexist, spin, retain
affinity, interfere with benchmarks, or nest.

Calibration now rejects an execution descriptor whose thread count differs
from ``ParallelForThreadCount()``. That check prevents a false profile, but it
does not make session parameters effective. The executor must become
session-aware before calibration can truthfully vary execution policy.

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

The process-visible CPU set is authoritative. Resolution must respect Linux
cpusets and containers, Windows processor groups, hybrid P/E cores, SMT
siblings, and platforms where pinning is unsupported. Processor identifiers
must not be inferred from adjacency.

Pool registry
+++++++++++++

Introduce a process-owned ``CpuExecutorRegistry``. A session resolves its
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

Use reference-counted leases. Releasing the last lease may stop the pool
immediately or place it in a bounded idle cache; the chosen lifecycle must be
deterministic and testable. The registry needs a strict bound so creating many
policies cannot leave an unbounded number of parked threads.

A serial policy does not create worker threads. Forked child behavior must be
defined explicitly: inherited worker state is never treated as usable.

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

``RuntimeContext`` or the prepared-kernel context carries a non-owning executor
view. Portable kernel helpers dispatch through that view. Standalone helpers
may use an explicitly supplied executor or a documented standalone default.

The existing free ``ParallelFor`` API can remain as a compatibility wrapper,
but runtime kernels must migrate to the context executor. A hidden global
fallback must not remain in a path that claims to obey session parameters.

Nesting and concurrency
+++++++++++++++++++++++

Nested parallel regions run inline by default. This includes:

* a kernel calling another parallel helper;
* a registered ``onnx-light-cpu`` kernel running inside a session worker;
* application code invoking a session from its own pool;
* callbacks that enter BLAS, OpenMP, or another runtime pool.

An executor marks its workers and caller-owned active regions. A nested call
may reuse the current participants only if a later explicit composition design
proves it deadlock-free and bounded; it must never wake an unrelated pool.

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

``onnx-light-cpu`` needs an adapter that accepts the ``CpuExecutor`` view and
invokes serial SIMD range functions. Its private pool remains available only
for standalone C++ entry points.

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
     - Pending
   * - Pool PR03
     - ``onnx-light``: session/runtime wiring.
     - ``RuntimeSession`` and ``ReferenceEvaluator`` use the leased executor;
       requested thread counts equal observed participants; global fallback is
       absent from runtime kernels.
     - PR02
     - Pending
   * - Pool PR04
     - ``onnx-light``: tuning, Python, and inspection.
     - Calibration uses the active executor; Python exposes policy and
       resolution; disabled counters have no measurable overhead.
     - PR03
     - Pending
   * - Pool PR05
     - ``onnx-light-cpu``: registered-kernel executor adapter.
     - Registered kernels use only the session executor, never wake the private
       CPU pool, and respect kernel participant limits; standalone behavior is
       unchanged.
     - PR03
     - Pending
   * - Pool PR06
     - ``onnx-light``: profiling and prepared-execution integration.
     - Region events identify the resolved executor; future prepared tasks do
       not introduce an incompatible pool or nested oversubscription.
     - PR04, PR05
     - Pending
   * - Pool PR07
     - Both repositories: compatibility and performance gate.
     - Policy matrix, concurrent sessions, registered/standalone kernels, and
       tuning caches pass; default latency and throughput do not regress.
     - PR06
     - Pending

Pool PR07 is the final roadmap PR.

