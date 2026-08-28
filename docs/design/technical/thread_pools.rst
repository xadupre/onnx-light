.. _l-design-runtime-thread-pools:
.. _l-technical-details-thread-pools:

Thread-pool dispatch in onnx-light, OpenMP, and ONNX Runtime
============================================================

This page compares how three CPU execution systems run a dependent sequence of
ONNX operators:

.. math::

    Y_1 = \operatorname{Gemm}(A, B),\qquad
    P = \operatorname{Softmax}(Y_1),\qquad
    Y_2 = \operatorname{Gemm}(P, C).

The comparison assumes that graph optimization has not fused the operators and
that every kernel uses the execution system named in its column. The second
operator cannot start before the first has produced ``Y1``, and the final
``Gemm`` cannot start before ``Softmax`` has produced ``P``. Thread-pool design
therefore changes dispatch and synchronization cost, but does not remove these
data dependencies.

Scope and terminology
---------------------

A *participant* is either the thread that called the parallel region or one
worker. A *dispatch* publishes work to workers. A *join* below means waiting for
the region, not destroying and joining the underlying operating-system
threads. Persistent workers normally survive all three operators.

OpenMP is a specification rather than one runtime implementation. The OpenMP
column describes the common fork-join design used by LLVM ``libomp`` and GCC
``libgomp``; exact queues, barriers, spin budgets, and operating-system wait
primitives differ between those runtimes and platforms.

Overview
--------

.. list-table::
    :header-rows: 1
    :widths: 20 27 26 27

    * - Aspect
      - ``onnx-light``
      - OpenMP
      - ONNX Runtime
    * - Pool ownership
      - A session lazily leases a :cpp:class:`CpuExecutor` from a bounded
        process registry. Compatible resolved policies share one executor.
      - The OpenMP runtime owns teams of workers. Implementations normally
        retain a hot team or reusable workers after the first parallel region.
      - An inference session owns an intra-operator pool by default. Applications
        can instead configure environment-level global pools shared by sessions.
    * - Worker creation
      - The first lease constructs ``effective_threads - 1`` ``std::thread``
        workers; the calling thread is the remaining participant.
      - The first relevant parallel region initializes or expands a team,
        commonly using POSIX or native platform threads. Later regions reuse it.
      - Pool creation creates the extra intra-operator workers. The default
        participant count targets physical cores and includes the calling
        thread.
    * - Work publication
      - A short mutex-protected update installs the callable and block count,
        then increments an atomic generation and calls
        ``condition_variable::notify_all``.
      - The compiler outlines the region. ``__kmpc_fork_call`` in LLVM or
        ``GOMP_parallel`` in GCC publishes that outlined function and releases
        the team through runtime barriers.
      - ORT submits ranges or tasks to its modified Eigen non-blocking pool.
        Workers can consume assigned work and attempt to steal available work.
    * - Work assignment
      - Static: block 0 runs on the caller and block ``j`` runs on worker
        ``j - 1``. A grain threshold and participant limit can keep a kernel
        inline or use only part of the pool.
      - ``schedule(static)`` derives fixed chunks without a shared work queue.
        Dynamic and guided schedules obtain chunks through runtime scheduling
        state and atomics.
      - Operator helpers choose a task count from work cost. Scheduling is more
        dynamic than ``onnx-light``'s fixed worker-to-block mapping.
    * - Idle waiting
      - Workers spin for the resolved iteration or duration budget and then
        park on a ``std::condition_variable``.
      - Workers normally spin or yield for a runtime-defined interval and then
        use an implementation- and platform-specific blocking wait.
      - Spinning is enabled by default. ORT supports disabling it or selecting
        a calibrated duration and exponential pause backoff before workers
        sleep.
    * - Region completion
      - Workers decrement an atomic remaining count. The caller spins, then
        waits on a second condition variable if work is still outstanding.
      - A fork-join barrier, commonly generation- or sense-based and optimized
        with atomics or a tree, releases the primary thread after the team
        arrives.
      - The caller waits for the submitted intra-operator work. Pool barriers
        and task counters complete the operator before graph execution advances.
    * - Nested work
      - A nested region on the same executor runs inline to prevent deadlock
        and oversubscription.
      - Controlled by OpenMP nesting and active-level settings; a nested region
        may serialize or form another team.
      - The thread pool detects parallel sections and limits nested
        parallelism; operator implementations also use cost thresholds.
    * - Concurrent callers
      - Dispatch metadata is serialized per shared executor. Compatible
        sessions may share workers, but their parallel regions do not execute
        concurrently on that executor.
      - Behavior depends on the runtime and team configuration. Independent
        host threads can request teams and may contend or oversubscribe.
      - Per-session pools can execute independently and may oversubscribe the
        machine. Global pools trade isolation for shared capacity.
    * - Shutdown
      - Releasing the final compatible lease sets an atomic stop flag, changes
        the generation, notifies all workers, and joins every ``std::thread``.
      - Workers usually live until runtime or process teardown; implementation
        shutdown joins or releases the native threads.
      - A per-session pool ends with its session; a global pool ends with its
        environment. Custom thread callbacks expose creation and joining.

The concrete three-operator execution
-------------------------------------

The following timeline shows the synchronization visible to the graph
executor. ``D`` is dispatch, ``B`` is the region-completion barrier, and
``idle`` means either spinning or parked:

.. code-block:: text

    graph caller:  D  Gemm 1  B | D  Softmax  B | D  Gemm 2  B
    workers:       wake/work/idle | wake/work/idle | wake/work/idle
    dependency:                Y1 |             P  |             Y2

``onnx-light``
++++++++++++++

``RuntimeSession::Run`` acquires the executor on first use and installs it on
the calling thread and :cpp:class:`RuntimeContext`. Each parallel kernel calls
``CpuExecutor::ParallelFor``:

1. The kernel may stay inline when its work is below ``grain``. Otherwise,
   ``ParallelFor`` divides the range into at most one block per admitted
   participant.
2. ``ThreadPool::Run`` locks the executor's region mutex, publishes one
   callable under its state mutex, increments ``generation_``, and notifies all
   workers.
3. The caller computes block 0 while worker ``j - 1`` computes block ``j``.
4. Workers decrement ``remaining_``. The caller first spins and then waits on
   ``cv_done_`` if needed.
5. The workers spin for the next generation and eventually park on
   ``cv_work_``. The nearby ``Softmax`` or second ``Gemm`` can reach them while
   they are still spinning and avoid an operating-system wake-up.

The mutexes do not cover matrix multiplication or softmax arithmetic. They
only serialize publication and concurrent regions. The unconditional
``notify_all`` can nevertheless wake workers that have no block when a kernel
admits fewer participants than the pool contains.

OpenMP
++++++

For three kernels implemented as three OpenMP parallel loops, the compiler
outlines each loop body and emits three runtime fork-join calls:

1. The first call obtains a team and may pay lazy worker creation.
2. A static ``Gemm`` loop gives each team member a deterministic tile range.
3. The join barrier makes ``Y1`` visible before the primary thread enters the
   ``Softmax`` region.
4. The same fork-join sequence occurs for ``Softmax`` and the second ``Gemm``.
   Workers are normally reused rather than recreated.

Static scheduling avoids a lock per tile. Dynamic scheduling can balance
irregular tiles but adds atomic scheduling traffic. Keeping one outer OpenMP
region around all three operators could remove two team releases, but it would
require team-aware kernels and explicit barriers between dependent operators.
Calling independently parallel kernels from that outer region can instead
trigger nested parallelism or serialization, so inference runtimes generally
dispatch each operator separately.

ONNX Runtime
++++++++++++

With the default sequential graph execution mode, the three nodes run in graph
order and use the session's intra-operator pool:

1. The first ``Gemm`` asks MLAS and ORT's thread-pool helpers to parallelize
   suitable matrix tiles.
2. The operator waits for its intra-operator tasks before the graph executor
   dispatches ``Softmax``.
3. ``Softmax`` parallelizes only when its row count and estimated work justify
   pool overhead; otherwise it can run on the caller while the extra workers
   remain idle.
4. The second ``Gemm`` reuses the same persistent intra-operator workers.

``ORT_PARALLEL`` adds a distinct inter-operator pool for independent graph
branches. It does not overlap this linear sequence because each node consumes
its predecessor's output. ORT's default worker spinning favors the short gaps
between these operators; disabling spinning saves CPU and power but can add a
sleep-to-running transition to the next dispatch.

Where time is spent
-------------------

For one inference after pool initialization, a useful decomposition is:

.. math::

    T \approx T_{\mathrm{gemm1}} + T_{\mathrm{softmax}} +
    T_{\mathrm{gemm2}} + \sum_{i=1}^{3}
    \left(T_{\mathrm{dispatch},i} + T_{\mathrm{barrier},i}\right).

Worker creation is outside this steady-state equation. If a program constructs
and destroys a session for every inference, pool creation and joining must be
added and can dominate small models.

Large ``Gemm`` operators normally dwarf mutex and dispatch costs. The middle
``Softmax`` is the important boundary case: waking a team can cost more than a
small softmax, so all three systems need a threshold that leaves insufficient
work inline. For medium work, the spin policy decides whether the next
operator observes a worker already running or pays scheduler wake-up latency.

The systems optimize different constraints:

* ``onnx-light`` favors explicit ownership, deterministic static assignment,
  compatible-session sharing, and inspectable policy.
* OpenMP offers highly tuned fork-join barriers and several schedules, but the
  exact lifetime and waiting behavior belongs to the selected OpenMP runtime.
* ONNX Runtime favors general operator task scheduling, work stealing, and
  configurable per-session or global pools.

Consequently, replacing one uncontended mutex is unlikely to change a large
``Gemm``--``Softmax``--``Gemm`` pipeline. Measurements should separate first
run from steady state, report whether workers spin or park, and include a small
softmax case where dispatch overhead is measurable.

Implementation references
-------------------------

* ``onnx-light``:
  `CPU executor
  <https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_core/runtime/tuning/cpu_executor.cc>`_,
  `persistent thread pool
  <https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_core/runtime/kernels/parallel_for.cc>`_,
  and
  :ref:`l-next-steps-session-execution-pools`.
* ONNX Runtime:
  `Thread management
  <https://onnxruntime.ai/docs/performance/tune-performance/threading.html>`_,
  `thread-pool construction
  <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/common/threadpool.cc>`_,
  and `modified Eigen non-blocking pool
  <https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/platform/EigenNonBlockingThreadPool.h>`_.
* OpenMP:
  `LLVM fork-join runtime
  <https://github.com/llvm/llvm-project/blob/main/openmp/runtime/src/kmp_runtime.cpp>`_,
  `LLVM wait and release primitives
  <https://github.com/llvm/llvm-project/blob/main/openmp/runtime/src/kmp_wait_release.h>`_,
  `GCC team lifecycle
  <https://github.com/gcc-mirror/gcc/blob/master/libgomp/team.c>`_, and
  `GCC barriers
  <https://github.com/gcc-mirror/gcc/blob/master/libgomp/barrier.c>`_.
