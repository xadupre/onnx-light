.. _l-technical-details-loading-prepacking:

Overlapping ONNX loading and prepacking
=======================================

Loading a large ONNX model and preparing its weights do not have to be two
global phases:

.. code-block:: text

    read every weight -> wait -> prepack every weight -> wait -> run

The useful unit is one prepared object. As soon as the bytes for one weight are
available, its prepack can start while other weights are still being read:

.. code-block:: text

    I/O:       read W0 | read W1 | read W2 | ...
                  |         |         |
    CPU:      prepack W0 | prepack W1 | prepack W2
                  |         |         |
    publish:   ready W0 | ready W1 | ready W2

This page explains how :ref:`the three execution systems compared previously
<l-technical-details-thread-pools>` can implement that pipeline. It also
distinguishes the implementation already present in ``onnx-light`` from
possible OpenMP and ONNX Runtime mappings.

The dependency boundary
-----------------------

Prepacking cannot start merely because parsing has found a ``TensorProto``.
The runtime must first know which physical representation the selected kernel
needs. In ``onnx-light``, the safe boundary is a frozen ``ResolvedModel`` and
its ``RequiredPayloadManifest``:

1. parse enough model metadata to construct and transform the graph;
2. select the live graph, kernels, devices, and prepared layouts;
3. freeze the payload manifest so no later rewrite adds another weight read;
4. create one session-scoped chain for each required prepared object;
5. submit independent chains as soon as their dependencies and memory budgets
   permit.

This ordering avoids reading dead initializers, prepacking a layout that a graph
rewrite removes, or publishing bytes for a kernel variant that was not selected.
It does not require all selected payloads to be resident before preparation
starts.

One weight has this dependency chain:

.. code-block:: text

    source or packed-cache lookup
             |
       +-----+----------------------+
       | cache hit                 | cache miss
       v                           v
    read packed payload       read portable weight
       |                           |
       |                        prepack
       |                           |
       +------------+--------------+
                    v
          atomically publish object
                    |
             dependent kernel

A compatible packed-cache hit skips both the portable-weight read and the
prepack. A miss retains the portable source as the fallback, creates the packed
object, publishes it before optional persistence, and may write the reusable
cache entry in the background.

Why two resource pools are necessary
------------------------------------

Reading external data is blocking and usually latency-bound. Prepacking is CPU-
and memory-bandwidth-bound. A single undifferentiated pool permits all workers
to block in file reads while ready prepacks wait, or all workers to prepack
speculative weights while a critical read has no admission slot.

The resources therefore have different ownership:

.. list-table::
    :header-rows: 1
    :widths: 22 25 26 27

    * - Resource
      - Work
      - Waiting behavior
      - Required bound
    * - I/O workers
      - External-data reads and packed-cache reads or writes
      - May block in operating-system file calls
      - Worker count and bytes in flight
    * - CPU executor
      - Transpose, quantize, hash, pack, and inference kernels
      - Must remain available for runnable compute
      - Participants and preparation scratch
    * - Inline coordinator
      - Dependency updates, publication, and trivial tasks
      - Must not perform long blocking or compute operations
      - Short non-blocking work only
    * - Device queues
      - Copies and device-side preparation
      - Completion through backend events
      - Streams, device memory, and staging bytes

Separate pools do not mean that all work should run simultaneously.
``estimated_input_bytes``, ``estimated_output_bytes``, and
``peak_temporary_bytes`` are admitted before dispatch. A portable weight and
its packed result may coexist during prepacking, so an unbounded pipeline can
temporarily approach twice the model weight size plus scratch.

Current onnx-light implementation
---------------------------------

Plan representation
+++++++++++++++++++

:cpp:class:`PreparedExecutionPlan` stores loading, preparation, publication,
and invocation work in one dependency graph. The relevant task descriptors are:

.. list-table::
    :header-rows: 1
    :widths: 23 20 20 37

    * - Task
      - Scope
      - Resource
      - Meaning
    * - ``kReadPayload``
      - ``kSession``
      - ``kIo``
      - Reads one selected portable or packed payload.
    * - ``kPrepare``
      - ``kSession``
      - ``kCpu``
      - Produces the kernel-specific packed allocation.
    * - ``kPublish``
      - ``kSession``
      - ``kInline``
      - Makes the complete immutable object visible and satisfies its event.
    * - ``kExecute``
      - ``kInvocation``
      - ``kCpu``
      - Pins and consumes the exact prepared generation.
    * - ``kPersist``
      - ``kSession``
      - normally ``kIo``
      - Writes a reusable cache entry after publication.

``ExpandMaterializationRecipe`` creates ``read -> prepack -> publish`` for a
portable-weight miss. Dependencies are represented by ``TaskId`` values, not by
waiting inside a worker. A task enters the ready set only when every producer
has succeeded.

Pool ownership and dispatch
+++++++++++++++++++++++++++

The implementation combines three execution mechanisms:

* ``PreparedExecutionState::SchedulerState`` owns a persistent bounded
  :cpp:class:`utils::WorkerPool` for ``kIo`` tasks;
* all ``kCpu`` tasks use the session's existing :cpp:class:`CpuExecutor`;
* ``PrepareAsync`` and ``RunAsync`` use one coordinating ``std::thread`` owned
  by their :cpp:class:`ExecutionHandle`.

The coordinator sorts ready tasks by effective priority and then by scope.
Admitted I/O tasks are enqueued independently. Ready CPU tasks are collected
and submitted as one ``CpuExecutor::ParallelFor`` batch. While that synchronous
CPU batch prepackages ``W0``, the I/O workers can continue reading ``W1`` and
``W2``. Completion changes the dependency state, allowing their prepacks into a
later CPU batch.

CPU preparation does not create another compute pool. A prepack already running
on one executor participant installs the same executor view; nested kernel
parallelism therefore runs inline. This prevents a prepack from recursively
waking another full team, but it also means that a single task must expose
enough outer parallel work or be batched with other ready prepacks to use all
participants.

The scheduler implementation currently starts its ``WorkerPool`` when
``PreparedExecutionState`` constructs ``SchedulerState``. The separate
``utils::ThreadPool`` batch adapter used by model parsing starts its underlying
workers lazily on the first submitted task and stops them in ``Wait``. That
parse/load adapter is not the prepared-runtime scheduler: it has no dependency
priorities, memory admission, or persistent session submissions.

Concrete three-weight trace
+++++++++++++++++++++++++++

Assume two I/O workers, four CPU participants, and three weights ordered by
first use:

.. code-block:: text

    time --->

    I/O 0:  read W0 --------| read W2 ----------------|
    I/O 1:  read W1 ------------------|

    CPU:                    | pack W0 |
                                      | pack W1 | pack W2 |

    ready:                            W0         W1       W2

The exact intervals depend on payload sizes. The important events are:

1. ``read W0`` and ``read W1`` are admitted up to ``io_workers`` and the I/O
   byte budget.
2. Completion of ``read W0`` makes only ``prepack W0`` ready. The coordinator
   submits it to ``CpuExecutor`` without waiting for ``W1``.
3. The released I/O slot starts ``read W2`` while ``W0`` is packing.
4. ``publish W0`` runs only after the packed allocation is complete. Consumers
   can never observe a partial object.
5. If inference is submitted, the load/prepack chain for the next blocked
   layer inherits ``kCritical`` priority. Later speculative weights remain
   ``kPrefetch`` or ``kBackground``.

With ideal independent resources, loading and preparation approach:

.. math::

    T_{\mathrm{pipeline}} \gtrsim
    \max\left(T_{\mathrm{all\ reads}}, T_{\mathrm{all\ prepacks}}\right)
    + T_{\mathrm{fill/drain}},

rather than their sum. Memory bandwidth, page-cache traffic, NUMA placement,
hashing, and CPU contention make this only a lower bound.

Backpressure and priority
+++++++++++++++++++++++++

``PreparedSchedulerOptions`` bounds global, I/O, preparation, prepared, and
execution memory independently. Admission reserves
``reserved_critical_memory`` and, while critical work is pending, leaves one
I/O slot unavailable to speculative tasks. This prevents prefetch from
occupying every byte or worker needed by the next demanded weight.

The three priorities are:

* ``kCritical`` for invocation work and producer chains blocking it;
* ``kPrefetch`` for selected near-future weights;
* ``kBackground`` for distant variants and cache persistence.

Priority propagates backwards through dependencies before dispatch. It changes
which ready task is considered first; it does not interrupt a prepack already
running. Chunk sizes and the look-ahead window must therefore remain bounded so
critical work does not wait behind one very large speculative task.

OpenMP mapping
--------------

OpenMP can express the CPU side with tasks and dependencies:

.. code-block:: cpp

    #pragma omp task depend(out : loaded[i])
    read_weight(i);

    #pragma omp task depend(in : loaded[i]) depend(out : packed[i])
    prepack_weight(i);

The simple form has an important flaw: a blocking ``read_weight`` occupies an
OpenMP worker. If enough reads block, no worker remains to run a ready prepack.
Neither ``schedule(static)`` nor a faster fork-join barrier solves resource
starvation.

A robust OpenMP integration still uses a bounded I/O service outside the
OpenMP team. I/O completion then creates or releases CPU prepack tasks.
``depend`` provides ordering, but the application must still own:

* source and packed-buffer lifetimes;
* byte-based admission and backpressure;
* publication events visible outside the OpenMP region;
* priority inheritance from an inference consumer;
* cancellation and exception propagation.

An outer persistent OpenMP region can amortize team fork-join overhead, but a
prepack that internally opens another parallel region may serialize or create
a nested team depending on runtime settings. Integrations should choose one
parallelization level: parallel prepared objects at the scheduler level, or
parallel tiles inside one prepack, rather than multiplying both team sizes.

ONNX Runtime mapping
--------------------

ONNX Runtime normally creates session state and performs provider prepacking
before regular graph execution. Its intra-operator pool is suitable for CPU
packing work, and its optional inter-operator pool schedules independent graph
nodes, but neither pool is a substitute for bounded blocking I/O.

An overlapped design would need the same resource split:

.. code-block:: text

    dedicated I/O executor
        -> prepared-weight completion event
            -> ORT intra-op packing work
                -> immutable SessionState/EP object
                    -> consuming node

Using the inter-operator pool for reads would mix graph scheduling with blocking
storage work. Using a separate per-session CPU pool for prepacking would
oversubscribe the intra-operator pool when first inference begins. A shared
intra-operator pool avoids the second team, but needs priority or reserved
capacity so speculative packing does not delay a demanded node.

ORT's work-stealing pool can balance prepacks with different durations more
dynamically than ``onnx-light``'s static ``ParallelFor`` batch. Work stealing
alone does not supply prepared-object identity, atomic publication, memory
admission, or cross-session cache compatibility. Those belong to session
initialization and execution planning.

Operational choices
-------------------

.. list-table::
    :header-rows: 1
    :widths: 25 25 25 25

    * - Mode
      - Submission
      - First inference
      - Appropriate use
    * - Eager
      - ``PrepareAsync().Wait()``
      - Starts only after all required objects are resident
      - Stable latency after startup
    * - Streaming
      - ``PrepareAsync()`` and ``RunAsync()`` overlap
      - Waits only at the first missing exact dependency
      - Minimum time to first useful execution
    * - On demand
      - ``RunAsync()`` submits its required closure
      - Loads each object when first reached
      - Tight memory budgets or unpredictable branches
    * - Warm packed cache
      - Reads compatible packed payloads
      - Skips portable reads and prepacking
      - Repeated startup on the same CPU/kernel ABI

Measurements must report at least model resolution, first payload read, first
prepack, first published object, first executable node, first output, and fully
prepared time. A task trace should also include resource class, priority, bytes
in flight, executor identity, and whether each object came from portable
weights or the packed cache. Aggregate ``load()`` duration alone cannot prove
that loading and prepacking overlapped.

Implementation references
-------------------------

* ``onnx-light``:
  `prepared execution interface
  <https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_core/compute/prepared_execution.h>`_,
  `prepared scheduler
  <https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_core/compute/prepared_execution.cc>`_,
  `I/O worker pool
  <https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_proto/thread_pool.cc>`_,
  :ref:`l-next-steps-prepared-execution`, and
  :ref:`l-next-steps-native-fast-loading-completion`.
* ONNX Runtime:
  `thread management
  <https://onnxruntime.ai/docs/performance/tune-performance/threading.html>`_
  and `thread-pool construction
  <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/common/threadpool.cc>`_.
* OpenMP:
  `task dependence syntax
  <https://www.openmp.org/spec-html/5.2/openmpsu33.html>`_ and
  `LLVM task runtime
  <https://github.com/llvm/llvm-project/blob/main/openmp/runtime/src/kmp_tasking.cpp>`_.
