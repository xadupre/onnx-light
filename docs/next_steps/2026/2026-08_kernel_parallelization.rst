.. _l-next-steps-kernel-parallelization:

Kernel parallelization and tuning sequence
==========================================

:Date: 2026-08

**planned**

Objective
+++++++++

Parallelize kernels from measured evidence without adding hidden worker pools,
fixed machine-specific thresholds, or runtime calibration. Every migrated
kernel uses the session ``CpuExecutor`` and the processor-aware tuning API.

Dependency chain
++++++++++++++++

The work proceeds in this order. Each step consumes the output of the previous
step and produces the input required by the next one.

.. list-table::
    :header-rows: 1
    :widths: 8 18 29 29 16

    * - Step
      - Requires
      - Produces
      - Why the next step depends on it
      - Status
    * - A. Stable execution
      - Session CPU policy
      - One ``CpuExecutor``, its effective participant count, and a stable
        execution identity.
      - Measurements and tuning profiles must identify the workers that
        actually execute the kernel.
      - Complete
    * - B. Measurement
      - Step A
      - Bounded ``ParallelFor`` events with work size, grain, participants,
        elapsed time, CPU utilization, and optional hardware counters.
      - A kernel cannot choose useful tuning candidates until its
        under-utilization or contention is observable.
      - Complete
    * - C. Tuning contract
      - Steps A and B
      - A validated ``KernelTuningSchema``, portable defaults, immutable
        snapshots, calibration callbacks, and persistent processor profiles.
      - Kernel changes need one reproducible API for serial thresholds, grains,
        tiles, algorithms, packing, and participant limits.
      - Complete
    * - D. Consumer attribution
      - Steps A--C, plus the benchmark contract from
        :ref:`l-next-steps-model-loading`
      - Comparable results for standalone ``onnx-light``, ORT with protobuf,
        ORT with ``onnx-light``, and ORT ``.ort`` using the same model and
        execution policy.
      - The project must first decide whether the bottleneck belongs to a
        native kernel, the loading and ownership boundary, or an ORT execution
        provider.
      - Planned
    * - E. Kernel baseline
      - Steps A--D
      - A ranked inventory of expensive serial regions and inefficient
        parallel regions, including correctness inputs and benchmark shapes.
      - Migration order must follow measured impact rather than source-file
        order or intuition.
      - Planned
    * - F. Kernel migration
      - Step E
      - A serial implementation and one or more bounded parallel candidates
        for each selected kernel, all controlled by named tuning parameters.
      - Calibration needs valid candidates with identical numerical and error
        behavior.
      - Planned
    * - G. Calibration
      - Step F
      - Validated processor-specific profiles published through
        ``CalibrateRegisteredKernels`` and persisted with
        ``UpdateKernelTuningCache``.
      - Rollout needs repeatable decisions that a later session can load
        without benchmarking during inference.
      - Planned
    * - H. Acceptance
      - Step G
      - Cross-platform correctness, determinism, memory, latency, throughput,
        nesting, and oversubscription results, compared with the matching ORT
        baselines from Step C.
      - Only candidates that improve the declared workload without regressions
        become defaults or published profiles.
      - Planned

Assignable issue sequence
+++++++++++++++++++++++++

The first implementation cycle is:

`#4669 <https://github.com/xadupre/onnx-light/issues/4669>`_ baseline and
inventory -> `#4670 <https://github.com/xadupre/onnx-light/issues/4670>`_
tuning coverage -> `#4671 <https://github.com/xadupre/onnx-light/issues/4671>`_
first measured migration batch ->
`#4672 <https://github.com/xadupre/onnx-light/issues/4672>`_ cross-machine
calibration and default promotion.

Only #4669 is immediately assignable. Each later issue states its prerequisite
and must remain unassigned until that prerequisite is closed.

Per-kernel implementation loop
++++++++++++++++++++++++++++++

Step F repeats the following sequence for one measured kernel family:

1. Record a serial baseline and representative shapes with the profiling API.
2. Separate the kernel into a deterministic range or tile operation that can
   run through ``ParallelFor`` on the active session executor.
3. Register every choice in ``KernelTuningSchema``. At minimum this includes
   the serial threshold, grain or tile size, and maximum participants; algorithm
   and packing choices are added when the kernel has multiple implementations.
4. Keep conservative portable defaults so an unknown processor remains correct
   and avoids pathological oversubscription.
5. Add a calibration callback that benchmarks only schema-valid candidates,
   validates their outputs against the serial reference, and reports the
   measurements that explain the selection.
6. Publish the winning immutable profile, persist it through the tuning cache,
   create a new session, and prove that the session resolves the same parameters
   without registry access or calibration on its execution hot path.
7. Run correctness and performance gates. If no candidate wins, retain the
   serial implementation and keep the profiling evidence.

ONNX Runtime decision boundary
++++++++++++++++++++++++++++++

Step D is mandatory before opening a kernel migration:

* if standalone ``onnx-light`` execution is slow and its profiled kernel region
  accounts for the difference, continue with Steps E--H in this plan;
* if ORT with ``onnx-light`` regresses against ORT with protobuf before session
  readiness, follow :ref:`l-next-steps-model-loading` and fix the payload,
  ownership, or parser boundary instead of tuning a kernel;
* if ORT session execution is slow in both loading configurations, the owner is
  the selected ORT execution provider; propose and validate that kernel change
  in ``microsoft/onnxruntime`` rather than adding an ``onnx-light`` tuning key;
* if the result improves only against ordinary ``.onnx`` but not against the
  matching ORT ``.ort`` baseline, report it as a loading-format tradeoff rather
  than a kernel speedup.

The repositories share benchmark inputs and acceptance metrics, not kernel
tuning state. ``onnxruntime_USE_ONNX_LIGHT`` replaces protobuf and ONNX model
handling; it does not make ORT execution-provider kernels consume
``KernelTuningParameters``. The ownership-aware ORT integration may consume
prepared payloads, but ORT remains responsible for its own kernel scheduler and
algorithm choices.

API boundary
++++++++++++

The tuning API owns decisions; the executor only applies them:

* ``KernelTuningSchema`` defines names, types, ranges, portable defaults, and
  cross-parameter validation.
* ``KernelTuningParameters`` carries the selected threshold, decomposition,
  algorithm, packing, and participant-limit values.
* ``CalibrateRegisteredKernels`` compares valid candidates outside inference.
* ``UpdateKernelTuningCache`` and ``LoadKernelTuningCache`` persist and restore
  profiles selected for an explicit processor and execution descriptor.
* ``RuntimeSession`` captures one immutable registry snapshot while preparing
  kernels. A kernel must not query or modify the registry while it executes.
* ``CpuExecutor`` and ``ParallelFor`` enforce the session limit and nested-inline
  behavior. A kernel may request fewer participants but never create a private
  pool or exceed the session policy.

Migration priority
++++++++++++++++++

Do not prescribe a static operator list before Step E. Rank candidates from the
same benchmark corpus by total CPU time, parallel-region utilization, cache
behavior, and frequency. Start with a small representative family, complete
Steps F--H, then repeat for the next measured bottleneck. This keeps the plan
useful when workloads or hardware change and makes every migration independently
revertible.

Definition of done
++++++++++++++++++

The plan is complete when every kernel selected by the baseline either has a
validated tuning schema and accepted parallel implementation or a recorded
reason to remain serial; all accepted kernels use the session executor, load
their immutable parameters before execution, and perform no calibration or
tuning-registry access on the hot path.
