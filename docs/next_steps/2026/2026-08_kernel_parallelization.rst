.. _l-next-steps-kernel-parallelization:

Kernel parallelization and tuning
==================================

:Date: 2026-08

**Steps D-E in progress**

Objective
+++++++++

``onnx-light`` should keep growing the set of native kernels that run
efficiently on many cores while remaining reproducible across machines. Three
capabilities already exist independently: a persistent CPU thread pool and
shared session execution policies (:ref:`l-next-steps-session-execution-pools`),
processor-aware tuning thresholds
(:ref:`l-next-steps-processor-aware-kernel-tuning`), and portable
``ParallelFor`` profiling with optional hardware counters
(:ref:`l-next-steps-parallel-for-profiling`). This roadmap sequences the work
that consumes all three to decide *which* kernel to parallelize or tune next,
instead of adding another ad-hoc benchmark script per kernel.

Steps
+++++

.. list-table::
   :header-rows: 1
   :widths: 10 22 52 16

   * - Step
     - Scope
     - Result
     - Status
   * - Step A
     - Persistent thread pool and shared session CPU policy.
     - ``ThreadPool``, ``CpuExecutor``, ``CpuExecutionPolicy``; see
       :ref:`l-next-steps-session-execution-pools`.
     - Complete
   * - Step B
     - Processor-aware tuning registry, calibration, and cache.
     - ``KernelTuningRegistry``, ``KernelTuningSchema``,
       ``CalibrateRegisteredKernels``; see
       :ref:`l-next-steps-processor-aware-kernel-tuning`.
     - Complete
   * - Step C
     - Portable ``ParallelFor`` region events and Linux hardware counters.
     - ``ParallelRegionCollector``, ``ParallelRegionReportEvent``; see
       :ref:`l-next-steps-parallel-for-profiling`.
     - Complete
   * - Step D
     - Enumerate every registered kernel path and classify its coverage state.
     - ``onnx_light.tools.kernel_inventory`` builds one inventory entry per
       ``(domain, op_type, device, element_type)`` path from the built-in
       dispatch table, the tuning registry, and static detection of
       ``ParallelFor`` call sites in the owning kernel source file.
     - Complete
   * - Step E
     - Deterministic benchmark corpus and cross-machine baseline report.
     - ``onnx_light.tools.kernel_baseline`` runs a fixed shape corpus under
       explicit serial and session-thread CPU policies and emits one
       machine-readable report combining the Step D inventory with wall time,
       process CPU utilization, participants, and grain size. Exposed as
       ``python -m onnx_light kernel-baseline``.
     - Started (x86-64 baseline published; ARM64 pending access to hardware)

Coverage states
+++++++++++++++

Every enumerated kernel path receives exactly one of the following states,
recorded once per ``(domain, op_type, device, element_type)`` path:

``serial``
    No ``ParallelFor`` call site was found in the kernel's source file and no
    tuning schema is registered for it. The kernel always runs on the calling
    thread; a ``serial_reason`` field records why (e.g. control-flow operators
    that recurse into another session, or operators whose per-call cost never
    justifies worker wake-up).

``parallel_fixed_policy``
    The kernel calls ``ParallelFor`` but has no registered tuning schema: the
    grain size and participant limits are compiled constants
    (``kParallelForGrainSize`` and friends), not processor-specific.

``tunable``
    The kernel registers a ``KernelTuningSchema``: its named thresholds have
    portable defaults and may be overridden by a persisted profile.

``calibratable``
    A subset of ``tunable`` paths that additionally register a
    ``KernelCalibrationFunction`` (``RegisterKernelTuningSchema`` +
    calibration callback), so ``CalibrateRegisteredKernels`` can measure a
    value instead of requiring a hand-picked one.

Benchmark corpus and machine reports
+++++++++++++++++++++++++++++++++++

The Step E corpus intentionally stays small and representative rather than
exhaustive: one memory-bound unary kernel (``Abs``), one compute-bound
kernel with an existing tuning schema (``Gemm``), and one boolean/logical
kernel (``Not``), each measured at a small, medium, and large shape under a
forced serial policy (``CpuExecutionPolicy.num_threads = 1``) and the default
session-thread policy (``CpuExecutionPolicy.num_threads = 0``). Every case
reports the CPU descriptor, executor policy, wall time, process CPU
utilization, requested/admitted/observed participants, grain size, and
hardware counters when the platform collector supports them (Linux only in
this version; other platforms report ``unsupported`` rather than fabricating
zero values). Model construction (``startup``) and steady-state execution
(``kernel execution``) are timed separately, and the tool never invokes
``onnxruntime``, so only native ``onnx-light`` kernels enter the migration
ranking. Running the corpus never writes to the kernel tuning cache: it only
reads ``kernel_tuning_parameters()`` and constructs ordinary
``RuntimeSession`` instances.

Published machine reports live under
``docs/next_steps/2026/kernel_parallelization_reports/``. Each file records
the host CPU descriptor in its own JSON payload so that an x86-64 and an
ARM64 report can be compared using the same schema and benchmark cases
without merging them into a single file.

First migration batch
++++++++++++++++++++++

Ranking the x86-64 baseline (see
``kernel_parallelization_reports/x86_64_baseline.json``) by large-shape
serial-vs-session-thread speedup and by absolute large-shape wall time
identifies ``Gemm`` (``FLOAT``, already tunable but currently using its
portable single-block default) and the still ``parallel_fixed_policy``
transcendental unary kernels (``Exp``, ``Log``, ``Tanh``, ``Sigmoid``) sharing
``Abs``'s fixed grain size as the first concrete migration candidates: they
show the largest large-shape wall time and the largest gap between the
serial and session-thread policies of the sampled kernels. The next
implementation batch should give ``Exp``, ``Log``, ``Tanh``, and ``Sigmoid``
their own ``KernelTuningSchema`` (mirroring ``Abs``'s ``CalibrateAbs``) before
selecting further kernels, and confirm the ranking against an ARM64 report
once one is available.
