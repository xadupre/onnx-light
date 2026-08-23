.. _l-next-steps-kernel-parallelization:

Kernel parallelization and tuning sequence
==========================================

:Date: 2026-08

**Step G in progress (x86-64 calibrated; ARM64 pending hardware access)**

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
      - Started (``onnx_light.tools.kernel_inventory`` and
        ``onnx_light.tools.kernel_baseline`` cover every native kernel path
        and the x86-64 baseline report; the ORT-attribution deliverable from
        Step D is still outstanding, so this evidence only ranks native
        ``onnx-light`` kernels)
    * - F. Kernel migration
      - Step E
      - A serial implementation and one or more bounded parallel candidates
        for each selected kernel, all controlled by named tuning parameters.
      - Calibration needs valid candidates with identical numerical and error
        behavior.
      - Started (``Gemm`` already ran through ``ParallelFor`` with tunable
        tile and grain parameters; it now also registers a calibration
        candidate for ``parallel.minimum_tasks`` mirroring the unary
        ``CalibrateAbs`` pattern. Calibrating the remaining ``Gemm``
        parameters (tile/pack sizes) and promoting any candidate to a
        portable default remain outstanding and belong to the next issue)
    * - G. Calibration
      - Step F
      - Validated processor-specific profiles published through
        ``CalibrateRegisteredKernels`` and persisted with
        ``UpdateKernelTuningCache``.
      - Rollout needs repeatable decisions that a later session can load
        without benchmarking during inference.
      - Started (every calibratable key -- ``Abs``, ``Add``, ``Gemm``,
        ``Log``, ``Not``, ``Sigmoid``, ``Tanh`` -- was calibrated on the
        x86-64 sandbox machine with ``python -m onnx_light tune-kernels
        --apply`` and persisted through ``UpdateKernelTuningCache``; a fresh
        process reloaded that cache and resolved every one of the 37
        calibrated keys through their exact processor and execution
        descriptor. None of the calibrated values were promoted to portable
        schema defaults, because only one architecture was measured; see
        ``kernel_parallelization_reports/x86_64_calibration.json``. An ARM64
        machine profile remains outstanding, blocked on hardware access)
    * - H. Acceptance
      - Step G
      - Cross-platform correctness, determinism, memory, latency, throughput,
        nesting, and oversubscription results, compared with the matching ORT
        baselines from Step C.
      - Only candidates that improve the declared workload without regressions
        become defaults or published profiles.
      - Planned (blocked on an ARM64 measurement to compare against the
        x86-64 calibration report)

Assignable issue sequence
+++++++++++++++++++++++++

The first implementation cycle is:

`#4669 <https://github.com/xadupre/onnx-light/issues/4669>`_ baseline and
inventory -> `#4670 <https://github.com/xadupre/onnx-light/issues/4670>`_
tuning coverage -> `#4671 <https://github.com/xadupre/onnx-light/issues/4671>`_
first measured migration batch ->
`#4672 <https://github.com/xadupre/onnx-light/issues/4672>`_ cross-machine
calibration and default promotion.

#4669 and #4670 are closed. Each later issue states its prerequisite and must
remain unassigned until that prerequisite is closed.

Coverage states
+++++++++++++++

``onnx_light.tools.kernel_inventory`` enumerates every registered native
kernel path from the built-in dispatch table and assigns it exactly one of
the following coverage states, recorded once per ``(domain, op_type, device,
element_type)`` path. ``validate_inventory()`` guarantees no path is left
unclassified.

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

``onnx_light.tools.kernel_baseline`` runs a deliberately small and
representative benchmark corpus rather than an exhaustive one: one
memory-bound unary kernel (``Abs``), one compute-bound kernel with an
existing tuning schema (``Gemm``), and one boolean/logical kernel (``Not``),
each measured at a small, medium, and large shape under a forced serial
policy (``CpuExecutionPolicy.num_threads = 1``) and the default
session-thread policy (``CpuExecutionPolicy.num_threads = 0``). Every case
reports the CPU descriptor, executor policy, wall time, process CPU
utilization, requested/admitted/observed participants, grain size, and
hardware counters when the platform collector supports them (Linux only in
this version; other platforms report ``unsupported`` rather than fabricating
zero values). Model construction (``startup``) and steady-state execution
(``kernel execution``) are timed separately, and the tool never invokes
``onnxruntime``, so only native ``onnx-light`` kernels enter the migration
ranking -- it does not substitute for Step D's ORT attribution. Running the
corpus never writes to the kernel tuning cache: it only reads
``kernel_tuning_parameters()`` and constructs ordinary ``RuntimeSession``
instances. It is exposed as ``python -m onnx_light kernel-baseline``.

Published machine reports live under
``docs/next_steps/2026/kernel_parallelization_reports/``. Each file records
the host CPU descriptor in its own JSON payload so that an x86-64 and an
ARM64 report can be compared using the same schema and benchmark cases
without merging them into a single file. An x86-64 baseline is published;
an ARM64 report is pending access to hardware.

First migration batch
++++++++++++++++++++++

Ranking the x86-64 baseline (see
``kernel_parallelization_reports/x86_64_baseline.json``) by large-shape
serial-vs-session-thread speedup and by absolute large-shape wall time
identified ``Gemm`` (``FLOAT``, ``DOUBLE``, ``FLOAT16``, ``BFLOAT16``,
already ``tunable`` via ``KernelTuningSchema`` but still running with its
portable single-block defaults) as the largest compute-bound outlier, and the
then-``parallel_fixed_policy`` transcendental unary kernels ``Log``,
``Tanh``, and ``Sigmoid`` (``Exp`` already migrated to ``tunable``, alongside
``Abs``) as the next fixed-grain-size candidates: they showed the largest gap
between the serial and session-thread policies of the sampled kernels.

Every remaining ``parallel_fixed_policy`` path in the built-in kernel
dispatch table (``onnx_light.tools.kernel_inventory`` reports zero of them)
has since been migrated: ``Log``, ``Tanh``, and ``Sigmoid`` each register a
``KernelTuningSchema`` with a calibration callback mirroring ``Abs``'s
``CalibrateAbs``, and every other fixed-grain unary math kernel (``Acos``,
``Acosh``, ``Asin``, ``Asinh``, ``Atan``, ``Atanh``, ``Ceil``, ``Cos``,
``Cosh``, ``Erf``, ``Floor``, ``HardSwish``, ``Mish``, ``Neg``,
``Reciprocal``, ``Relu``, ``Round``, ``Sign``, ``Sin``, ``Sinh``,
``Softplus``, ``Softsign``, ``Sqrt``, and ``Tan``) registers the same
portable ``parallel.minimum_elements`` schema as the logical kernels
(``And``, ``Or``, ...), without a calibration callback.

``Gemm`` now registers a bounded ``CalibrateGemm`` candidate for
``parallel.minimum_tasks`` (the threshold on the ``ParallelFor`` task-grid
size, tested with fixed ``tile_m``/``tile_n``/``k`` at the portable
defaults), mirroring the unary ``CalibrateAbs`` crossover search: reference
and candidate share the same deterministic tiled accumulation order, so their
outputs are bit-identical regardless of the selected threshold, and the
candidate never exceeds a bounded duration or memory budget. Calibrating
``Gemm``'s remaining algorithm parameters
(``algorithm.tile_m/tile_n/tile_k``, ``algorithm.pack_b_minimum_elements``,
``parallel.fmas_per_work_unit``) and the newly registered
``parallel.minimum_elements`` schemas above where the baseline shows a
measurable gap remain outstanding; promoting any winning candidate to a
portable default belongs to the next issue, along with confirming the
ranking against an ARM64 report once one is available. A C++ test
(``KernelClass.GemmCalibratesParallelMinimumTasksThreshold``) exercises
``CalibrateGemm`` through ``CalibrateRegisteredKernels`` and asserts the
published candidate validates against the registered schema.

Cross-machine calibration and default promotion
+++++++++++++++++++++++++++++++++++++++++++++++

Every calibratable key registered after Step F (``Abs``, ``Add``, ``Gemm``,
``Log``, ``Not``, ``Sigmoid``, and ``Tanh``, 37 ``(kernel, element_type)``
keys in total) was calibrated on the x86-64 sandbox machine with
``python -m onnx_light tune-kernels --apply`` and persisted through
``UpdateKernelTuningCache``. A fresh process then reloaded that cache with
``load_kernel_tuning_cache``/``kernel_tuning_parameters`` and resolved all 37
profiles through their exact ``KernelTuningKey`` and
``CpuExecutionDescriptor`` identity, with zero incompatible or invalid
entries -- proving persisted profiles survive a process restart without
recalibration or extra registry access. See
``kernel_parallelization_reports/x86_64_calibration.json`` for the complete
selected values and per-key diagnostics.

This pass fixed a real cross-process matching defect uncovered while
verifying the reload: ``KernelTuningCacheOptions::execution``, when left
unset (the common case for an ad hoc calibration run), fell back to the
processor's raw logical-core count in ``CurrentExecutionDescriptor()``,
while a default (``num_threads == 0``) ``RuntimeSession`` and
``ParallelForThreadCount()``'s no-executor fallback both resolve to
``RuntimeParameters::EffectiveNumThreads()`` (physical cores when
detected). On any machine with simultaneous multithreading these two counts
differ, so a profile persisted without an explicit execution descriptor
could never be found again by the default query path. ``CurrentExecutionDescriptor()``
now reuses ``RuntimeParameters::EffectiveNumThreads()``, and
``KernelTuningCache.DefaultExecutionDescriptorMatchesDefaultSessionThreadCount``
guards the fix.

None of the calibrated x86-64 values were promoted to the portable schema
defaults in ``portable_parallel_tuning.cc``/``portable_gemm_tuning.cc``:
only one architecture was measured in this pass, and promoting an untested
value risks an undeclared regression on ARM64, which the acceptance
criteria for this step explicitly forbid. Instead, the calibrated values
stay in the persisted machine cache, where they are selected only for the
exact processor and execution descriptor recorded during calibration --
this machine already benefits from them (for example unary
``parallel.minimum_elements`` dropped from the portable default of 32768 to
8192, and ``Gemm``'s ``parallel.minimum_tasks`` dropped from 2 to 1) without
changing behavior anywhere else. Publishing an ARM64 report and comparing it
against ``x86_64_calibration.json`` is the next measured step before any
default promotion; it remains blocked on ARM64 hardware access, matching the
Step E baseline's outstanding ARM64 gap.

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
