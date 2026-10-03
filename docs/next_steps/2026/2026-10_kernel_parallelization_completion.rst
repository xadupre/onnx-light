.. _l-next-steps-kernel-parallelization-completion:

Complete native kernel parallelization
======================================

:Date: 2026-10
:Updated: 2026-10-02

**started**

Objective
+++++++++

Finishes the native CPU kernel parallelization started by
:ref:`l-next-steps-kernel-parallelization`. Every kernel path with enough
independent work must use the session ``CpuExecutor`` through ``ParallelFor``;
every path that stays serial must have benchmark evidence and an explicit
reason. ``MatMul`` and ``Transpose`` are already parallel. This plan covers the
remaining migrations, ``Gemm`` tuning completion, cross-platform calibration,
and final acceptance.

Completion does not mean adding a worker launch to every operator. Shape-only,
metadata-only, control-flow, very small, or inherently ordered paths may remain
serial when the inventory records why parallel execution cannot win or cannot
preserve semantics.

Current baseline
++++++++++++++++

At the 2026-10-02 source revision,
``onnx_light.tools.kernel_inventory.build_kernel_inventory`` reports:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Coverage state
     - Paths
     - Meaning
   * - ``calibratable``
     - 37
     - A tuning schema and bounded calibration callback are registered.
   * - ``tunable``
     - 290
     - A tuning schema exists, but no runtime calibration callback is needed
       or implemented.
   * - ``parallel_fixed_policy``
     - 0
     - No remaining path may use an untracked compiled grain policy.
   * - ``serial``
     - 170
     - The source contains no ``ParallelFor`` call site. These paths need
       measurement, migration, or an accepted serial exemption.

The 170 serial entries are an inventory, not 170 mandatory implementations.
Several operators share one implementation family, while sequence, optional,
constant, shape, and control-flow operations may have no profitable independent
work. The inventory must therefore gain a reviewed disposition for each path
rather than treating source-text detection as proof that parallelism is useful.

Rules shared by every migration
+++++++++++++++++++++++++++++++

* Uses the active session ``CpuExecutor``; no kernel creates a thread pool,
  launches ``std::async``, or consults a process-global worker count.
* Partitions output ranges, reduction groups, batches, channels, heads, pages,
  or tiles with disjoint writes. Shared reduction state and atomic accumulation
  are rejected unless deterministic merging is part of the algorithm.
* Keeps the serial implementation and selects it below a named schema threshold.
  Grain, tile, participant, packing, and algorithm choices are represented by
  ``KernelTuningSchema`` parameters rather than hidden constants.
* Preserves exact error validation and the existing numerical contract.
  Deterministic kernels remain deterministic across participant counts.
* Uses bounded temporary storage. Per-worker scratch is included in the memory
  gate and cannot scale as ``threads * full_output_size``.
* Allows nested execution to reuse the current executor without
  oversubscription or deadlock.
* Adds C++ correctness tests for serial and parallel policies and benchmark
  cases large enough to cross the parallel threshold.
* Records runtime-event participants, work, grain, allocations, copies, and
  scratch where the kernel already exposes those costs.

Ordered implementation batches
+++++++++++++++++++++++++++++++

The batches are ordered by expected model impact. A batch starts only after the
previous batch has a published baseline and a reviewed list of serial
exemptions.

The first implementation wave is now in the source tree: ``Conv``,
``ConvTranspose``, ``ConvInteger``, ``QLinearConv``, ``Attention``,
``LinearAttention`` and ``FlexAttention`` use typed
``parallel.minimum_elements`` tuning and partition independent output planes
or batch/head recurrences. Attention scratch is allocated before worker
launches and sliced by task, so workers never concurrently access the
non-thread-safe runtime allocator. The migration is implemented; the batch
remains open until the required x86-64 and ARM64 crossover reports are
published and portable defaults are accepted.

.. list-table::
   :header-rows: 1
   :widths: 10 22 33 22 13

   * - Batch
     - Kernel families
     - Required parallel decomposition
     - Exit evidence
     - Status
   * - 1
     - ``Conv``, ``ConvTranspose``, ``ConvInteger``, ``QLinearConv``,
       ``Attention``, ``LinearAttention``, ``FlexAttention``
     - Output batches/channels/spatial tiles or query-head ranges, with bounded
       per-worker accumulation and no duplicate cache decoding.
     - CNN and transformer shapes beat or match serial execution on x86-64 and
       ARM64 without memory or determinism regressions.
     - Migration implemented; cross-platform calibration pending.
   * - 2
     - Reductions, ``Softmax``, ``LogSoftmax``, normalization, global/local
       pooling, ``TopK``
     - Independent outer rows or reduction groups; large single reductions use
       deterministic partials only when measurements justify the merge cost.
     - Scalar, empty, strided, dynamic-axis, low-precision, and large-axis
       correctness plus crossover measurements.
     - ``Softmax`` and ``LogSoftmax`` migration implemented; other families
       and crossover measurements pending.
   * - 3
     - ``Cast``, quantize/dequantize, ``Gather*``, ``Scatter*``, ``Where``,
       ``Pad``, ``Resize``, ``Slice``, ``Concat``, ``Split``, ``Tile``,
       ``Expand``
     - Contiguous copy/conversion blocks or independent destination regions.
       Scatter paths must prove writes cannot conflict before parallelizing.
     - Memory-bandwidth scaling, overlap/alias safety, and bounded participant
       counts across contiguous and non-contiguous cases.
     - ``Cast`` and numeric ``Where`` migrations implemented; other families
       and crossover measurements pending.
   * - 4
     - Recurrent operators, ``DFT``, ``STFT``, ``Einsum``, image/object
       detection, traditional-ML and training kernels
     - Batch, feature, tree, class, frequency, or parameter ranges where
       dependencies permit.
     - Representative backend models identify which paths merit migration;
       every retained serial path has measured justification.
     - Not started.
   * - 5
     - Sequences, optionals, text, metadata and remaining utility paths
     - Parallelize only payload-scale independent work. Preserve sequence
       ordering.
     - The inventory contains no unexplained serial path and no fixed-policy
       parallel path.
     - Not started.

The operator lists seed measurement; they do not authorize speculative
parallel code. Within each batch, rank families by serial wall time and model
attribution, then migrate the highest impact family first. A low-impact family
may receive a serial exemption without waiting for the rest of its batch.

Gemm and portable tuning completion
+++++++++++++++++++++++++++++++++++

``Gemm`` already uses ``ParallelFor`` and calibrates
``parallel.minimum_tasks``. Finish its contract by measuring whether
``tile_m``, ``tile_n``, ``tile_k``, packing thresholds, and FMA work units
produce stable wins. Promote a choice to the public tuning schema only when a
bounded calibrator can validate it; otherwise keep it an internal algorithm
constant and document that decision.

Run the native report workflow on x86-64 and ARM64 from the same commit. Compare
three runs per architecture, reject unstable candidates, and promote portable
defaults only when both architectures improve the declared corpus without a
material regression. Machine-specific winners remain persisted profiles keyed
by processor and execution descriptor.

Serial exemptions
+++++++++++++++++

Add a machine-readable disposition beside the inventory for every path that
remains ``serial``. Each exemption contains:

* the operator/domain and implementation family;
* the dependency that prevents safe partitioning, or benchmark shapes proving
  worker overhead dominates;
* the source revision, processor, execution policy, and raw measurements;
* the size or semantic boundary after which the exemption must be revisited;
* an owner category: ordered/control-flow, metadata-only, tiny payload,
  conflicting writes, deterministic RNG, or unsupported benchmark.

``validate_inventory`` fails when a serial path has no current disposition,
when a source change invalidates its implementation fingerprint, or when a
``parallel_fixed_policy`` path reappears.

Random generators are serial exemptions by design
++++++++++++++++++++++++++++++++++++++++++++++++++

``RandomNormal``, ``RandomNormalLike``, ``RandomUniform``,
``RandomUniformLike``, ``Bernoulli`` and ``Multinomial`` consume an ordered
pseudo-random sequence. Splitting their output across workers would change the
mapping between stream positions and tensor elements, and can also change how
many values are consumed by rejection-based algorithms. They remain serial so
seeded execution preserves the current bit-for-bit sequence and state
advancement. This plan does not introduce per-worker seeds, skip-ahead streams,
or a counter-based replacement generator.

Deterministic construction operators such as ``ConstantOfShape``, ``Range`` and
``EyeLike`` are not covered by this exemption. They may use parallel fill or
copy ranges when measurements show that the payload is large enough.

Benchmark and attribution contract
++++++++++++++++++++++++++++++++++

Expand ``kernel_baseline`` from the current small corpus to one representative
case per migration family. Each case records serial and session-thread latency,
CPU utilization, admitted and observed participants, grain, peak scratch,
allocations, and copies. Report small, crossover, and large shapes and preserve
the raw samples, not only medians.

For kernels exercised by ONNX Runtime, compare standalone ``onnx-light``, ORT
with protobuf ONNX, ORT with ``onnx-light``, and ORT format under the same model,
thread policy, affinity, and warmup. This attribution prevents a native
parallelization from being accepted when the measured bottleneck belongs to
model loading, layout conversion, or another execution provider.

Acceptance
++++++++++

This next step is complete when:

1. The generated inventory has zero ``parallel_fixed_policy`` paths and every
   serial path has a reviewed, fingerprinted exemption.
2. Every non-exempt payload-scale family uses the session executor and a named
   tuning schema with conservative portable defaults.
3. Targeted tests cover one and many participants, nested execution,
   cancellation, invalid inputs, aliasing, determinism, and bounded scratch.
4. Full native correctness passes under forced serial and default parallel
   policies, including sanitizers and reduced builds where applicable.
5. Published x86-64 and ARM64 reports use the same commit and corpus. No
   accepted default has a significant correctness, latency, throughput, memory,
   or oversubscription regression on either architecture.
6. ORT attribution is published for the representative model corpus.
7. ``Gemm`` parameters are either calibrated through validated schema entries
   or explicitly retained as internal constants with measurements.
8. The roadmap records final coverage counts, accepted defaults, persisted
   profiles, serial exemptions, and links to the raw reports.

Implementation unit
+++++++++++++++++++

Use one pull request per kernel family or tightly coupled group. Each pull
request carries its benchmark fixture, serial/parallel correctness coverage,
tuning schema, memory accounting, and before/after report. Batch-wide default
promotion follows only after both architecture reports are available; do not
combine speculative migrations into one CI cycle.
