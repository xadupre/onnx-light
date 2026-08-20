.. _l-next-steps-parallel-for-profiling:

ParallelFor profiling and hardware counters
===========================================

:Date: 2026-08

**implementation ready**

Objective
+++++++++

The objective is to make parallel-loop under-utilization observable without
adding mandatory overhead to every ``ParallelFor`` call. This refines
`issue #4282 <https://github.com/xadupre/onnx-light/issues/4282>`_ into two
layers:

* portable loop events with source location, work decomposition, elapsed time,
  process CPU time, and normalized CPU utilization;
* optional platform collectors for cycles, instructions, and last-level-cache
  counters.

The instrumentation is diagnostic. It may explain a tuning result, but it must
not silently change a kernel threshold or worker count while inference runs.

Next implementation batch
+++++++++++++++++++++++++

Profile PR01 is the next runtime implementation batch. It is intentionally
limited to portable, session-owned region events:

* add ``ParallelRegionEvent`` and a fixed-capacity
  ``ParallelRegionCollector`` under ``onnx_core/runtime/tuning``;
* add an optional collector to ``RuntimeSessionOptions`` and install its
  non-owning view beside the session ``CpuExecutor`` for each run;
* extend the public ``ParallelFor`` wrapper with an optional label and trailing
  ``std::source_location`` while preserving every existing call site;
* record total iterations, grain size, requested, admitted, and observed
  participants, wall time, executor identity, source location, and whether the
  region ran nested-inline;
* report dropped events after the configured capacity is exhausted rather than
  allocating or blocking;
* prove that the disabled path performs no clock read, allocation, collector
  lock, or event construction.

The implementation hooks belong in ``tuning/cpu_executor.{h,cc}`` and
``kernels/parallel_for.{h,cc}``; session ownership belongs in
``runtime_session.{h,cc}``. Unit tests cover serial, limited, parallel,
nested-inline, bounded-capacity, and disabled cases. A focused benchmark
compares the disabled median against the current executor baseline.

Profile PR01 explicitly excludes process CPU time, Python bindings,
``perf_event_open``, calibration integration, and platform hardware-counter
backends. Those additions depend on the portable event contract and land in
later batches.

Metric definitions
++++++++++++++++++

For one profiled parallel region:

.. code-block:: text

    cpu_utilization = process_cpu_time / (wall_time * effective_threads)
    ipc             = retired_instructions / cpu_cycles
    llc_miss_rate   = llc_misses / llc_references

``effective_threads`` is the number of participants admitted to that region,
not ``std::thread::hardware_concurrency()``. The event also records requested,
admitted, and observed participant counts so a consumer can distinguish a
small task from scheduler under-utilization.

CPU utilization is descriptive rather than a universal efficiency score. SMT,
hybrid cores, frequency scaling, memory stalls, blocking I/O, preemption, and
other process threads can keep it below or above an intuitive target. No fixed
``0.8`` threshold is valid for every kernel or machine.

IPC and LLC miss rate are present only when the platform collector reports
valid counters. Missing permissions, unsupported events, multiplexing, and
counter overflow are explicit statuses; they never become zero-valued
measurements.

Portable event contract
+++++++++++++++++++++++

An opt-in collector receives one bounded event per region:

.. code-block:: cpp

    struct ParallelRegionEvent {
      uint64_t region_id;
      std::string_view label;
      std::source_location location;
      int64_t total_iterations;
      int64_t grain_size;
      int32_t requested_threads;
      int32_t admitted_threads;
      int32_t observed_threads;
      uint64_t wall_time_ns;
      uint64_t process_cpu_time_ns;
      std::optional<double> cpu_utilization;
      HardwareCounterSample counters;
    };

``ParallelFor`` remains source-compatible and returns ``void``. A final
``std::source_location`` argument defaults to
``std::source_location::current()``, and a lightweight label is optional.
Events are appended to a session-owned collector or passed to a callback.
Returning a result from every loop would break existing call sites and make
nested loops difficult to aggregate.

When profiling is disabled, the hot path performs at most one predictable
disabled check. It does not read clocks, allocate, copy file names, lock a
mutex, or access hardware counters.

Nested and concurrent regions
+++++++++++++++++++++++++++++

Every event has a region identifier, optional parent identifier, session/run
identifier, and calling thread identifier. Nested regions that execute inline
still emit their actual admitted participant count. Concurrent sessions write
to independent collectors or bounded per-thread buffers and merge only when
the caller requests the report.

The collector records dropped-event counts when its capacity is exhausted. It
must never block inference or grow without a configured bound.

Hardware-counter backend
++++++++++++++++++++++++

Linux support should use ``perf_event_open`` with one grouped collector for
cycles, retired instructions, LLC references, and LLC misses. It records
``time_enabled`` and ``time_running`` and either scales multiplexed counts or
marks them unsuitable according to an explicit policy.

Windows and macOS need separate backends. Until implemented, they return
``unsupported`` while portable timing remains available. Counter collection is
disabled by default because permissions, system configuration, and
virtualization commonly prevent reliable access.

Process CPU time is portable only as an aggregate across process threads. It is
acceptable for isolated diagnostic runs but can include unrelated concurrent
work. A later worker-level collector may sum per-thread CPU clocks when the
thread pool can identify every participant without adding steady-state cost.

Integration with runtime events and tuning
++++++++++++++++++++++++++++++++++++++++++

Parallel-region events should use a dedicated bounded stream linked to the
existing runtime run/node identifiers. The current tensor-mutation
``RuntimeEvent`` remains unchanged; embedding high-frequency loop records in
that payload would mix value lifetime with scheduler telemetry.

Calibration may request the collector and include the resulting metrics in its
diagnostics. Candidate selection remains based on validated outputs and stable
elapsed-time wins. IPC or LLC misses can explain a choice but cannot override a
slower candidate automatically.

Benchmark and acceptance criteria
+++++++++++++++++++++++++++++++++

The benchmark covers serial, fully occupied, memory-bound, compute-bound,
nested, concurrent, and intentionally imbalanced loops. It validates metric
formulas against known participant counts and compares disabled and enabled
overhead.

Acceptance requires:

* disabled instrumentation adds no allocation, lock, clock read, or counter
  syscall and changes a large-loop median by less than measurement noise;
* source location identifies the caller without macros;
* portable events report truthful requested/admitted/observed participants;
* unsupported or permission-denied counters are distinguishable from zero;
* nested and concurrent regions retain parent/run identity;
* bounded collectors report dropped events without blocking;
* hardware-counter values agree with ``perf stat`` within a documented
  tolerance on an isolated Linux benchmark.

Implementation sequence
+++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 12 31 42 15

   * - PR
     - Scope
     - Merge criterion
     - Status
   * - Profile PR01
     - Portable event contract and bounded session collector.
     - Disabled execution has no instrumentation work; enabled serial,
       parallel, limited, and nested regions emit truthful bounded events.
     - Ready
   * - Profile PR02
     - Run/parent identity, process CPU time, and normalized utilization.
     - Nested and concurrent regions retain identity and report explicit metric
       validity.
     - Planned
   * - Profile PR03
     - C++ report API and Python inspection.
     - Bounded events and dropped counts are inspectable without exposing
       mutable collector storage.
     - Planned
   * - Profile PR04
     - Linux grouped ``perf_event_open`` collector.
     - Unsupported, denied, multiplexed, and valid samples remain
       distinguishable and agree with ``perf stat`` within tolerance.
     - Planned
   * - Profile PR05
     - Calibration diagnostics integration.
     - Metrics explain candidates but never override correctness or elapsed
       time selection.
     - Planned
   * - Profile PR06
     - Additional platform backends.
     - A backend lands only with equivalent documented and tested semantics.
     - Optional
