.. _l-design-kernel-tuning:

Processor-aware kernel tuning
=============================

Purpose and boundary
++++++++++++++++++++

Kernel tuning selects performance parameters without changing operator
semantics. ``onnx_core`` owns processor detection, schemas, profile resolution,
calibration orchestration, immutable publication, and cache persistence.
Kernel libraries own parameter names, portable defaults, validation rules,
processor profiles, and calibration callbacks.

Every tunable value has a compiled portable default. Cache absence, corruption,
incompatibility, or validation failure therefore affects performance only; it
cannot prevent execution.

Exact tuning identity
+++++++++++++++++++++

``KernelTuningKey`` identifies one implementation:

* library and kernel names;
* implementation name;
* ONNX element type;
* device;
* tuning ABI.

The registry contains one ``KernelTuningSchema`` for each exact key. A schema
defines the complete set of named, typed values and their portable defaults.
Registering the same exact key twice is rejected. Different element types,
implementations, or ABI revisions use distinct schemas and may resolve to
different values.

Profiles add a ``CpuExecutionDescriptor`` containing the detected processor and
effective thread count. The cache may consequently contain several profiles
for one tuning key, provided their execution descriptors differ.

Registration and resolution
+++++++++++++++++++++++++++

Kernel libraries register schemas and optional calibration callbacks before
sessions are created. During the first initialization of a
``RuntimeSession``:

1. dispatch constructs the selected kernel implementation;
2. the kernel returns its exact tuning key for the resolved element type;
3. the session captures one immutable registry generation;
4. the registry resolves values for the processor and effective thread count;
5. the kernel validates and copies those values into its typed configuration.

Resolution precedence is deterministic:

1. an explicitly published or calibrated execution profile;
2. an exact vendor/family/model processor profile;
3. a processor-list or microarchitecture profile;
4. an instruction-set profile;
5. the portable defaults.

Priority only breaks ties between profiles of equal specificity. Ambiguous
registrations are rejected. Existing sessions retain their captured generation;
steady-state kernel execution never reads the registry or cache.

Calibration
+++++++++++

Calibration callbacks are trusted native functions registered for exact tuning
keys. The shared unary/binary crossover search:

* generates deterministic inputs;
* compares every candidate with the forced serial implementation;
* validates outputs before accepting timings;
* uses median measurements and consecutive wins;
* enforces duration, memory, and thread budgets;
* keeps the portable value when no stable crossover is found.

Schemas without callbacks remain manually tunable but cannot be calibrated
automatically. ``Abs``, ``Add``, ``Gemm``, ``Log``, ``Not``, ``Sigmoid``, and
``Tanh`` provide callbacks. ``Gemm`` uses its own bounded task-grid search;
the elementwise kernels use the shared crossover search.

Cache and Python lifecycle
++++++++++++++++++++++++++

The cache is a versioned text file keyed by tuning key and execution
descriptor. Updates validate complete profiles, use an inter-process lock,
merge unrelated entries, and atomically replace the file.

The full Python extension registers built-in schemas and loads compatible
profiles from the default cache when imported. Explicit cache paths require
``load_kernel_tuning_cache``. Python exposes:

* ``kernel_tuning_parameters`` for schemas, defaults, matching cache values,
  and active values;
* ``inspect_kernel_tuning_cache`` for non-mutating cache inspection;
* ``remove_kernel_tuning_cache`` for race-safe removal of the persisted cache;
* ``set_kernel_tuning_parameters`` for validated partial updates;
* ``calibrate_kernel_tuning`` for one selected kernel;
* ``propose_kernel_tuning_updates`` and ``apply_kernel_tuning_updates`` for
  local coverage.

The proposal workflow defines a key as covered when the selected cache contains
a profile matching the local processor and effective thread count. It separates
missing keys with callbacks from schema-only keys requiring manual work.
Applying proposals is always explicit.

The ``python -m onnx_light kernel --kernel NAME --tune`` command calibrates
and persists one exact selected kernel. Bulk proposal and application remain
available through the Python API.

An explicit ``--parameter NAME=default,VALUE,...`` comparison reuses the same
kernel-specific callback workload. Every value runs on the same bounded cases;
the current active value is the first baseline, and the fastest validated value
is published and persisted.

Backend-case parameter search
+++++++++++++++++++++++++++++

Kernel calibration answers where a kernel-specific synthetic workload crosses
a threshold. Backend-case tuning answers a different question: which parameter
set minimizes latency over a user-selected application corpus. The
``onnx-light backend`` command selects that corpus by regular expression and
evaluates the Cartesian product of the values supplied for one or more
parameters.

The search keeps measurement and analysis separate:

1. Python resolves one exact tuning schema and expands the parameter product.
2. Each parameter set is written to a temporary cache and measured in a fresh
   backend worker, so a previously initialized session cannot retain another
   set's values.
3. Case names define stable columns across every parameter set. The first set,
   composed of the active ``default`` values, is the speedup baseline.
4. :cpp:func:`AnalyzeKernelTuningLatencies` computes all aggregate metrics and
   selects the best complete set according to the requested criterion.
5. Temporary profiles are deleted after the comparison. The machine tuning
   cache is not changed.

The native analyzer reports latency sum, average, median, and maximum, plus
average, median, and maximum per-case speedup against the baseline. Latency
criteria are minimized and speedup criteria are maximized. A timeout makes that
parameter set incomplete; its metrics are reported as unavailable. If the
baseline is incomplete, complete candidates still receive latency metrics and
can be selected by a latency criterion, while speedup metrics remain
unavailable.

This boundary makes the C++ statistics and selection reusable independently of
the command line through
:func:`onnx_light.kernel_tuning.analyze_kernel_tuning_latencies`. Python owns
backend case discovery, process isolation, temporary-cache lifecycle, and
progress display.

The comparison result exists only in the command report: standard output by
default, JSON with ``--json``, or the file selected by ``--output``. The
selected set is advisory and is not published to the tuning registry, persisted
to the machine cache, or used by later kernel sessions. Persisting it requires
an explicit ``set_kernel_tuning_parameters`` call.

Persisted calibration and manual-update profiles follow a different lifecycle.
They are stored at :cpp:func:`DefaultKernelTuningCachePath` (or an explicit
path), published into the process tuning registry when loaded, and copied into
new kernel instances as their sessions initialize. Removing the file through
:cpp:func:`RemoveKernelTuningCache` prevents later processes from loading it,
but deliberately does not mutate existing sessions or profiles already
published in the current process.

See :ref:`l-how-to-tune-kernel-thresholds` for usage and
:ref:`l-example-plot-kernel-tuning` for an executable Python example. The
completed implementation history remains in
:ref:`l-next-steps-processor-aware-kernel-tuning`.
