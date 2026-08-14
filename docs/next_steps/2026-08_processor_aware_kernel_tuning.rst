.. _l-next-steps-processor-aware-kernel-tuning:

Processor-aware kernel tuning
=============================

:Date: 2026-08

**implementation complete**

The user workflow for calibrating one kernel, persisting or loading its cache,
and promoting a measured threshold to a portable compiled default is described
in :ref:`l-how-to-tune-kernel-thresholds` (`PR #4428
<https://github.com/xadupre/onnx-light/pull/4428>`_).

Objective
+++++++++

Kernel thresholds should adapt to the processor instead of being compiled as
universal constants. A threshold may select an algorithm, decide whether data
packing is worthwhile, or determine when parallel execution becomes faster
than serial execution. The correct value depends on the kernel, element type,
processor, available instruction sets, cache hierarchy, and thread count.

``onnx-light`` should provide the registration, processor matching, resolution,
calibration, and persistence mechanisms. It should not define the thresholds
of every kernel. A kernel library such as ``onnx-light-cpu`` owns:

* the names and meanings of its tuning parameters;
* safe portable defaults compiled into the kernel library;
* optional profiles for known processors or processor families;
* an optional calibration function;
* validation of the resulting values.

Every tunable value must therefore have a hard-coded value. The cache is an
optional optimization source, never the only source of a threshold. A missing,
unreadable, incompatible, or invalid cache leaves the compiled values in use,
so deleting the cache or starting on a new machine cannot prevent execution.

The resolved values are runtime configuration. They are not serialized in an
ONNX model and must not change the numerical contract of an operator.

Threshold categories
++++++++++++++++++++

The first design should support at least two categories.

Algorithm thresholds
^^^^^^^^^^^^^^^^^^^^

These values select between correct implementations of the same operation:

* scalar, SIMD, blocked, packed, or library-backed algorithms;
* direct convolution versus im2col;
* unrolled reduction versus a generic reduction;
* prepacking a constant matrix versus reading its original layout;
* one-dimensional versus two-dimensional task decomposition;
* tile dimensions and reduction chunk sizes.

An algorithm threshold may depend on several dimensions rather than one scalar.
For example, ``Gemm`` may need separate decisions for skinny ``M``, large ``K``,
and wide ``N``. A profile must therefore contain named values, not one global
``small_tensor_threshold``.

Parallel thresholds
^^^^^^^^^^^^^^^^^^^

These values decide when worker wake-up and synchronization costs are
amortized:

* minimum elements or bytes per parallel block;
* minimum estimated operations per task;
* maximum useful worker count;
* minimum number of independent tasks;
* conversion, packing, and post-processing grains.

Parallel thresholds depend on the effective thread count and affinity as well
as the processor. A profile calibrated for one thread must not be reused for
twenty threads.

Processor descriptor
++++++++++++++++++++

The core runtime detects one immutable descriptor per process:

.. code-block:: cpp

    struct CpuDescriptor {
      std::string architecture;       // x86_64, aarch64, ...
      std::string vendor;
      std::optional<uint32_t> family;
      std::optional<uint32_t> model;
      std::optional<uint32_t> stepping;
      std::string microarchitecture;  // when known
      CpuFeatureSet features;         // SSE2, AVX2, AVX-512, NEON, SVE, ...
      std::optional<size_t> cache_line_bytes;
      std::optional<size_t> l1_data_bytes;
      std::optional<size_t> l2_bytes;
      std::optional<size_t> l3_bytes;
      std::optional<uint32_t> physical_cores;
      std::optional<uint32_t> logical_cores;
    };

Missing information remains unknown; it must not be replaced by an invented
value. Detection is platform-specific but the descriptor and matching rules
belong to ``onnx_core``. ``GetCpuDescriptor()`` performs detection once and
returns the same immutable descriptor for the lifetime of the process.

Runtime properties that may change between sessions are kept separately:

.. code-block:: cpp

    struct CpuExecutionDescriptor {
      CpuDescriptor processor;
      uint32_t effective_threads;
      CpuSet affinity;
      NumaPolicy numa_policy;
    };

The execution descriptor is part of profile resolution and calibration-cache
keys. Power policy and frequency are intentionally excluded from the first
version because they are difficult to identify reliably; calibration reports
them when available for diagnostics.

Kernel tuning key
+++++++++++++++++

Thresholds are scoped to an exact kernel implementation and element type:

.. code-block:: cpp

    struct KernelTuningKey {
      std::string library;       // "onnx_light_cpu"
      std::string kernel;        // "Gemm"
      std::string implementation;// "blocked_avx2"
      int32_t element_type;      // TensorProto::FLOAT
      Device device;             // CPU
      uint32_t tuning_abi;       // invalidates incompatible cached profiles
    };

``FLOAT``, ``FLOAT16``, ``BFLOAT16``, and integer variants resolve
independently. Two types may share values only when the kernel explicitly
registers a common profile or copies validated values between its own schemas.
The registry must not silently fall back from one element type to another.

Named threshold set
+++++++++++++++++++

The core stores a portable set of named scalar values:

.. code-block:: cpp

    using TuningValue = std::variant<int64_t, double, bool, std::string>;

    struct KernelTuningParameters {
      KernelTuningKey key;
      std::unordered_map<std::string, TuningValue> values;
    };

Each kernel library converts this generic representation into a strongly typed
configuration once:

.. code-block:: cpp

    struct GemmFloatTuning {
      int64_t parallel_fmas_per_work_unit = 256;
      int64_t minimum_parallel_tasks = 2;
      int64_t pack_b_minimum_elements = 16384;
      int64_t skinny_m_limit = 8;
      int64_t tile_m = 64;
      int64_t tile_n = 256;
      int64_t tile_k = 256;
    };

    void ValidateAndApply(
        GemmFloatTuning& tuning,
        const KernelTuningParameters& parameters);

These initializers are the portable, hard-coded configuration and must remain
usable without registration, calibration, or file I/O. Loading a matching
profile validates all of its values and then rewrites the corresponding fields
in a new typed configuration. It does not make kernels look up values in the
cache during execution.

Unknown names, missing required values, invalid types, non-positive grains,
and inconsistent tile sizes are errors when a profile is registered or loaded.
They must not fail later in the execution hot path.

Profile registration
++++++++++++++++++++

A library may register one profile for an exact processor, a processor family,
an instruction-set class, or an explicit list of processors:

.. code-block:: cpp

    struct CpuSelector {
      std::optional<std::string> architecture;
      std::optional<std::string> vendor;
      std::optional<uint32_t> family;
      std::vector<uint32_t> models;
      std::optional<std::string> microarchitecture;
      CpuFeatureSet required_features;
      CpuFeatureSet excluded_features;
      std::optional<uint32_t> minimum_threads;
      std::optional<uint32_t> maximum_threads;
    };

    RegisterKernelTuningProfile(
        KernelTuningKey key,
        CpuSelector processors,
        KernelTuningParameters parameters,
        int priority);

``models`` provides the requested one-processor-or-list-of-processors
registration without duplicating profiles. An empty list matches every model
that satisfies the other fields.

Resolution uses deterministic precedence:

1. explicit session overrides;
2. a valid calibrated profile for the exact execution descriptor;
3. an exact vendor/family/model profile;
4. the most specific matching processor-list or microarchitecture profile;
5. an instruction-set profile;
6. the kernel's portable defaults.

Priority resolves only registrations with equal specificity. Ambiguous
profiles with the same specificity and priority are rejected. The registry
reports the selected profile and rejected candidates for diagnostics.

Kernel integration
++++++++++++++++++

The tuning registry belongs to ``onnx_core`` next to kernel dispatch, but the
resolved values are attached to the concrete kernel instance:

.. code-block:: cpp

    class KernelBase {
    public:
      virtual KernelTuningKey TuningKey(int32_t element_type) const;
      virtual void Configure(
          const KernelTuningParameters& parameters);
    };

During ``RuntimeSession`` kernel initialization:

1. kernel dispatch selects and constructs the implementation;
2. the runtime asks it for the tuning key of the resolved element type;
3. the registry resolves one profile for the session's execution descriptor;
4. the kernel copies its hard-coded typed configuration, then validates and
   applies the resolved values to that copy;
5. the typed configuration remains immutable for the kernel's lifetime.

No registry lookup, processor detection, string lookup, allocation, or mutex is
allowed in ``KernelBase::Run``. A session keeps its resolved profile even if
another thread registers or calibrates a newer profile. A new session observes
the newer registry generation.

Kernels whose element type is unknown until the first input arrives may cache
one immutable typed configuration per encountered type. The cache is populated
before that type's first execution and remains read-only afterward.

Calibration functions
+++++++++++++++++++++

A kernel library may register a trusted native calibration callback:

.. code-block:: cpp

    using KernelCalibrationFunction = KernelTuningParameters (*)(
        const KernelTuningKey& key,
        const CpuExecutionDescriptor& execution,
        const CalibrationOptions& options,
        CalibrationReporter& reporter);

    RegisterKernelCalibrationFunction(key, function);

``onnx-light`` exposes a batch entry point over every registered callback or a
filtered selection:

.. code-block:: cpp

    struct KernelCalibrationSelection {
      std::optional<std::string> library;
      std::vector<std::string> kernels;
      std::vector<std::string> implementations;
      std::vector<int32_t> element_types;
      std::optional<Device> device;
      bool only_missing;
    };

    CalibrationBatchReport CalibrateRegisteredKernels(
        const KernelCalibrationSelection& selection = {},
        const CalibrationOptions& options = {});

An empty selection runs every calibration callback currently registered for
the active device. Every non-empty field is a filter; fields combine with
logical AND and values inside one field combine with logical OR. For example,
``kernels = {"Abs", "Gemm"}`` and ``element_types = {FLOAT, DOUBLE}`` calibrates
the registered float and double variants of those two kernels.

``only_missing`` skips keys for which a compatible calibrated profile is
already loaded. The batch report distinguishes calibrated, skipped,
and unsupported keys. Callback and validation exceptions propagate to the
caller. Publication occurs only after every selected callback succeeds, so a
failure leaves the complete registry generation unchanged instead of returning
a partially successful report. The lower-level entry point for an exact list
of keys may remain available:

.. code-block:: cpp

    CalibrationBatchReport CalibrateKernelThresholds(
        std::span<const KernelTuningKey> keys,
        const CalibrationOptions& options);

Calibration is never triggered implicitly by loading an untrusted model. It is
run on user request, during package installation, or by a deployment image
builder. Only compiled callbacks registered by trusted kernel libraries are
executed.

A calibration function should:

1. generate deterministic synthetic inputs through shared runtime tensor
   generators accepting an element type, shape, seed, and optional allocator;
2. invoke the kernel implementation being calibrated rather than reimplementing
   its computation inside the calibration callback;
3. verify every candidate configuration against the same kernel forced onto
   its portable serial path;
4. warm every candidate and shared thread pool;
5. measure enough repetitions using robust statistics;
6. search crossover regions rather than every possible shape;
7. repeat measurements near a crossover and require a minimum winning margin;
8. return conservative values when measurements overlap;
9. respect time, memory, and thread budgets.

The report records all candidates, samples, medians, dispersion, rejected
measurements, selected values, processor information, thread count, runtime
version, and tuning ABI. A calibration failure is explicit and leaves the
portable defaults unchanged; it is not converted into a successful batch
report carrying a ``failed`` entry.

Example: ``Abs``
++++++++++++++++

``Abs`` needs few parameters but resolves them independently by type:

.. code-block:: text

    onnx_light_cpu / Abs / avx2 / FLOAT
      algorithm.simd_min_elements = 32
      parallel.minimum_elements = 1048576
      parallel.maximum_threads = 8

    onnx_light_cpu / Abs / avx2 / DOUBLE
      algorithm.simd_min_elements = 16
      parallel.minimum_elements = 524288
      parallel.maximum_threads = 8

Calibration compares serial scalar, serial SIMD, and parallel SIMD execution.
It finds the scalar/SIMD crossover and the serial/parallel crossover
separately. The current fixed ``32 * kParallelForGrainSize`` policy becomes a
portable default rather than a universal decision.

The portable ``onnx_light`` implementation registers a concrete calibration
callback for every element type supported by ``Abs``. It obtains deterministic
inputs from the shared ``RandnTensor(element_type, shape, seed, allocator)``
runtime helper. Two configured ``Abs`` instances then execute the actual
kernel: one is forced onto the serial path and the other uses the candidate
parallel grain. The callback checks their outputs, takes the median of five
measurements per candidate size, and requires two consecutive wins of at least
five percent before selecting a parallel grain. It does not duplicate the
element-wise absolute-value implementation.

The first callback keeps the crossover search, resource budgets, warm-up,
correctness comparison, and timing policy directly in ``CalibrateAbs``. This
is intentional: a common calibration API should not be extracted from unary
kernels alone. If no stable crossover is found within the requested time and
memory budgets, calibration retains ``32 * kParallelForGrainSize``. SIMD
crossover calibration remains the responsibility of an implementation such as
``onnx-light-cpu`` that owns the SIMD algorithm.

Example: ``Not``
++++++++++++++++

``RandnTensor`` supports ``BOOL`` inputs, and two configured ``Not`` instances
measure the real serial and parallel kernel paths. ``CalibrateNot`` initially
keeps its own search and measurement policy. The deliberate duplication makes
the requirements of more than one callback visible without prematurely
publishing an API specialized for unary kernels.

Example: ``Gemm``
+++++++++++++++++

``Gemm`` requires several independent decisions:

.. code-block:: text

    onnx_light_cpu / Gemm / blocked_avx2 / FLOAT
      algorithm.tile_m = 64
      algorithm.tile_n = 256
      algorithm.tile_k = 256
      algorithm.pack_b_minimum_elements = 16384
      algorithm.skinny_m_limit = 8
      parallel.fmas_per_work_unit = 256
      parallel.minimum_tasks = 2
      conversion.parallel_minimum_elements = 1048576

The calibrator must not search the full Cartesian product blindly. It first
selects safe tile candidates from cache sizes and SIMD width, then benchmarks
representative shape families:

* square matrices;
* skinny ``M`` and wide ``N``;
* large ``K`` split across chunks;
* transposed inputs;
* ``FLOAT``, ``FLOAT16``, ``BFLOAT16``, and ``DOUBLE`` separately.

Widening and narrowing thresholds are part of the half-precision profile, not
the float32 profile. Algorithm selection and parallel selection remain
separate even when both use the same shape variables.

Persistence
+++++++++++

Calibrated profiles may be stored outside the model in a versioned file:

.. code-block:: text

    schema_version
    onnx_light_version
    kernel_library
    kernel_library_version
    tuning_abi
    processor descriptor
    execution descriptor
    parameters
    calibration report digest

The default location is a user cache directory. Applications may provide a
read-only deployment profile or disable persistence.

Two explicit APIs separate cache mutation from cache loading:

.. code-block:: cpp

    KernelTuningCacheUpdateReport UpdateKernelTuningCache(
        std::span<const CalibratedKernelProfile> profiles,
        const KernelTuningCacheOptions& options = {});

    KernelTuningCacheLoadReport LoadKernelTuningCache(
        const KernelCalibrationSelection& selection = {},
        const KernelTuningCacheOptions& options = {});

``UpdateKernelTuningCache`` validates the profiles, acquires an inter-process
cache lock, reads the current file, merges entries by their complete tuning and
execution-descriptor key, and replaces the file atomically. It preserves
unselected valid entries. Options control the path, read-only mode, replacement
of older results, and pruning of stale tuning ABIs. A partial write must never
destroy the previous valid cache.

``LoadKernelTuningCache`` reads compatible entries if the file is available,
validates them, filters them with the same selection type used for calibration,
rewrites the in-memory values initially populated from hard-coded defaults, and
publishes one new immutable registry generation. Replacement is transactional:
all values of an entry are validated before any of them replace the defaults,
and an invalid entry leaves its complete hard-coded configuration unchanged. A
missing cache file is a normal ``not_found`` result, not an exception. An
unreadable file, malformed entry, duplicate key, or invalid parameter set is
reported separately and is never partially published. The load report lists
loaded, incompatible, stale, invalid, and missing registered keys.

The values rewritten by loading are runtime copies, not C++ ``constexpr``
objects or process-wide mutable variables used directly by kernels. Existing
sessions retain their immutable configuration; only sessions created from the
new registry generation observe the cached replacements.

The common workflow is therefore explicit:

.. code-block:: cpp

    KernelCalibrationSelection selection;
    selection.library = "onnx_light_cpu";

    auto loaded = LoadKernelTuningCache(selection);

    selection.only_missing = true;
    auto calibrated = CalibrateRegisteredKernels(selection, options);

    auto updated =
        UpdateKernelTuningCache(calibrated.successful_profiles());

Loading never launches calibration, and calibration never writes the cache.
This separation lets applications use a read-only deployment cache, inspect
results before accepting them, or calibrate without persistence.

The Python API mirrors the same three operations:

.. code-block:: python

    loaded = onnx_light.runtime.load_kernel_tuning_cache(
        library="onnx_light_cpu"
    )
    calibrated = onnx_light.runtime.calibrate_registered_kernels(
        library="onnx_light_cpu",
        kernels=["Abs", "Gemm"],
        only_missing=True,
    )
    updated = onnx_light.runtime.update_kernel_tuning_cache(
        calibrated.successful_profiles
    )

Loading validates every field and treats incompatible entries as cache misses.
A malformed profile is reported and rejected; it must not be partially
applied. In every cache-miss or cache-error case, the compiled values remain
available and are used unchanged.

Profiles should be exportable so a build farm can calibrate one representative
machine and deploy the result to an explicit processor list. An imported
profile never matches a processor outside its selector.

Concurrency and reproducibility
+++++++++++++++++++++++++++++++

Registration and profile loading occur before sessions are created. The global
registry publishes immutable generations so session creation can resolve
profiles concurrently. Calibration writes a new generation only after all
selected profiles validate.

For reproducible benchmarks, a session may request:

* portable defaults only;
* one named registered profile;
* one calibrated profile digest;
* explicit parameter overrides.

The selected source and complete resolved parameter set are exposed through
runtime diagnostics. This makes a performance result reproducible without
embedding machine-specific tuning in the ONNX model.

Cold resolution and hot-path verification
+++++++++++++++++++++++++++++++++++++++++

``RuntimeSession::tuning_resolution_statistics`` reports the one-time cost of
capturing an immutable registry generation and resolving profiles for tunable
kernels. ``KernelTuningRegistry::AccessCounts`` exposes monotonic snapshot and
snapshot-lookup counters, including profile resolutions, for diagnostics: the
first session run advances them, whereas every later run of that session must
leave every counter unchanged.

The standalone ``bench_kernel_tuning`` executable measures repeated cold
snapshot capture and profile resolution separately from steady
``RuntimeSession::Run`` calls. It fails instead of printing a successful result
if a steady-state run accesses the registry:

.. code-block:: bash

    cmake --build build --target bench_kernel_tuning -j
    ./build/bench_kernel_tuning -n 100000

Non-goals
+++++++++

The first version does not:

* tune continuously while inference requests are running;
* download profiles from a network service;
* let model contents execute calibration code;
* replace kernel dispatch or device placement;
* permit a tuning profile to relax correctness checks;
* guarantee that one profile is optimal under every system load or power mode.

Implementation order
++++++++++++++++++++

1. Add ``CpuDescriptor`` detection and stable processor matching in
   ``onnx_core`` (`PR #4367
   <https://github.com/xadupre/onnx-light/pull/4367>`_).
2. Add tuning keys, named parameter sets, validation hooks, and hard-coded
   portable defaults for every tunable value (`PR #4380
   <https://github.com/xadupre/onnx-light/pull/4380>`_).
3. Add ``LoadKernelTuningCache`` and immutable profile publication (`PR #4390
   <https://github.com/xadupre/onnx-light/pull/4390>`_).
4. Resolve an immutable profile while ``RuntimeSession`` initializes a kernel
   (`PR #4393 <https://github.com/xadupre/onnx-light/pull/4393>`_).
5. Migrate the portable ``Abs``, ``Exp``, and ``Not`` parallel grains to typed
   tuning parameters (`PR #4409
   <https://github.com/xadupre/onnx-light/pull/4409>`_). SIMD algorithm
   selection remains owned by ``onnx-light-cpu``.
6. Migrate ``Gemm`` tiling, packing, task, and conversion thresholds (`PR #4413
   <https://github.com/xadupre/onnx-light/pull/4413>`_).
7. Add exact, processor-list, and instruction-set profile registration and
   transactional publication of execution-specific calibrated profiles
   (`PR #4415 <https://github.com/xadupre/onnx-light/pull/4415>`_).
8. Add calibration callbacks and ``CalibrateRegisteredKernels`` selection,
   initially integrating ``Abs`` and ``Not`` through the shared random-tensor
   helper while keeping their calibration searches local (`PR #4418
   <https://github.com/xadupre/onnx-light/pull/4418>`_).
9. Add typed parallel tuning to all portable binary elementwise kernels,
   covering arithmetic, activation, logical, comparison, bitwise, equal-shape,
   and broadcasting execution paths (`PR #4421
   <https://github.com/xadupre/onnx-light/pull/4421>`_).
10. Add one calibration API for unary and binary kernels, with deterministic
    input generation for every operand, equal-shape and broadcasting cases,
    output validation, bounded resource accounting, and kernel-specific
    benchmark groups. Migrate the local ``Abs`` and ``Not`` searches and
    exercise the binary path through ``Add`` (`PR #4422
    <https://github.com/xadupre/onnx-light/pull/4422>`_).
11. Add inter-process-locked, atomic ``UpdateKernelTuningCache`` merging with
    read-only, replacement, and stale-ABI controls, plus transactional
    deployment-profile import constrained by explicit processor selectors
    (`PR #4424 <https://github.com/xadupre/onnx-light/pull/4424>`_).
12. Benchmark cold resolution separately from steady-state kernel execution and
    verify that the hot path performs no registry access (`PR #4426
    <https://github.com/xadupre/onnx-light/pull/4426>`_).
