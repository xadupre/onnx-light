.. _l-how-to-tune-kernel-thresholds:

Tune kernel thresholds
======================

``onnx-light`` separates two kinds of tuning values:

* **portable defaults** are compiled into the kernel library and always exist;
* **calibrated profiles** are measured for one processor and effective thread
  count, then optionally persisted in a cache.

A cache profile overrides the portable defaults only when its complete tuning
key, processor descriptor, and effective thread count match. It never changes
the ONNX model or the numerical contract of the operator.

Inspect parameters from Python
++++++++++++++++++++++++++++++

The tuning API is exposed by the full Python build:

.. code-block:: python

    from onnx_light import kernel_tuning
    from onnx_light.onnx import TensorProto

    report = kernel_tuning.kernel_tuning_parameters(
        kernel="Gemm",
        element_type=int(TensorProto.FLOAT),
    )
    for kernel in report["kernels"]:
        print(kernel["parameter_names"])
        print("portable:", kernel["defaults"])
        print("local cache:", kernel["cached_values"])
        print("active:", kernel["active_values"], kernel["active_source"])

Each result identifies the tuning ``library``, ``kernel``, ``implementation``,
ONNX ``element_type``, CPU ``device``, and ``tuning_abi``. ``cached_values`` is
``None`` when the cache has no profile matching both the local processor and
the requested effective thread count. ``active_source`` distinguishes
``portable_default``, ``published_profile`` (loaded, calibrated, or explicitly
overridden), and a statically registered processor ``registered_profile``.

Without a ``kernel`` filter, the function returns every registered exact key.
The optional ``library``, ``implementation``, ``element_type``, ``path``, and
``num_threads`` arguments narrow the query. The report also includes the cache
path, parse status, and diagnostics.

Change local values from Python
+++++++++++++++++++++++++++++++

``set_kernel_tuning_parameters`` accepts a partial update. Unspecified values
come from the matching cached profile when one exists, or from the portable
defaults otherwise. The complete result is validated before the cache is
modified:

.. code-block:: python

    update = kernel_tuning.set_kernel_tuning_parameters(
        "Gemm",
        int(TensorProto.FLOAT),
        {
            "parallel.minimum_tasks": 4,
        },
    )
    assert update["status"] == "updated", update["diagnostics"]

By default the update is persisted atomically in the default cache and loaded
into the current process. Pass ``path=...`` for another cache,
``num_threads=...`` for a profile scoped to that effective thread count, or
``load=False`` to persist without activating it. Unknown names, wrong Python
types, and values rejected by the kernel schema raise an exception without
changing the file.

``inspect_kernel_tuning_cache(path=None, num_threads=0)`` returns every
persisted profile and marks the profiles matching the local processor and
thread count with ``local=True``. It does not change active runtime values.
``default_kernel_tuning_cache_path()`` returns the default path.

See :ref:`l-example-plot-kernel-tuning` for an executable gallery example that
combines discovery, a temporary validated update, cache inspection, inference,
and bounded calibration.

Calibrate one kernel from the command line
+++++++++++++++++++++++++++++++++++++++++++

Select one exact native kernel and add ``--tune`` to calibrate and persist its
parameters:

.. code-block:: bash

    python -m onnx_light kernel \
        --kernel Abs --dtype FLOAT --impl portable \
        --tune

Use ``--cache`` to select another cache, ``--json`` for machine-readable output,
and ``--maximum-duration-ms`` or ``--maximum-memory-mb`` to bound calibration.
The Python functions ``propose_kernel_tuning_updates`` and
``apply_kernel_tuning_updates`` remain available for bulk workflows.

To compare explicit values for one integer parameter, list ``default`` first
so the current active value becomes the speedup baseline:

.. code-block:: bash

    python -m onnx_light kernel \
        --kernel Abs --dtype FLOAT --impl portable --tune \
        --parameter parallel.minimum_elements=default,16384,32768,65536

The existing calibration cases measure every value side by side. The fastest
validated value is persisted, while text and JSON output report elapsed times,
speedups, the baseline, and the selected value.

Optimize over backend cases
+++++++++++++++++++++++++++

Use ``onnx-light backend`` when the objective is latency over a list of
backend cases rather than a kernel's synthetic calibration workload. A regular
expression is required so the corpus is explicit:

.. code-block:: bash

    python -m onnx_light backend \
        --regex "^test_cc_not.*benchmark" --mode benchmark \
        --kernel Not --dtype BOOL --impl portable \
        --parameter parallel.minimum_elements=default,16384,32768,65536 \
        --criterion median-speedup --json

Repeat ``--parameter`` when a schema exposes interacting parameters. The command
evaluates their Cartesian product, with a limit of 256 sets. Each specification
starts with ``default``; the resulting all-default set is the baseline.
Parameter names must be unique, integer values must be positive, and
``--kernel``, ``--dtype``, and ``--impl`` must identify exactly one schema.

``--criterion`` is required and accepts:

* ``average``, ``sum``, ``median``, or ``max-latency``, which minimize the
  corresponding latency across selected cases;
* ``average-speedup``, ``median-speedup``, or ``max-speedup``, which maximize
  per-case speedup relative to the all-default baseline.

Every parameter set reports all seven metrics, its timeout count, and whether
it was selected. A timed-out set has unavailable metrics. When the baseline
times out, latency criteria can still select a complete candidate, but speedup
metrics and speedup-based selection are unavailable. Progress is written to
standard error, so ``--json`` on standard output remains machine-readable:

.. code-block:: text

    [backend tune] [##########----------] 2/4
    [backend tune] [####################] 4/4

The comparison uses temporary cache files and does not modify the machine
tuning cache. Its result is printed to standard output, returned as JSON with
``--json``, or written as CSV/XLSX with ``--output``. The selected set is
advisory: kernels do not use it after the command ends. Use
``set_kernel_tuning_parameters`` separately to persist and publish a selected
set.

Analyze measurements from Python
++++++++++++++++++++++++++++++++

The same native C++ metric analyzer is exposed in Python for measurements
collected by another harness. Rows represent parameter sets, columns represent
the same ordered cases, and the first row is the speedup baseline:

.. code-block:: python

    report = kernel_tuning.analyze_kernel_tuning_latencies(
        [
            [0.002, 0.008, 0.010],
            [0.001, 0.004, 0.020],
            [0.004, 0.004, 0.005],
        ],
        "average-speedup",
    )
    print(report["selected_index"])
    for metrics in report["values"]:
        print(metrics)

Use ``None`` for a missing case measurement. The corresponding row is
incomplete. The return value contains ``criterion``, ``selected_index``, and
``values``. Every complete value contains ``average``, ``sum``, ``median``,
``average_speedup``, ``median_speedup``, ``max_speedup``, and ``max_latency``.

Calibrate one kernel from Python
++++++++++++++++++++++++++++++++

The built-in calibration callbacks cover ``Abs``, ``Add``, ``Gemm``, ``Log``,
``Not``, ``Sigmoid``, and ``Tanh``. The Python extension registers them when
imported. Select a kernel and optionally one or more ONNX element types:

.. code-block:: python

    calibration = kernel_tuning.calibrate_kernel_tuning(
        "Abs",
        element_types=[int(TensorProto.FLOAT)],
        maximum_duration_ms=1000,
        maximum_memory_bytes=128 << 20,
    )
    print(calibration["calibrated"])
    print(calibration["diagnostics"])
    print(calibration["cache_update"])

Calibration generates deterministic inputs, checks every candidate output
against the forced serial implementation, warms the implementations, and uses
median timings. The shared unary/binary crossover search requires the
configured speedup for consecutive problem sizes. Resource limits bound the
search. Inspect ``diagnostics`` to see the selected value. Schema-only keys
without a callback appear in ``unsupported``.

The selected profile is published in the current process immediately.
With the default ``save=True``, it is also validated, locked, merged, and
atomically persisted for later processes. Use ``save=False`` for an in-memory
calibration or ``only_missing=True`` to skip an already active local profile.

Add calibration to another kernel
+++++++++++++++++++++++++++++++++

A registered tuning schema does not imply that a calibration callback exists.
``CalibrateRegisteredKernels`` reports schema-only keys in
``CalibrationBatchReport::unsupported``. To make another kernel calibratable:

1. Define a ``KernelCalibrationFunction`` near the kernel implementation.
2. Construct a ``KernelCalibrationBenchmark`` with its portable parameters,
   deterministic cases, reference runner, candidate runner, and output
   validation.
3. Call ``CalibrateKernelBenchmark`` from that function.
4. Register it for every supported exact key with
   ``RegisterKernelCalibrationFunction`` in the kernel's
   ``RegisterTuningSchemas`` function.

``onnx_light/onnx_extensions/kernels/kernels/math/kernel_abs.cc`` is the
unary example. ``kernel_add.cc`` demonstrates equal-shape and broadcasting
binary cases. A kernel with several interacting parameters, such as ``Gemm``,
needs a kernel-specific search rather than treating every value as an
independent scalar crossover.

Promote a threshold to a compiled default
+++++++++++++++++++++++++++++++++++++++++

A cache result is processor-specific. Measure several representative machines
and thread counts before making it the portable value used by every machine.
Keep the conservative value when crossover measurements overlap.

For kernels using ``ParallelTuning``:

* change the ``portable_minimum_elements`` passed to
  ``RegisterParallelTuningSchemas``;
* change the kernel object's initial fallback to the same value;
* change ``benchmark.portable_parameters`` in its calibration callback;
* add or update tests that exercise serial and parallel boundaries.

These values are in the corresponding implementation under
``onnx_light/onnx_extensions/kernels/kernels/``. For example, all three ``Abs``
fallback occurrences are in ``kernels/math/kernel_abs.cc``.

For ``Gemm``, the compiled values are the fields of ``GemmTuning`` in
``onnx_light/onnx_extensions/kernels/tuning/portable_gemm_tuning.h``.
``MakeGemmDefaults`` registers those fields as the schema defaults.

Increment ``tuning_abi`` when persisted profiles become structurally
incompatible, such as after renaming a parameter or changing its meaning or
type. A value-only default adjustment does not require an ABI change.

Locate and inspect the cache
++++++++++++++++++++++++++++

``default_kernel_tuning_cache_path()`` in Python and
``DefaultKernelTuningCachePath()`` in C++ return the exact default path:

* Windows: ``%LOCALAPPDATA%\onnx-light\kernel_tuning.cache``;
* other platforms with ``XDG_CACHE_HOME``:
  ``$XDG_CACHE_HOME/onnx-light/kernel_tuning.cache``;
* otherwise with ``HOME``:
  ``$HOME/.cache/onnx-light/kernel_tuning.cache``;
* without any supported cache-directory environment variable:
  ``onnx-light-kernel-tuning.cache`` in the current directory.

The cache is a versioned text file beginning with
``onnx_light_kernel_tuning_cache 1``. It can be inspected as text, but should
be modified through ``UpdateKernelTuningCache`` so validation, locking, merging,
and atomic replacement remain effective. Set ``KernelTuningCacheOptions::path``
to use an explicit location.

Remove cached results
+++++++++++++++++++++

Remove the default cache, or the explicit cache passed to tuning operations,
through the Python API:

.. code-block:: python

    removal = kernel_tuning.remove_kernel_tuning_cache()
    print(removal["path"], removal["removed"], removal["diagnostics"])

    # Removes an explicitly selected cache instead.
    removal = kernel_tuning.remove_kernel_tuning_cache("/path/to/kernel_tuning.cache")

The function takes the cache's inter-process lock before deleting the file.
``removed`` is ``False`` without diagnostics when the file was already absent.
After removal, a new process falls back to registered processor profiles or
portable defaults.

Removing a cache does not reconfigure kernels already initialized in the
current process. It also does not retract profiles already published into that
process's immutable tuning-registry generations. Restart the process after
removal when subsequent sessions must stop using a profile that was previously
loaded. The equivalent native operation is
:cpp:func:`RemoveKernelTuningCache`.

Load and use cached values
++++++++++++++++++++++++++

Importing ``onnx_light.onnx_py._onnxpykernels`` registers the built-in tuning
schemas and automatically loads compatible profiles from the default cache.
Therefore Python ``RuntimeSession`` and ``ReferenceEvaluator`` instances use
the local default cache without another call.

An explicit path is not loaded automatically. Load it before the first session
initializes its kernels:

.. code-block:: python

    load = kernel_tuning.load_kernel_tuning_cache(
        path="/path/to/kernel_tuning.cache",
    )
    assert load["status"] == "loaded", load["diagnostics"]

The C++ API remains explicit for every path:

.. code-block:: cpp

    onnx_light::onnx_kernels::RegisterKernelFunctions();

    rt::KernelCalibrationSelection selection;
    selection.library = "onnx_light";
    const rt::KernelTuningCacheLoadReport load =
        rt::LoadKernelTuningCache(selection);

    // Construct RuntimeSession only after loading the cache.
    rt::RuntimeSession session(plan);

Check ``load.status``, ``loaded``, ``incompatible``, ``stale``, ``invalid``,
``missing``, and ``diagnostics`` rather than assuming that a present file
matched. A missing, unreadable, malformed, stale, or processor-incompatible
entry leaves the compiled portable default active.

At initialization, ``RuntimeSession`` captures one immutable registry
generation, resolves a profile using its effective thread count, and copies the
typed values into each kernel. Later calls to ``Run`` do not access the
registry or reread the cache. Consequently, load a newer cache before creating
a new session; existing sessions deliberately retain their original values.
