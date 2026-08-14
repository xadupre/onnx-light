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

Calibrate one kernel
++++++++++++++++++++

The built-in calibration callbacks currently cover ``Abs``, ``Add``, and
``Not``. Register the built-in kernels first, select a kernel, run its callback,
and persist the successful profiles:

.. code-block:: cpp

    #include "onnx_core/runtime/kernel_tuning.h"
    #include "onnx_core/runtime/kernel_tuning_cache.h"
    #include "onnx_extensions/kernels/kernel_dispatch_table.h"

    #include <iostream>

    namespace rt = onnx_light::core::runtime;

    int main() {
      onnx_light::onnx_kernels::RegisterKernelFunctions();

      rt::KernelCalibrationSelection selection;
      selection.library = "onnx_light";
      selection.kernels = {"Abs"};
      selection.element_types = {
          static_cast<int32_t>(rt::DataType::FLOAT)};

      rt::CalibrationOptions options;
      options.maximum_duration_ms = 1000;
      options.maximum_memory_bytes = uint64_t{128} << 20;

      const rt::CalibrationBatchReport calibration =
          rt::CalibrateRegisteredKernels(selection, options);
      if (calibration.calibrated.empty()) {
        std::cerr << "No selected kernel has a calibration callback.\n";
        return 1;
      }

      const rt::KernelTuningCacheUpdateReport update =
          rt::UpdateKernelTuningCache(
              calibration.successful_profiles());
      if (update.status != rt::KernelTuningCacheUpdateStatus::kUpdated) {
        for (const std::string &message : update.diagnostics) {
          std::cerr << message << '\n';
        }
        return 1;
      }

      std::cout << rt::DefaultKernelTuningCachePath() << '\n';
    }

Calibration generates deterministic inputs, checks every candidate output
against the forced serial implementation, warms the implementations, and uses
median timings. The shared unary/binary crossover search requires the
configured speedup for consecutive problem sizes. Resource limits bound the
search. Inspect ``CalibrationBatchReport::diagnostics`` and ``resources`` to
see the selected value and measurement cost.

The selected profile is published in the current process immediately.
``UpdateKernelTuningCache`` is needed only to make it available to later
processes. It validates the complete profile, locks the cache across processes,
merges it, and atomically replaces the file.

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

``DefaultKernelTuningCachePath()`` returns the exact active default path:

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

Load and use cached values
++++++++++++++++++++++++++

The cache is **not loaded merely because the file exists**. Each process must
load it explicitly after tuning schemas are registered and before its first
``RuntimeSession`` initializes kernels:

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
