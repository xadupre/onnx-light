.. _l-how-to-tune-kernel-thresholds-command-line:

Tune kernel thresholds
======================

``onnx-light`` separates two kinds of tuning values:

* **portable defaults** are compiled into the kernel library and always exist;
* **calibrated profiles** are measured for one processor and effective thread
  count, then optionally persisted in a cache.

A cache profile overrides the portable defaults only when its complete tuning
key, processor descriptor, and effective thread count match. It never changes
the ONNX model or the numerical contract of the operator.

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

``parallel.minimum_elements`` is the serial/parallel crossover: a loop with
exactly that many elements is eligible for parallel execution. Once the loop is
dispatched, the executor derives the block grain from the number of admitted
participants, independently from the crossover value.

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
