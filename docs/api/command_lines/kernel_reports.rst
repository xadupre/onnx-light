kernel_reports
==============

``python -m onnx_light.tools.kernel_reports`` collects native kernel measurements
and verifies persisted tuning profiles in separate processes. It requires a
build with kernels enabled. Run collection from the repository checkout so the
report can record the Git revision.

.. code-block:: bash

    python -m onnx_light.tools.kernel_reports collect --output /tmp/kernel-run-1 --threads 2
    python -m onnx_light.tools.kernel_reports verify --output /tmp/kernel-run-1
    python -m onnx_light.tools.kernel_reports verify-incompatible --output /tmp/kernel-run-1

Command-line reference
----------------------

.. code-block:: text

    python -m onnx_light.tools.kernel_reports [-h]
        --output OUTPUT [--threads THREADS] [--repeat REPEAT]
        [--maximum-duration-ms MAXIMUM_DURATION_MS]
        [--maximum-memory-bytes MAXIMUM_MEMORY_BYTES]
        {collect,verify,verify-incompatible}

``phase``
    ``collect`` creates a new output directory, measures a baseline, and
    calibrates kernels. ``verify`` checks saved profiles in a fresh process.
    ``verify-incompatible`` checks rejection of incompatible profiles.

``--output OUTPUT``
    Required report directory. It must not already exist for ``collect``;
    verification uses the directory created by collection.

``--threads THREADS``
    Positive thread count (default: ``2``).

``--repeat REPEAT``
    Positive measurement repetition count (default: ``10``).

``--maximum-duration-ms MAXIMUM_DURATION_MS``
    Positive calibration duration budget in milliseconds (default: ``10000``).

``--maximum-memory-bytes MAXIMUM_MEMORY_BYTES``
    Positive calibration memory budget in bytes (default: ``268435456``).

``-h``, ``--help``
    Shows help and exits.
