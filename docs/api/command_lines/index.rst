Command lines
=============

The ``onnx-light`` executable and ``python -m onnx_light`` provide the same
commands. Each page below describes one command, with its arguments and examples.

.. list-table:: Command-line summary
    :header-rows: 1
    :widths: 45 55

    * - Command line
      - Summary
    * - :doc:`python -m onnx_light <onnx_light>`
      - Lists available commands and displays command-specific help.
    * - :doc:`python -m onnx_light fillshape <fillshape>`
      - Adds inferred shapes and optional memory and shape-tag metadata to a model.
    * - :doc:`python -m onnx_light show <show>`
      - Renders a model as text, a diagram, or code that rebuilds it.
    * - :doc:`python -m onnx_light run <run>`
      - Generates random inputs and executes a model with the reference evaluator.
    * - :doc:`python -m onnx_light backend <backend>`
      - Measures selected backend test cases and optionally writes a report.
    * - :doc:`python -m onnx_light kernel <kernel>`
      - Lists registered kernels, inspects tunable parameters, or calibrates a kernel.
    * - :doc:`python -m onnx_light.tools.kernel_reports <kernel_reports>`
      - Collects kernel measurements and verifies persisted tuning profiles.

.. toctree::
    :maxdepth: 1

    onnx_light
    fillshape
    show
    run
    backend
    kernel
    kernel_reports
