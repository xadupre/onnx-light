layers
======

Extracts repeated computations into local Layer functions without loading large
raw weight payloads. Saves a rewritten ONNX model and prints a layer summary.

.. code-block:: bash

    python -m onnx_light layers model.onnx --output model.layers.onnx

See :ref:`l-cli-layers` for structural export details, output placement and
the layer summary.

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["layers"].format_help())
