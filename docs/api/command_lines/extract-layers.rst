extract-layers
==============

Extracts repeated computations into local Layer functions without loading large
raw weight payloads. Saves a rewritten ONNX model and prints a layer summary.

.. code-block:: bash

    python -m onnx_light extract-layers model.onnx --output model.layers.onnx

See :ref:`l-cli-extract-layers` for loading restrictions, output placement and
the layer summary.

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["extract-layers"].format_help())
