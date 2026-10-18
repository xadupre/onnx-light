show
====

Renders an ONNX model as text, a diagram, or code that rebuilds the model.
The result is printed unless ``--output`` is specified.

.. code-block:: bash

    python -m onnx_light show model.onnx
    python -m onnx_light show model.onnx --format svg --output model.svg
    python -m onnx_light show model.onnx --format onnx-compact

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["show"].format_help())
