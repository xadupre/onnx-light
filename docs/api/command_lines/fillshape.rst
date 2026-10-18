fillshape
=========

Fills an ONNX model with shape-inference information and optional memory and
shape-tag metadata. Without ``--output`` or ``--show``, the input file is
overwritten.

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --output model_shapes.onnx
    python -m onnx_light fillshape model.onnx --show

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["fillshape"].format_help())
