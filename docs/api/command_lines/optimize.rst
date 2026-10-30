optimize
========

Optimizes a model and writes a new binary ONNX file beside the input, reusing
existing external weight files. The output path is required and must not exist.

.. code-block:: bash

    onnx-light optimize model.onnx --output model.opt.onnx
    python -m onnx_light optimize model.onnx -o model.opt.onnx --patterns L1,L2
    onnx-light optimize model.onnx -o model.opt.onnx --patterns NotNot,CastCast

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["optimize"].format_help())
