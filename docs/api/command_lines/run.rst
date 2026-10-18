run
===

Generates random inputs and executes an ONNX model with the reference evaluator.
Use ``--dim`` to set symbolic input dimensions and ``--dump`` to save tensors.

.. code-block:: bash

    python -m onnx_light run model.onnx
    python -m onnx_light run model.onnx --dim batch=4 --seed 0 --dump tensors.onnx

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["run"].format_help())
