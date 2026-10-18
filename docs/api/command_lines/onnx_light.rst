onnx-light
==========

Lists the available model, backend-test, and kernel utilities.
Use ``onnx-light COMMAND --help`` or ``python -m onnx_light COMMAND --help``
for command-specific options.

.. code-block:: bash

    python -m onnx_light --help

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    print(_build_parser().format_help())
