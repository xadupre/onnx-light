kernel
======

Lists registered kernels, inspects tunable parameters, or calibrates a selected
kernel. Requires a build with kernels enabled.

.. code-block:: bash

    python -m onnx_light kernel --list
    python -m onnx_light kernel --help

See also :doc:`../../howto/tune_kernel_thresholds`.

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["kernel"].format_help())
