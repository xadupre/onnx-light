backend
========

Measures backend test cases selected by a regular expression and optionally
writes a report. Requires a build with backend tests and kernels enabled.

.. code-block:: bash

    python -m onnx_light backend --help

See also :doc:`../../howto/run_backend_test_case`.

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["backend"].format_help())
