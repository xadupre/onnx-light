Python API
==========

The Python API exposes :mod:`onnx_light.onnx` (a drop-in replacement for
the upstream ``onnx`` package) together with a handful of auxiliary
packages: :mod:`onnx_light.backend` (ONNX backend test infrastructure)
and :mod:`onnx_light.onnx_expressions` (symbolic dimension expression
utilities).

onnx\_light.onnx
++++++++++++++++

Sub-modules of :mod:`onnx_light.onnx`:

.. toctree::
    :maxdepth: 1

    checker
    compose
    defs
    schema_diff
    helper
    inliner
    io_helper
    numpy_helper
    parser
    shape_inference
    utils
    version_converter

Top-level members of :mod:`onnx_light.onnx`:

.. automodule:: onnx_light.onnx
    :members:

Auxiliary packages
++++++++++++++++++

.. toctree::
    :maxdepth: 1

    backend/index
    expressions
