onnx_core
=========

The ``onnx_core`` library provides shared types and graph-manipulation
helpers used by both ``onnx_op`` (which builds operator schemas) and
``onnx_optim`` (which runs shape inference), without either of those
libraries depending on the other.

.. toctree::
    :maxdepth: 1

    expressions
    graph_manipulations
    light_op_schema
