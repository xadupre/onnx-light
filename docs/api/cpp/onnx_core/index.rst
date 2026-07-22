onnx_core
=========

The ``onnx_core`` library provides shared types and graph-manipulation
helpers used by both ``onnx_op`` (which builds operator schemas) and
``onnx_shapes`` (which runs shape inference), without either of those
libraries depending on the other.

.. toctree::
    :maxdepth: 1

    expressions
    graph_manipulations
    light_op_schema
    sym_tensor
    sym_sequence
    sym_map
    shapes/index
    runtime/index
    compute/index
    backend_test/index
