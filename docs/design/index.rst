.. _l-design:

Design
======

`onnx-light` replaces :epkg`protobuf` by a custom implementation
but keeps the same ONNX to make it fully compatible. It offers
more freedom to implement any custom loading, parsing scenario
and speed up this first step.

It replicates the same Python API and the same C++ API to enable
a smooth replacement.

`onnx-light` is intentionally split into several small C++ libraries
(``lib_onnx_proto``, ``lib_onnx_op``, ``lib_onnx_lib``,
``lib_onnx_optim``, ``lib_onnx_backend_test``) so that any downstream
project can link **only** the assembly it actually needs — from a bare
proto parser up to the full schema / shape-inference / runtime stack.
See :ref:`l-design-library-split` for the detailed breakdown.

.. toctree::
    :caption: Without Protobuf
    :maxdepth: 1

    differences
    protobuf_format
    no_copy_ownership
    library_split
    cplusplus_linking

Shape Inference is still under development as a new algorithm
is being implemented to handle small tensor and other backward propagation.
It supports simple expressions often used in models.

.. toctree::
    :caption: Shape Inference
    :maxdepth: 1

    expressions

A C++ backend test is implemented enabling the testing of a runtime
in both Python and C++. It does not store any big tensor to avoid having
a huge package to publish.

.. toctree::
    :caption: Backend
    :maxdepth: 1

    backend_tests
    test_coverage
    runtime_coverage
