.. _l-design:

Design
======

`onnx-light` replaces :epkg`protobuf` by a custom implementation
but keeps the same ONNX to make it fully compatible. It offers
more freedom to implement any custom loading, parsing scenario
and speed up this first step.

It replicates the same Python API and the same C++ API to enable
a smooth replacement.

.. toctree::
    :caption: Without Protobuf

    differences
    protobuf_format
    no_copy_ownership
    cplusplus_linking

Shape Inference is still under development as a new algorithm
is being implemented to handle small tensor and other backward propagation.
It supports simple expressions often used in models.

.. toctree::
    :caption: Shape Inference

    schema_comparison
    expressions

A C++ backend test is implemented enabling the testing of a runtime
in both Python and C++. It does not store any big tensor to avoid having
a huge package to publish.

.. toctree::
    :caption: Backend

    backend_tests
    test_coverage
    runtime_coverage
