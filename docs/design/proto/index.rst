Proto
=====

The proto design covers the ONNX concepts and compatibility goals, binary
formats, model ownership, loading and validation.

Streams delegate worker management to the independent ``onnx_threads`` library;
see :ref:`l-design-threading`. Adaptive I/O policy and proto ownership remain
in this component.

.. toctree::
    :maxdepth: 1

    goals
    differences
    protobuf_format
    ort_flatbuffer_format
    no_copy_ownership
    loading_saving_scenarios
    fuzz
