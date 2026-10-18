.. _l-howto-python:

How-tos Python
==============

These recipes use the Python APIs. Each mixed-language recipe has a separate
C++ counterpart linked from its introduction.

.. toctree::
    :caption: Getting started
    :maxdepth: 1

    install_onnx_light
    replace_onnx

.. toctree::
    :caption: Model building and I/O
    :maxdepth: 1

    ../graph_builder_basics
    load_save_onnx_files
    align_external_data_streaming
    save_model_with_shared_external_data
    save_ort_flatbuffers

.. toctree::
    :caption: Testing and benchmarking
    :maxdepth: 1

    collect_backend_test_cases
    ../run_backend_test_case
    ../benchmark_methodology

.. toctree::
    :caption: Custom runtime and operators
    :maxdepth: 1

    use_custom_kernel
    persistent_feedback
    quantized_values
    tune_kernel_thresholds
    use_custom_shape_inference
    register_builtin_operators

.. toctree::
    :caption: Graph rewrites and utilities
    :maxdepth: 1

    add_custom_pattern
    onnx_graph_manipulations
