.. _l-howto-cpp:

How-tos C++
===========

These recipes use the native C++ APIs and CMake targets. Each mixed-language
recipe has a separate Python counterpart linked from its introduction.

.. toctree::
    :caption: Getting started
    :maxdepth: 1

    install_onnx_light
    ../link_prebuilt_cpp
    replace_onnx
    ../onnxruntime_migration

.. toctree::
    :caption: Model I/O
    :maxdepth: 1

    load_save_onnx_files
    align_external_data_streaming
    save_model_with_shared_external_data
    save_ort_flatbuffers

.. toctree::
    :caption: Testing and tuning
    :maxdepth: 1

    collect_backend_test_cases
    tune_kernel_thresholds

Benchmarking guidance shared by all interfaces is documented in
:doc:`/howto/benchmark_methodology`.

.. toctree::
    :caption: Custom runtime and operators
    :maxdepth: 1

    use_custom_kernel
    persistent_feedback
    quantized_values
    use_custom_shape_inference
    register_builtin_operators

.. toctree::
    :caption: Graph rewrites and utilities
    :maxdepth: 1

    add_custom_pattern
    onnx_graph_manipulations
