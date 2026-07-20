.. _l-design-contexts:

Contexts
========

``onnx-light`` uses several context classes to carry state while executing
kernels, inferring shapes, and computing graph-level annotations.

.. list-table::
    :header-rows: 1

    * - Context
      - Main role
      - C++ API
      - Python API
    * - ``KernelContext``
      - Stores operator schema metadata and lookup tables used by runtime
        dispatch.
      - :cpp:class:`onnx::onnx_kernels::KernelContext`
      - :class:`onnx_light.onnx.reference.KernelContext`
    * - ``RuntimeContext``
      - Holds runtime values and executes custom kernels.
      - :cpp:class:`onnx::onnx_kernels::RuntimeContext`
      - :class:`onnx_light.onnx.reference.RuntimeContext`
    * - ``ShapesContext`` (shape context)
      - Tracks symbolic tensor shapes and types during shape inference.
      - :cpp:class:`onnx::core::shapes::ShapesContext`
      - :class:`onnx_light.onnx_shapes.shape_inference.ShapesContext`
    * - ``ComputeContext``
      - Computes graph-level annotation results (value tags, in-place reuse,
        memory profile).
      - :cpp:class:`onnx::onnx_shapes::annotations::ComputeContext`
      - :class:`onnx_light.onnx_shapes.shape_inference.ComputeContext`

See also
--------

* :doc:`../api/cpp/onnx_kernels/kernels/kernel_context`
* :doc:`../api/cpp/onnx_kernels/runtime_context`
* :doc:`../api/cpp/onnx_core/shapes/shapes_context`
* :doc:`../api/cpp/onnx_core/annotations/inplace_reuse`
* :doc:`../api/python/onnx/reference`
* :doc:`../api/python/onnx_shapes/shape_inference`
* :doc:`../api/python/onnx_shapes/compute_context`
