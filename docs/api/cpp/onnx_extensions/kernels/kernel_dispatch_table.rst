kernel_dispatch_table.h
=======================

``onnx_extensions/kernels/kernel_dispatch_table.h`` declares
:cpp:func:`onnx_light::onnx_kernels::RegisterKernelFunctions`, which
populates ``onnx_core``'s dispatch table (see
:doc:`../../onnx_core/runtime/kernels/kernel_dispatch_table`) with every built-in
operator kernel and the ``SequenceMap`` output-packing callback.

.. doxygenfile:: onnx_extensions/kernels/kernel_dispatch_table.h
   :project: onnx-light
