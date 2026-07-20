test_case.h (compatibility shim)
=================================

This header re-exports all names from
:doc:`../../onnx_core/backend_test/test_case` into the ``onnx_backend_test``
namespace for backward compatibility.  New code should include
``onnx_core/backend_test/test_case.h`` and use the ``core::backend_test``
namespace directly.

.. doxygenfile:: onnx_backend_test/test_case.h
   :project: onnx-light
