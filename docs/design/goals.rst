Uncompromising Objective
========================

**No protobuf**

`onnx-light` replaces :epkg:`protobuf` by a custom implementation
but keeps the same ONNX format to make it fully compatible. It offers
more freedom to implement any custom loading, parsing scenario
and speed up this first step (see :ref:`l-how-to` section).

**Compile only what you need**

`onnx-light` is intentionally split into several small C++ libraries
(``lib_onnx_proto``, ``lib_onnx_op``, ``lib_onnx_lib``,
``lib_onnx_shape``, ``lib_onnx_kernels``, ``lib_onnx_backend_test``)
so that any downstream
project can link **only** the assembly it actually needs — from a bare
proto parser up to the full schema / shape-inference / runtime stack.
See :ref:`l-design-library-split` for the detailed breakdown.

**C++ Backend Test and Kernels**

Backend Tests are implemented in C++. They cannot contain any large tensor
and any output is generated through a C++ kernel implemented in C++.
The kernels themselves form a self-contained **reference implementation**
of the ONNX operator set, shipped as the ``lib_onnx_kernels`` static
library: it provides a runtime ``struct Tensor``, ``RunGraph`` /
``RunFunction`` / ``RunModel`` entry points and a per-operator kernel
under ``onnx_light/onnx_extensions/kernels/kernels/``.  The same kernels are used
both to compute the expected outputs of every backend test case and to
evaluate arbitrary models in C++ without pulling in a third-party
runtime (see :doc:`runtime/runtime_coverage`).
