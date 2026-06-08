.. _l-design:

Design
======

Uncompromising Objective
++++++++++++++++++++++++

**No protobuf**

`onnx-light` replaces :epkg:`protobuf` by a custom implementation
but keeps the same ONNX format to make it fully compatible. It offers
more freedom to implement any custom loading, parsing scenario
and speed up this first step (see :ref:`l-how-to` section).

**Compile only what you need**

`onnx-light` is intentionally split into several small C++ libraries
(``lib_onnx_proto``, ``lib_onnx_op``, ``lib_onnx_lib``,
``lib_onnx_optim``, ``lib_onnx_kernels``, ``lib_onnx_backend_test``)
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
under ``onnx_light/onnx_kernels/kernels/``.  The same kernels are used
both to compute the expected outputs of every backend test case and to
evaluate arbitrary models in C++ without pulling in a third-party
runtime (see :doc:`backend_tests` and :doc:`runtime_coverage`).


In Details
++++++++++

It replicates the same Python API and the same C++ API to enable
a smooth replacement.

.. toctree::
    :caption: No protobuf
    :maxdepth: 1

    differences
    protobuf_format
    no_copy_ownership
    loading_saving_scenarios

.. toctree::
    :caption: Library Split
    :maxdepth: 1

    library_split
    cplusplus_linking

.. toctree::
    :caption: Backend
    :maxdepth: 1

    backend_tests
    test_coverage
    runtime_coverage

Shape Inference
++++++++++++++++

.. toctree::
    :caption: Shape Inference
    :maxdepth: 1

    expressions
    inference_coverage

A set of `atheris`-based Python fuzz targets exercises the public API
surface (loader, checker, parser, shape inference, version converter)
from random inputs. They are driven by OSS-Fuzz for long-running
campaigns and by a short CI smoke workflow.

.. toctree::
    :caption: Fuzzing
    :maxdepth: 1

    fuzz
