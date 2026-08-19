.. _l-howto-replace-onnx:

How to replace *onnx* by *onnx-light*
=====================================

``onnx-light`` mirrors the public :epkg:`onnx` API in both Python and C++, so an
existing project can switch to it by changing a few imports, includes and build
targets.  This page shows the code **before** and **after** the replacement for
Python, C++ and the matching build configuration.

Python
------

The :mod:`onnx_light.onnx` module exposes the same protobuf message types
(:class:`~onnx_light.onnx_lib.ModelProto`,
:class:`~onnx_light.onnx_lib.TensorProto`, ...) and submodules
(``helper``, ``numpy_helper``, ``reference``, ``backend``).  Only the imports
change; the rest of the code (``load``, ``save``, ``oh.make_node``,
``ReferenceEvaluator``, ...) stays the same.

.. tab-set::

   .. tab-item:: Before (onnx)

      .. code-block:: python

          import onnx
          import onnx.helper as oh
          import onnx.numpy_helper as onh
          from onnx.reference import ReferenceEvaluator

          model = onnx.load("model.onnx")
          node = oh.make_node("Add", ["a", "b"], ["c"])
          ref = ReferenceEvaluator(model)

   .. tab-item:: After (onnx-light)

      .. code-block:: python

          import onnx_light.onnx as onnx
          import onnx_light.onnx.helper as oh
          import onnx_light.onnx.numpy_helper as onh
          from onnx_light.onnx.reference import ReferenceEvaluator

          model = onnx.load("model.onnx")
          node = oh.make_node("Add", ["a", "b"], ["c"])
          ref = ReferenceEvaluator(model)

C++
---

``onnx-light`` replicates the upstream ``onnx`` C++ API under the ``onnx_light``
namespace.  The ``ONNX_NAMESPACE`` macro resolves to ``onnx_light`` and the
headers keep their familiar names, so the source change is the include root
(``onnx/`` becomes ``onnx_lib/``) and, where the namespace is spelled out
explicitly, ``onnx::`` becomes ``onnx_light::``.  Code that already uses the
``ONNX_NAMESPACE`` macro instead of a hard-coded ``onnx::`` needs only the
include-path change.

.. tab-set::

   .. tab-item:: Before (onnx)

      .. code-block:: cpp

          #include "onnx/onnx_pb.h"
          #include "onnx/checker.h"

          int main() {
            onnx::ModelProto model;
            onnx::checker::check_model(model);
            return 0;
          }

   .. tab-item:: After (onnx-light)

      .. code-block:: cpp

          #include "onnx_lib/onnx_pb.h"
          #include "onnx_lib/checker.h"

          int main() {
            onnx_light::ModelProto model;
            onnx_light::checker::check_model(model);
            return 0;
          }

Build (CMake)
-------------

Downstream CMake projects replace the ``onnx`` package and target with the
``onnx_light`` package and the equivalent ``onnx_light::...`` target.  Use
``onnx_light::lib_onnx_lib`` for the schema / checker / shape-inference layer or a
lighter target such as ``onnx_light::lib_onnx_proto`` for proto-only parsing —
see :ref:`l-design-cpp-linking` for the full list.

.. tab-set::

   .. tab-item:: Before (onnx)

      .. code-block:: cmake

          find_package(ONNX REQUIRED)
          add_executable(my_target main.cc)
          target_link_libraries(my_target PRIVATE onnx)

   .. tab-item:: After (onnx-light)

      .. code-block:: cmake

          find_package(onnx_light REQUIRED)
          add_executable(my_target main.cc)
          target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_lib)

If *onnx-light* is installed to a non-standard prefix, configure the downstream
project with ``-DCMAKE_PREFIX_PATH=<prefix>`` before calling
``find_package(onnx_light REQUIRED)``.

See also
--------

* :ref:`l-design-cpp-linking` — the full list of exported CMake targets.
* :ref:`l-howto-install-onnx-light` — installation workflows for Python and C++.
* :ref:`l-onnx-tutorial` — ONNX introduction examples ported to ``onnx_light``.
