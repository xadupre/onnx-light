.. _l-howto-replace-onnx:
.. _l-howto-replace-onnx-python:

How to replace *onnx* by *onnx-light*
=====================================

This guide covers the Python interface. For the other language, see
:doc:`../cpp/replace_onnx`.

``onnx-light`` mirrors the public :epkg:`onnx` API in both Python and C++, so an
existing project can switch to it by changing a few imports, includes and build
targets. This page shows the Python code **before** and **after** the replacement.

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

See also
--------

* :ref:`l-design-cpp-linking` — the full list of exported CMake targets.
* :ref:`l-howto-install-onnx-light-python` — installation workflows for Python and C++.
* :ref:`l-onnx-tutorial` — ONNX introduction examples ported to ``onnx_light``.
