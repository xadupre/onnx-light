.. _l-design-gradient:

Gradient
========

``onnx-light`` implements reverse-mode automatic differentiation as a graph
transformation. Gradient builders registered for individual operators produce
the nodes required to propagate output gradients back to graph inputs while
preserving the original forward graph.

The generic gradient interfaces live in ``lib_onnx_core`` and the ONNX
operator implementations in ``lib_onnx_gradient``. Applications select the
inputs and outputs to differentiate, then receive an ONNX graph that can be
optimized and executed through the same pipeline as any other model.

Gradient dispatch has no mutable process-wide registry used as hidden
transformation state. ``DefaultGradRegistry`` is immutable; callers that add a
custom backward rule register it in a caller-owned ``GradRegistry`` and pass
that registry explicitly to the transformation.

See the :ref:`gradient training example <l-example-gradient-linear-regression>`
for an end-to-end use case.

API reference
-------------

* **Python API**: :doc:`/api/python/onnx_core/gradient`.
* **C++ core API**: :doc:`/api/cpp/onnx_core/gradient/index`.
* **C++ ONNX implementations**:
  :doc:`/api/cpp/onnx_extensions/gradient/index`.
