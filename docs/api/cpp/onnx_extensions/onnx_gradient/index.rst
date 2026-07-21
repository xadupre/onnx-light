onnx_gradient
=============

Reverse-mode automatic differentiation for ONNX graphs.

Provides two entry points that compute gradient
:cpp:class:`FunctionProto` objects from an ONNX graph description:

* :cpp:func:`GradientOfNodes` — takes a sequence of
  :cpp:class:`NodeProto` objects together with graph metadata
  (inputs, initializers, xs, y, zs) and returns a
  :cpp:class:`FunctionProto` encoding the backward computation.
* :cpp:func:`GradientOfFunction` — takes an existing
  :cpp:class:`FunctionProto` together with xs, y, zs and returns the
  corresponding gradient :cpp:class:`FunctionProto`.

See :doc:`/api/python/onnx_extensions/onnx_gradient/index` for the Python interface and
:doc:`/design/gradient` for design notes.

.. doxygenfile:: gradient.h
   :project: onnx-light
