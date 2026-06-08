.. _l-onnx-tutorial:

==================================
Introduction to ONNX with onnx-light
==================================

This documentation describes the ONNX concepts (**Open Neural Network Exchange**).
It shows how it is used with examples in Python and finally explains
some of the challenges faced when moving to ONNX in production.

The pages below are an adaptation of the upstream
`ONNX introduction <https://github.com/onnx/onnx/tree/main/docs/docsgen/source/intro>`_
where every Python example uses the ``onnx_light`` API instead of ``onnx``.
``onnx-light`` exposes the same protobuf message types
(:class:`~onnx_light.onnx_lib.ModelProto`, :class:`~onnx_light.onnx_lib.TensorProto`,
...) and helper modules (``helper``, ``checker``, ``numpy_helper``,
``shape_inference``, ``parser``) under :mod:`onnx_light.onnx_lib`, so most code
written against the standard ``onnx`` package can be ported by changing a few
``import`` statements.

.. toctree::
    :maxdepth: 2

    concepts
    python
