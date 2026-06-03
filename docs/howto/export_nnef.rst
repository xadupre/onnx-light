Export a model to NNEF format
=============================

``onnx_light`` ships with a small, dependency-free exporter that
translates an ONNX :class:`ModelProto` into the
`Khronos NNEF v1.0 <https://www.khronos.org/nnef>`_ representation.

The exporter produces a directory containing:

* ``graph.nnef`` – a textual description of the graph;
* one ``<label>.dat`` file per initializer, encoded with the standard
  NNEF binary tensor format (128-byte header followed by raw data).

Quick start
-----------

.. code-block:: python

    import onnx_light.onnx as onnxl
    from onnx_light.nnef import save_nnef

    model = onnxl.load("model.onnx")
    save_nnef(model, "model_nnef")

The exporter also works on a :mod:`onnx` ``ModelProto`` because it only
relies on the public attributes shared by both implementations:

.. code-block:: python

    import onnx
    from onnx_light.nnef import save_nnef, to_nnef_text

    model = onnx.load("model.onnx")
    print(to_nnef_text(model))   # textual graph only
    save_nnef(model, "model_nnef")

Supported operators
-------------------

The list of ONNX op types that have a builtin NNEF converter is
returned by :func:`onnx_light.nnef.supported_ops`.  It currently covers
the most common operators of vision-style networks: ``Conv``,
``BatchNormalization``, ``MaxPool``, ``AveragePool``,
``GlobalAveragePool``, ``GlobalMaxPool``, ``Relu``, ``Sigmoid``,
``Tanh``, ``Softmax``, ``Gemm``, ``MatMul``, ``Reshape``, ``Flatten``,
``Transpose``, ``Concat``, ``Add``/``Sub``/``Mul``/``Div``/``Pow``,
``Min``/``Max``, ``Identity`` and ``Clip``.

Extending the converter
-----------------------

If you need to export an operator that is not built in, register a
custom converter:

.. code-block:: python

    from onnx_light.nnef import register_op_converter

    def convert_my_op(ctx, node, attrs, inputs, outputs):
        # ``inputs`` and ``outputs`` are NNEF identifiers already mapped
        # from the ONNX names; ``attrs`` is a {name: value} dictionary.
        ctx.add_statement(f"{outputs[0]} = my_op({inputs[0]});")

    register_op_converter("MyOp", convert_my_op)

If the exporter encounters an op without a registered converter it
raises :class:`onnx_light.nnef.NNEFExportError` with a descriptive
message.
