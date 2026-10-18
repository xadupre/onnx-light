onnx_light.onnx_core.graph_builder
==================================

Full builds load the compiled kernel registry when this module is imported,
so constant folding does not depend on importing ``ReferenceEvaluator`` first.
Reduced builds without the kernels extension still support graph authoring.
Required shape constant folding in such builds raises an error identifying the
missing operator kernel; importing the builder does not suppress extension
loading failures when the extension is present.

``GraphBuilder.to_onnx()`` returns a ``ModelProto`` by default. To export another
proto type, use the ``ProtoKind`` enum instead of a string::

    from onnx_light.onnx_core.graph_builder import GraphBuilder, ProtoKind

    builder = GraphBuilder("g")
    model = builder.to_onnx()
    graph = builder.to_onnx(ProtoKind.kGraph)
    function = builder.to_onnx(ProtoKind.kFunction, domain="custom")

``ProtoKind.kModel`` explicitly selects a model. The ``ir_version`` argument
applies to models, and ``domain`` applies to functions. String selectors such as
``"model"``, ``"graph"`` and ``"function"`` are no longer accepted.

.. automodule:: onnx_light.onnx_core.graph_builder
    :members:
    :imported-members:
