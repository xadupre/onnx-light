onnx_light.onnx_core.graph_builder
==================================

Full builds load the compiled kernel registry when this module is imported,
so constant folding does not depend on importing ``ReferenceEvaluator`` first.
Reduced builds without the kernels extension still support graph authoring.
Required shape constant folding in such builds raises an error identifying the
missing operator kernel; importing the builder does not suppress extension
loading failures when the extension is present.

.. automodule:: onnx_light.onnx_core.graph_builder
    :members:
    :imported-members:
