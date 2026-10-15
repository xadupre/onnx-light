=============
PipelineProto
=============

.. autoclass:: onnx_light.onnx.PipelineProto
    :members:

``PipelineProto`` bundles the ONNX models used by a text-generation pipeline.
``prefill`` is required; ``embedding`` and ``decode`` are optional. Tensor
interfaces and token-generation settings are stored directly on the pipeline.
A value transferred between two models must have the same name on the source
output and target input; the pipeline does not contain a connection or
renaming graph.
