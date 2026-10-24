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

The default binary file extensions are ``.pipe-onnx`` and ``.pipeline``.
``onnx_light.onnx.save(pipeline, path)`` saves the container, and
``onnx_light.onnx.load(path)`` selects ``PipelineProto`` for either extension.
Other extensions and bytes inputs continue to select ``ModelProto``.

See :ref:`l-example-pipeline-proto` for a self-contained example covering
construction, validation, binary serialization, and deterministic generation
from the reloaded models, including the current Python execution limitations.
