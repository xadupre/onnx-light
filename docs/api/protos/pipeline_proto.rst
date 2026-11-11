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

Native generation
=================

The C++ ``Generate`` overload runs ``prefill`` for the prompt, then ``decode``
for each subsequent token when supplied. Without ``decode``, the prefill model
continues to run as before. A separate decode graph receives one token per row,
the complete growing attention mask, and the current position IDs. An optional
embedding graph runs before either language-model stage. Sampling, EOS, padding
and step observation use the same generation loop.

Both language-model interfaces are checked before execution, even when zero
tokens are requested. They must agree on generated input names, element types
and ranks, logits metadata, and persistent input/output binding names and types.
Fixed dimensions must agree; missing shapes and symbolic or unspecified
dimensions are wildcards, not equations between separately exported models.
Sequence dimensions may be
specialized for the prompt versus single-token decode; cache feature dimensions
and vocabulary dimensions must remain compatible. Persistent state is carried
by transferring retained buffers and their append capacity, without copying cache
payloads to change models. Both persistent binding sets may also be empty; an
explicit decode graph still receives only one token per row.
Each generation call starts with fresh state; generation is not resumable across
calls.

``LoadedRuntimePipeline`` loads and prepares every stage once, then shares its
static preparation store with fresh generation sessions across calls. Initializer
payloads are currently model-local: equal inline weights or equal external-data
ranges in different models are loaded separately, and static preparations are
keyed by model identity rather than deduplicated across stages. Budget for both
sets of weights and preparations. Within each stage, directly readable weights
remain borrowed views and the owning loader reuses preparations across calls.

The default binary file extensions are ``.pipe-onnx`` and ``.pipeline``.
``onnx_light.onnx.save(pipeline, path)`` saves the container, and
``onnx_light.onnx.load(path)`` selects ``PipelineProto`` for either extension.
Other extensions and bytes inputs continue to select ``ModelProto``.

See :ref:`l-example-pipeline-proto` for a self-contained example covering
construction, validation, binary serialization, and deterministic generation
from the reloaded models, including the current Python execution limitations.
