"""
.. _l-example-pipeline-proto:

Builds, saves, and runs a small generation pipeline
===================================================

``PipelineProto`` bundles models, tensor-interface names, and generation settings.
This example creates all weights locally, validates the container, saves and
reloads its binary representation, and generates tokens from the reloaded models.
No tokenizer, external model, or download is needed: the vocabulary is simply
the integers 0, 1, and 2.

The Python runtime currently accepts individual models, not ``PipelineProto``
itself. A binding for the C++ ``Generate(PipelineProto, ...)`` overload is still
needed for direct container execution. Below, we explicitly pass the restored
models and supported settings to ``ReferenceEvaluator.generate`` instead.
Serialization alone does not execute a pipeline.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from onnx_light import onnx
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_proto import verify

# %%
# Create two tiny stages
# ----------------------
#
# ``prefill`` is required. ``embedding`` optionally maps token IDs to features;
# ``decode`` optionally stores a separate model for subsequent tokens and state.
# The current generation runtime does not support a separate decode model, so
# this example omits it and uses prefill for both the prompt and later steps.
# Without persistent cache bindings, each step evaluates the full growing prefix.
#
# The embedding table makes greedy decoding follow 0 -> 1 -> 2 -> 0.
# Its output and the prefill input are both named ``inputs_embeds``: stages
# exchange values by matching names, not by a connection or renaming graph.
# Private initializer/intermediate names must be unique across stages.

embedding = oh.make_model(
    oh.make_graph(
        [oh.make_node("Gather", ["transition_table", "input_ids"], ["inputs_embeds"])],
        "embedding",
        [oh.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, ["B", "S"])],
        [oh.make_tensor_value_info("inputs_embeds", onnx.TensorProto.FLOAT, ["B", "S", 3])],
        initializer=[
            onh.from_array(
                numpy.array([[0, 4, 0], [0, 0, 4], [4, 0, 0]], dtype=numpy.float32),
                name="transition_table",
            )
        ],
    ),
    opset_imports=[oh.make_opsetid("", 23)],
    ir_version=10,
)
prefill = oh.make_model(
    oh.make_graph(
        [oh.make_node("Identity", ["inputs_embeds"], ["logits"])],
        "prefill",
        [oh.make_tensor_value_info("inputs_embeds", onnx.TensorProto.FLOAT, ["B", "S", 3])],
        [oh.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, ["B", "S", 3])],
    ),
    opset_imports=[oh.make_opsetid("", 23)],
    ir_version=10,
)

# %%
# Set generation parameters and interface names
# ---------------------------------------------
#
# Zero temperature selects deterministic greedy decoding. The seed would control
# sampling at positive temperatures. EOS ends generation early; padding applies
# to already-finished rows in a batch. Top-k and top-p filtering are not currently
# implemented by the runtime: top_k may be zero, but top_p must remain unset
# (even top_p=1.0 is rejected by the C++ container execution API).
#
# ReferenceEvaluator.generate uses the standard interface names shown here.
# The optional mask and position names do not add inputs to these tiny graphs.
# verify_pipeline checks model structure and parameter ranges, not whether a
# particular execution API supports every stored setting.

pipeline = onnx.PipelineProto(
    name="three_token_cycle",
    embedding=embedding,
    prefill=prefill,
    max_new_tokens=4,
    temperature=0.0,
    seed=42,
    eos_token_id=2,
    pad_token_id=0,
    top_k=0,
    input_ids_name="input_ids",
    logits_name="logits",
    attention_mask_name="attention_mask",
    position_ids_name="position_ids",
)
verify.verify_pipeline(pipeline)
assert pipeline.HasField("prefill")
assert pipeline.HasField("embedding")
assert not pipeline.HasField("decode")
assert not pipeline.HasField("top_p")

# %%
# Save and reload the binary container
# -------------------------------------
#
# The .pipeline and .pipe-onnx extensions select PipelineProto when loading.
# A temporary directory keeps the example from leaving files behind, including
# when Sphinx Gallery executes it.

with TemporaryDirectory() as directory:
    path = Path(directory) / "cycle.pipeline"
    onnx.save(pipeline, path)
    restored = onnx.load(path)

verify.verify_pipeline(restored)
assert not restored.HasField("top_p")
for stage in ("embedding", "prefill", "decode"):
    assert restored.HasField(stage) == pipeline.HasField(stage)
    if pipeline.HasField(stage):
        assert (
            getattr(restored, stage).SerializeToString()
            == getattr(pipeline, stage).SerializeToString()
        )
for setting in (
    "name",
    "max_new_tokens",
    "temperature",
    "seed",
    "eos_token_id",
    "pad_token_id",
    "top_k",
    "input_ids_name",
    "logits_name",
    "attention_mask_name",
    "position_ids_name",
):
    assert restored.HasField(setting)
    assert getattr(restored, setting) == getattr(pipeline, setting)
print("Binary round trip preserved stages, settings, and interface names.")

# %%
# Generate from the restored models
# -----------------------------------
#
# ReferenceEvaluator imports the native runtime, which registers its kernels
# unconditionally. The supported split-model Python API runs embedding before
# prefill on every step. It does not automatically read PipelineProto settings:
# forward them explicitly, using the standard tensor names configured above.

embedding_evaluator = ReferenceEvaluator(restored.embedding)
evaluator = ReferenceEvaluator(restored.prefill)
feeds = {restored.input_ids_name: numpy.array([[0]], dtype=numpy.int64)}
tokens = evaluator.generate(
    feeds,
    embedding=embedding_evaluator,
    max_new_tokens=restored.max_new_tokens,
    temperature=restored.temperature,
    seed=restored.seed,
    eos_token_id=restored.eos_token_id,
    pad_token_id=restored.pad_token_id,
)
numpy.testing.assert_array_equal(tokens, [[0, 1, 2]])
assert tokens.dtype == numpy.int64
numpy.testing.assert_array_equal(feeds["input_ids"], [[0]])
print("Tokens (including prompt, stopping at EOS):", tokens.tolist())

# %%
# Disable EOS to observe the token limit
# --------------------------------------
#
# This is an explicit execution-time override; the saved pipeline is unchanged.

limited = evaluator.generate(
    feeds,
    embedding=embedding_evaluator,
    max_new_tokens=restored.max_new_tokens,
    temperature=restored.temperature,
    seed=restored.seed,
)
numpy.testing.assert_array_equal(limited, [[0, 1, 2, 0, 1]])
assert restored.eos_token_id == 2
print("Tokens (including prompt, stopping at token limit):", limited.tolist())
