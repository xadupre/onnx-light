"""
Benchmark GraphBuilder against onnxscript GraphBuilder
======================================================

This example builds the same attention-style graph with 100, 200, and
500 nodes using :class:`onnx_light.onnx_core.graph_builder.GraphBuilder` and
:class:`onnxscript.GraphBuilder`. It checks that both models produce the same
outputs before measuring model construction with and without final protobuf
serialization. The timings include model finalization (``to_onnx`` or
``onnx_ir.to_proto``), but exclude imports, input generation, and execution.

Run this example with the ``docs`` optional dependencies installed.

Each 20-node block splits a symbolic width into two heads using runtime
``Shape``, ``Gather``, ``Div``, and ``Concat`` operations. It reshapes and
transposes the input, computes scaled dot-product self-attention, then merges
the heads and applies an averaged residual connection and ``Relu``. Batch size,
sequence length, and width are all dynamic; width must be divisible by two.
The same exported models are checked on several shapes, including a singleton
sequence. Both builders receive identical operators and initializers; their
default shape-inference behavior is included in the construction timings.
"""

from __future__ import annotations

import statistics
import time

import numpy
import onnx
import onnx_ir
import onnxscript
from onnx.reference import ReferenceEvaluator

from onnx_light.onnx import TensorProto
from onnx_light.onnx_core.graph_builder import GraphBuilder

NODE_COUNTS = (100, 200, 500)
OPSET = 18
SHAPE = ["batch", "sequence", "width"]
INPUT_SHAPES = ((1, 1, 4), (2, 3, 6), (3, 5, 8))
BLOCK_SIZE = 20


def attention_blocks(op, value, constants, node_count: int):
    """Builds attention blocks with runtime-derived head and output shapes."""
    if node_count <= 0 or node_count % BLOCK_SIZE:
        raise ValueError(f"node_count must be a positive multiple of {BLOCK_SIZE}.")
    batch_index, sequence_index, width_index, heads, half = constants
    for _ in range(node_count // BLOCK_SIZE):
        shape = op.Shape(value)
        batch = op.Gather(shape, batch_index, axis=0)
        sequence = op.Gather(shape, sequence_index, axis=0)
        width = op.Gather(shape, width_index, axis=0)
        head_width = op.Div(width, heads)
        head_shape = op.Concat(batch, sequence, heads, head_width, axis=0)
        query = op.Reshape(value, head_shape)
        query = op.Transpose(query, perm=[0, 2, 1, 3])
        key = op.Transpose(query, perm=[0, 1, 3, 2])
        scores = op.MatMul(query, key)
        scale = op.Cast(head_width, to=TensorProto.FLOAT)
        scale = op.Sqrt(scale)
        scores = op.Div(scores, scale)
        weights = op.Softmax(scores, axis=-1)
        context = op.MatMul(weights, query)
        context = op.Transpose(context, perm=[0, 2, 1, 3])
        context = op.Reshape(context, shape)
        value = op.Add(value, context)
        value = op.Mul(value, half)
        value = op.Relu(value)
    return value


def constants():
    """Returns the shared shape indices, head count, and residual scale."""
    return (
        numpy.array([0], dtype=numpy.int64),
        numpy.array([1], dtype=numpy.int64),
        numpy.array([2], dtype=numpy.int64),
        numpy.array([2], dtype=numpy.int64),
        numpy.array(0.5, dtype=numpy.float32),
    )


def build_light(node_count: int):
    """Builds dynamic attention blocks with onnx-light and returns its model."""
    builder = GraphBuilder("attention")
    builder.set_opset_version("", OPSET)
    value = builder.inp("X", TensorProto.FLOAT, SHAPE)
    initializers = [
        builder.init(array, name=f"c{index}") for index, array in enumerate(constants())
    ]
    value = attention_blocks(builder.op, value, initializers, node_count)
    builder.out(value, TensorProto.FLOAT, SHAPE)
    return builder.to_onnx("model")


def build_onnxscript(node_count: int):
    """Builds the same dynamic attention blocks with onnxscript."""
    graph = onnx_ir.Graph(
        inputs=[], outputs=[], nodes=[], opset_imports={"": OPSET}, name="attention"
    )
    builder = onnxscript.GraphBuilder(graph)
    value = builder.input("X", dtype=onnx_ir.DataType.FLOAT, shape=SHAPE)
    initializers = [
        builder.initializer(onnx_ir.tensor(array), name=f"c{index}")
        for index, array in enumerate(constants())
    ]
    value = attention_blocks(builder.op, value, initializers, node_count)
    value.shape = onnx_ir.Shape(SHAPE)
    value.type = onnx_ir.TensorType(onnx_ir.DataType.FLOAT)
    builder.add_output(value, None)
    return onnx_ir.to_proto(onnx_ir.Model(graph, ir_version=10))


def check_models(node_count: int) -> None:
    """Checks both models and compares outputs across dynamic input shapes."""
    light = onnx.load_from_string(build_light(node_count).SerializeToString())
    scripted = build_onnxscript(node_count)
    # ONNX's checker and evaluator expect "" rather than "ai.onnx".
    for opset in light.opset_import:
        if opset.domain == "ai.onnx":
            opset.domain = ""
    for model in (light, scripted):
        onnx.checker.check_model(model)
        assert len(model.graph.node) == node_count
    light_session = ReferenceEvaluator(light)
    scripted_session = ReferenceEvaluator(scripted)
    rng = numpy.random.default_rng(0)
    for shape in INPUT_SHAPES:
        feeds = {"X": rng.standard_normal(shape).astype(numpy.float32)}
        light_outputs = light_session.run(None, feeds)
        scripted_outputs = scripted_session.run(None, feeds)
        assert len(light_outputs) == len(scripted_outputs) == 1
        assert light_outputs[0].shape == scripted_outputs[0].shape == shape
        numpy.testing.assert_allclose(light_outputs[0], scripted_outputs[0], rtol=0, atol=0)


def measure(build, node_count: int, serialize: bool, repeats: int = 3) -> float:
    """Returns the median construction time in milliseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model = build(node_count)
        if serialize:
            model.SerializeToString()
        elapsed = (time.perf_counter() - start) * 1000
        samples.append(elapsed)
        del model
    return statistics.median(samples)


if __name__ == "__main__":
    print("nodes  builder       model (ms)  model + serialization (ms)")
    for count in NODE_COUNTS:
        check_models(count)
        for name, build in (("onnx-light", build_light), ("onnxscript", build_onnxscript)):
            print(
                f"{count:5}  {name:12}  {measure(build, count, False):10.2f}"
                f"  {measure(build, count, True):26.2f}"
            )
