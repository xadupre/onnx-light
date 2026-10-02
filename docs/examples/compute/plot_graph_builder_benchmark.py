"""
Benchmark GraphBuilder against onnxscript GraphBuilder
======================================================

This example builds the same alternating Add/Relu chain with 100, 200, and
500 nodes using :class:`onnx_light.onnx_core.graph_builder.GraphBuilder` and
:class:`onnxscript.GraphBuilder`. It checks that both models produce the same
outputs before measuring model construction with and without final protobuf
serialization. The timings include model finalization (``to_onnx`` or
``onnx_ir.to_proto``), but exclude imports, input generation, and execution.

Run this example with the ``docs`` optional dependencies installed.
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
SHAPE = [2, 3]


def build_light(node_count: int):
    """Builds a chain with onnx-light and returns its model."""
    builder = GraphBuilder("chain")
    builder.set_opset_version("", OPSET)
    value = builder.inp("X", TensorProto.FLOAT, SHAPE)
    bias = builder.inp("B", TensorProto.FLOAT, SHAPE)
    for index in range(node_count):
        value = (
            builder.op.Add(value, bias, outputs=f"v{index}")
            if index % 2 == 0
            else builder.op.Relu(value, outputs=f"v{index}")
        )
    builder.out(value, TensorProto.FLOAT, SHAPE)
    return builder.to_onnx("model")


def build_onnxscript(node_count: int):
    """Builds the same chain with onnxscript and returns its model."""
    graph = onnx_ir.Graph(
        inputs=[], outputs=[], nodes=[], opset_imports={"": OPSET}, name="chain"
    )
    builder = onnxscript.GraphBuilder(graph)
    value = builder.input("X", dtype=onnx_ir.DataType.FLOAT, shape=SHAPE)
    bias = builder.input("B", dtype=onnx_ir.DataType.FLOAT, shape=SHAPE)
    for index in range(node_count):
        value = builder.op.Add(value, bias) if index % 2 == 0 else builder.op.Relu(value)
        value.name = f"v{index}"
    builder.add_output(value, None)
    return onnx_ir.to_proto(onnx_ir.Model(graph, ir_version=10))


def check_models(node_count: int) -> None:
    """Checks both models and compares their outputs on the same inputs."""
    light = onnx.load_from_string(build_light(node_count).SerializeToString())
    scripted = build_onnxscript(node_count)
    for model in (light, scripted):
        onnx.checker.check_model(model)
        assert len(model.graph.node) == node_count
    assert [output.name for output in light.graph.output] == [
        output.name for output in scripted.graph.output
    ]
    # ReferenceEvaluator expects the default domain as "" rather than "ai.onnx".
    for opset in light.opset_import:
        if opset.domain == "ai.onnx":
            opset.domain = ""
    feeds = {
        "X": numpy.arange(6, dtype=numpy.float32).reshape(SHAPE) - 2,
        "B": numpy.full(SHAPE, 0.25, dtype=numpy.float32),
    }
    light_outputs = ReferenceEvaluator(light).run(None, feeds)
    scripted_outputs = ReferenceEvaluator(scripted).run(None, feeds)
    assert len(light_outputs) == len(scripted_outputs) == 1
    numpy.testing.assert_allclose(light_outputs[0], scripted_outputs[0], rtol=0, atol=0)


def measure(build, node_count: int, serialize: bool, repeats: int = 3) -> float:
    """Returns the median construction time in milliseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model = build(node_count)
        if serialize:
            model.SerializeToString()
        samples.append((time.perf_counter() - start) * 1000)
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
