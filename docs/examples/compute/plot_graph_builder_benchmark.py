"""
Benchmark GraphBuilder against onnxscript GraphBuilder
======================================================

This example builds the same attention-style graph with 100, 200, and
500 nodes using :class:`onnx_light.onnx_core.graph_builder.GraphBuilder` and
:class:`onnxscript.GraphBuilder`. It checks that both models produce the same
outputs before measuring model construction with and without final protobuf
serialization. The timings include model finalization (``to_onnx`` or
``onnx_ir.to_proto``), but exclude imports, input generation, and execution.
The example also reports the operator-type distribution and plots both timing
modes against the graph size.

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

from collections import Counter
import gc
import statistics
import time

import matplotlib.pyplot
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
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            gc.collect()
            start = time.perf_counter()
            model = build(node_count)
            if serialize:
                model.SerializeToString()
            elapsed = (time.perf_counter() - start) * 1000
            samples.append(elapsed)
            del model
    finally:
        gc.collect()
        if gc_was_enabled:
            gc.enable()
    return statistics.median(samples)


def node_type_distribution(model) -> Counter:
    """Counts nodes by operator type."""
    return Counter(node.op_type for node in model.graph.node)


def format_node_type_table(distributions: dict[int, Counter]) -> str:
    """Formats node-type counts for every benchmark graph size."""
    node_counts = sorted(distributions)
    operator_types = sorted(
        {
            operator_type
            for distribution in distributions.values()
            for operator_type in distribution
        }
    )
    header = f"{'operator':<12}" + "".join(f"{node_count:>8}" for node_count in node_counts)
    separator = "-" * len(header)
    rows = [header, separator]
    for operator_type in operator_types:
        rows.append(
            f"{operator_type:<12}"
            + "".join(
                f"{distributions[node_count].get(operator_type, 0):>8}"
                for node_count in node_counts
            )
        )
    rows.extend(
        [
            separator,
            f"{'Total':<12}"
            + "".join(
                f"{sum(distributions[node_count].values()):>8}" for node_count in node_counts
            ),
        ]
    )
    return "\n".join(rows)


def plot_benchmark(results: list[dict]):
    """Plots construction times with and without serialization."""
    figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(11, 4), sharex=True)
    for axis, key, title in (
        (axes[0], "model", "Model construction"),
        (axes[1], "serialized", "Model construction and serialization"),
    ):
        for builder_name in ("onnx-light", "onnxscript"):
            rows = [row for row in results if row["builder"] == builder_name]
            axis.plot(
                [row["nodes"] for row in rows],
                [row[key] for row in rows],
                "o-",
                label=builder_name,
            )
        axis.set_title(title)
        axis.set_xlabel("number of nodes")
        axis.set_ylabel("median time (ms)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    print("nodes  builder       model (ms)  model + serialization (ms)")
    results = []
    distributions = {}
    for count in NODE_COUNTS:
        check_models(count)
        distribution_model = build_light(count)
        distributions[count] = node_type_distribution(distribution_model)
        del distribution_model
        for name, build in (("onnx-light", build_light), ("onnxscript", build_onnxscript)):
            model_time = measure(build, count, False)
            serialized_time = measure(build, count, True)
            results.append(
                {
                    "nodes": count,
                    "builder": name,
                    "model": model_time,
                    "serialized": serialized_time,
                }
            )
            print(f"{count:5}  {name:12}  {model_time:10.2f}  {serialized_time:26.2f}")
    print("\nNode-type distribution (identical for both builders):")
    print(format_node_type_table(distributions))
    plot_benchmark(results)
    matplotlib.pyplot.show()
