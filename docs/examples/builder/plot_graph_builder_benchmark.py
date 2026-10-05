"""
Benchmark GraphBuilder and pattern fusion against onnxscript
=============================================================

This example builds the same attention-style graph with 100, 200, and
500 nodes using :class:`onnx_light.onnx_core.graph_builder.GraphBuilder` and
:class:`onnxscript.GraphBuilder`. It checks that both models produce the same
outputs before measuring model construction with and without in-memory
serialization. The timings include model finalization (``to_onnx`` or
``onnx_ir.to_proto``), but exclude imports, input generation, and execution.
The serialization column additionally calls ``SerializeToString``: it creates
the complete protobuf wire representation in memory rather than writing it to
a file.

Run this example with the ``docs`` optional dependencies installed.

Each 20-node block splits a symbolic width into two heads using runtime
``Shape``, ``Gather``, ``Div``, and ``Concat`` operations. It reshapes and
transposes the input, computes scaled dot-product self-attention, then merges
the heads and applies an averaged residual connection and ``Relu``. Batch size,
sequence length, and width are all dynamic; width must be divisible by two.
The same exported models are checked on several shapes, including a singleton
sequence. Both builders receive identical operators and initializers; their
default shape-inference behavior is included in the construction timings.

To make the serialization cost visible, the timed models also expose four
large ``UINT8`` initializers as outputs. They share one source array while the
builders are running, but each initializer is present in the final model, whose
serialized size is approximately 1.5 GB. The functional checks omit this large
payload.

The final section compares pattern-fusion throughput with four rules:
``Add``-``Mul``, ``MatMul``-``Add``, ``Add``-``Relu``, and a six-node
``Sub``-``Abs``-``Sqrt``-``Neg``-``Exp``-``Log`` chain. Both implementations
start from the same ONNX model and include conversion into their native graph
representation, matching, rewriting, and conversion back to ONNX in the
measured time.
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
from onnxscript import rewriter

from onnx_light.onnx import TensorProto
import onnx_light.onnx.helper as oh
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_core.optimization import (
    GraphBuilder as OptimizationGraphBuilder,
    GraphGraph,
    PatternOptimization,
)

NODE_COUNTS = (100, 200, 500, 1000, 2000)
OPSET = 18
SHAPE = ["batch", "sequence", "width"]
INPUT_SHAPES = ((1, 1, 4), (2, 3, 6), (3, 5, 8))
BLOCK_SIZE = 20
LARGE_INITIALIZER_BYTES = 375_000_000
LARGE_INITIALIZER_COUNT = 4
_LARGE_INITIALIZER = None
FUSION_BLOCK_COUNTS = (50, 100, 250, 500)
FUSIONS_PER_BLOCK = 4


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


def large_initializer():
    """Returns the shared array used to produce an approximately 1.5 GB model."""
    global _LARGE_INITIALIZER
    if _LARGE_INITIALIZER is None:
        _LARGE_INITIALIZER = numpy.zeros(LARGE_INITIALIZER_BYTES, dtype=numpy.uint8)
    return _LARGE_INITIALIZER


def build_light(node_count: int, large_initializers: bool = False):
    """Builds dynamic attention blocks with onnx-light and returns its model."""
    builder = GraphBuilder("attention")
    builder.set_opset_version("", OPSET)
    value = builder.inp("X", TensorProto.FLOAT, SHAPE)
    initializers = [
        builder.init(array, name=f"c{index}") for index, array in enumerate(constants())
    ]
    value = attention_blocks(builder.op, value, initializers, node_count)
    builder.out(value, TensorProto.FLOAT, SHAPE)
    if large_initializers:
        payload = large_initializer()
        for index in range(LARGE_INITIALIZER_COUNT):
            name = builder.init(payload, name=f"large{index}", copy=False)
            builder.out(name, TensorProto.UINT8, [LARGE_INITIALIZER_BYTES])
    return builder.to_onnx("model")


def build_onnxscript(node_count: int, large_initializers: bool = False):
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
    if large_initializers:
        payload = large_initializer()
        for index in range(LARGE_INITIALIZER_COUNT):
            initializer = builder.initializer(onnx_ir.tensor(payload), name=f"large{index}")
            initializer.type = onnx_ir.TensorType(onnx_ir.DataType.UINT8)
            initializer.shape = onnx_ir.Shape([LARGE_INITIALIZER_BYTES])
            builder.add_output(initializer, None)
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


class AddMulFusionPattern(PatternOptimization):
    """Fuses Add followed by Mul into one benchmark operator."""

    def __init__(self):
        super().__init__(priority=1, name="AddMulFusion")

    def fast_op_type(self):
        """Returns the candidate root operator."""
        return {"Mul"}

    def match(self, graph, node):
        """Matches an Add consumed only by the candidate Mul."""
        previous = graph.node_before(node.input[0])
        if previous is None or previous.op_type != "Add":
            return self.no_match(node, "the first input is not produced by Add")
        if len(graph.next_nodes(previous.output[0])) != 1:
            return self.no_match(node, "the Add output has multiple consumers")
        return self.result([previous, node], insert_at=node)

    def apply(self, graph, nodes):
        """Builds the fused replacement node."""
        del graph
        add, mul = nodes
        return [
            oh.make_node(
                "FusedAddMul",
                [add.input[0], add.input[1], mul.input[1]],
                list(mul.output),
                domain="onnx_light.benchmark",
            )
        ]


class MatMulAddFusionPattern(PatternOptimization):
    """Fuses MatMul followed by Add into one benchmark operator."""

    def __init__(self):
        super().__init__(priority=1, name="MatMulAddFusion")

    def fast_op_type(self):
        """Returns the candidate root operator."""
        return {"Add"}

    def match(self, graph, node):
        """Matches a MatMul consumed only by the candidate Add."""
        previous = graph.node_before(node.input[0])
        if previous is None or previous.op_type != "MatMul":
            return self.no_match(node, "the first input is not produced by MatMul")
        if len(graph.next_nodes(previous.output[0])) != 1:
            return self.no_match(node, "the MatMul output has multiple consumers")
        return self.result([previous, node], insert_at=node)

    def apply(self, graph, nodes):
        """Builds the fused replacement node."""
        del graph
        matmul, add = nodes
        return [
            oh.make_node(
                "FusedMatMulAdd",
                [matmul.input[0], matmul.input[1], add.input[1]],
                list(add.output),
                domain="onnx_light.benchmark",
            )
        ]


class AddReluFusionPattern(PatternOptimization):
    """Fuses Add followed by Relu into one benchmark operator."""

    def __init__(self):
        super().__init__(priority=1, name="AddReluFusion")

    def fast_op_type(self):
        """Returns the candidate root operator."""
        return {"Relu"}

    def match(self, graph, node):
        """Matches an Add consumed only by the candidate Relu."""
        previous = graph.node_before(node.input[0])
        if previous is None or previous.op_type != "Add":
            return self.no_match(node, "the input is not produced by Add")
        if len(graph.next_nodes(previous.output[0])) != 1:
            return self.no_match(node, "the Add output has multiple consumers")
        return self.result([previous, node], insert_at=node)

    def apply(self, graph, nodes):
        """Builds the fused replacement node."""
        del graph
        add, relu = nodes
        return [
            oh.make_node(
                "FusedAddRelu", list(add.input), list(relu.output), domain="onnx_light.benchmark"
            )
        ]


class SixNodeFusionPattern(PatternOptimization):
    """Fuses a six-node chain into one benchmark operator."""

    def __init__(self):
        super().__init__(priority=1, name="SixNodeFusion")

    def fast_op_type(self):
        """Returns the candidate root operator."""
        return {"Log"}

    def match(self, graph, node):
        """Matches the chain with exclusive intermediate values."""
        matched = [node]
        current = node
        for op_type in ("Exp", "Neg", "Sqrt", "Abs", "Sub"):
            previous = graph.node_before(current.input[0])
            if previous is None or previous.op_type != op_type:
                return self.no_match(node, f"the chain does not contain {op_type}")
            if len(graph.next_nodes(previous.output[0])) != 1:
                return self.no_match(node, f"the {op_type} output has multiple consumers")
            matched.append(previous)
            current = previous
        matched.reverse()
        return self.result(matched, insert_at=node)

    def apply(self, graph, nodes):
        """Builds the fused replacement node."""
        del graph
        first, *_, last = nodes
        return [
            oh.make_node(
                "FusedSixNode",
                list(first.input),
                list(last.output),
                domain="onnx_light.benchmark",
            )
        ]


def _onnxscript_add_mul_pattern(op, x, bias, scale):
    return op.Mul(op.Add(x, bias), scale)


def _onnxscript_fused_add_mul(op, x, bias, scale):
    return op.FusedAddMul(x, bias, scale, _domain="onnx_light.benchmark", _version=1)


def _onnxscript_matmul_add_pattern(op, x, weight, bias):
    return op.Add(op.MatMul(x, weight), bias)


def _onnxscript_fused_matmul_add(op, x, weight, bias):
    return op.FusedMatMulAdd(x, weight, bias, _domain="onnx_light.benchmark", _version=1)


def _onnxscript_add_relu_pattern(op, x, bias):
    return op.Relu(op.Add(x, bias))


def _onnxscript_fused_add_relu(op, x, bias):
    return op.FusedAddRelu(x, bias, _domain="onnx_light.benchmark", _version=1)


def _onnxscript_six_node_pattern(op, x, bias):
    return op.Log(op.Exp(op.Neg(op.Sqrt(op.Abs(op.Sub(x, bias))))))


def _onnxscript_fused_six_node(op, x, bias):
    return op.FusedSixNode(x, bias, _domain="onnx_light.benchmark", _version=1)


ONNXSCRIPT_FUSION_RULES = rewriter.RewriteRuleSet(
    [
        rewriter.RewriteRule(
            _onnxscript_add_mul_pattern, _onnxscript_fused_add_mul, name="AddMulFusion"
        ),
        rewriter.RewriteRule(
            _onnxscript_matmul_add_pattern, _onnxscript_fused_matmul_add, name="MatMulAddFusion"
        ),
        rewriter.RewriteRule(
            _onnxscript_add_relu_pattern, _onnxscript_fused_add_relu, name="AddReluFusion"
        ),
        rewriter.RewriteRule(
            _onnxscript_six_node_pattern, _onnxscript_fused_six_node, name="SixNodeFusion"
        ),
    ]
)


def build_fusion_model(block_count: int):
    """Builds a graph containing four fusion types per block."""
    if block_count <= 0:
        raise ValueError("block_count must be positive.")
    builder = GraphBuilder("fusion")
    builder.set_opset_version("", OPSET)
    value = builder.inp("X", TensorProto.FLOAT, ["N", 16])
    bias = builder.init(numpy.ones((1, 16), dtype=numpy.float32), name="bias")
    scale = builder.init(numpy.array(0.5, dtype=numpy.float32), name="scale")
    weight = builder.init(numpy.eye(16, dtype=numpy.float32), name="weight")
    for _ in range(block_count):
        value = builder.op.Mul(builder.op.Add(value, bias), scale)
        value = builder.op.Add(builder.op.MatMul(value, weight), bias)
        value = builder.op.Relu(builder.op.Add(value, bias))
        value = builder.op.Log(
            builder.op.Exp(
                builder.op.Neg(builder.op.Sqrt(builder.op.Abs(builder.op.Sub(value, bias))))
            )
        )
    builder.out(value, TensorProto.FLOAT, ["N", 16])
    return builder.to_onnx("model")


def fuse_light(model):
    """Runs the four fusion patterns with onnx-light."""
    builder = OptimizationGraphBuilder(model)
    patterns = [
        AddMulFusionPattern(),
        MatMulAddFusionPattern(),
        AddReluFusionPattern(),
        SixNodeFusionPattern(),
    ]
    rewrites = GraphGraph(builder, patterns).optimize()
    assert len(rewrites) == len(model.graph.node) // 3
    return builder.to_onnx("model")


def fuse_onnxscript(model):
    """Runs the four fusion patterns with onnxscript."""
    proto = onnx.load_from_string(model.SerializeToString())
    ir_model = onnx_ir.from_proto(proto)
    count = ONNXSCRIPT_FUSION_RULES.apply_to_model(ir_model)
    assert count == len(model.graph.node) // 3
    return onnx_ir.to_proto(ir_model)


def measure_fusion(fuse, model, repeats: int = 3) -> float:
    """Returns median end-to-end pattern-fusion time in milliseconds."""
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        optimized = fuse(model)
        samples.append((time.perf_counter() - start) * 1000)
        del optimized
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
    """Plots construction times with and without in-memory serialization."""
    figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(11, 4), sharex=True)
    for axis, key, title in (
        (axes[0], "model", "Model construction"),
        (axes[1], "serialized", "Model construction and in-memory serialization"),
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


def plot_fusion_benchmark(results: list[dict]):
    """Plots multi-pattern fusion times."""
    figure, axis = matplotlib.pyplot.subplots(figsize=(6, 4))
    for optimizer in ("onnx-light", "onnxscript"):
        rows = [row for row in results if row["optimizer"] == optimizer]
        axis.plot(
            [row["patterns"] for row in rows],
            [row["time"] for row in rows],
            "o-",
            label=optimizer,
        )
    axis.set_title("Four pattern fusions (including one six-node pattern)")
    axis.set_xlabel("number of fused patterns")
    axis.set_ylabel("median time (ms)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    print("nodes  builder       model (ms)  model + in-memory serialization (ms)")
    results = []
    distributions = {}
    for count in NODE_COUNTS:
        check_models(count)
        distribution_model = build_light(count)
        distributions[count] = node_type_distribution(distribution_model)
        del distribution_model
        for name, build in (
            ("onnx-light", lambda n: build_light(n, large_initializers=True)),
            ("onnxscript", lambda n: build_onnxscript(n, large_initializers=True)),
        ):
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
            print(f"{count:5}  {name:12}  {model_time:10.2f}  {serialized_time:36.2f}")
    print("\nNode-type distribution (identical for both builders):")
    print(format_node_type_table(distributions))
    plot_benchmark(results)

    print("\npatterns  optimizer     fusion (ms)")
    fusion_results = []
    for count in FUSION_BLOCK_COUNTS:
        source = build_fusion_model(count)
        pattern_count = count * FUSIONS_PER_BLOCK
        for name, fuse in (("onnx-light", fuse_light), ("onnxscript", fuse_onnxscript)):
            elapsed = measure_fusion(fuse, source)
            fusion_results.append({"patterns": pattern_count, "optimizer": name, "time": elapsed})
            print(f"{pattern_count:8}  {name:12}  {elapsed:11.2f}")
    plot_fusion_benchmark(fusion_results)
    matplotlib.pyplot.show()
