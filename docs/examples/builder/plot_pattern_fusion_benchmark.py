"""
Benchmark pattern fusion against onnxscript
===========================================

This example compares *onnx-light* and *onnxscript* on the same synthetic
workload. Construction and fusion are timed separately:

* **construction** builds and exports the unfused model;
* **fusion** starts after conversion to each optimizer's native graph, then
  measures indexing, matching, and rewriting without model construction or
  ONNX serialization.

Every block contains four patterns: three two-node patterns
(``Add``-``Mul``, ``MatMul``-``Add``, and ``Add``-``Relu``) and one six-node
pattern (``Sub``-``Abs``-``Sqrt``-``Neg``-``Exp``-``Log``). Each match is
replaced by one custom benchmark node.
"""

from __future__ import annotations

import gc
import statistics
import time

import matplotlib.pyplot
import numpy
import onnx_ir
import onnxscript
from onnxscript import rewriter

import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_core.optimization import (
    GraphBuilder as OptimizationGraphBuilder,
    GraphGraph,
    PatternOptimization,
)

BLOCK_COUNTS = (50, 100, 250, 500)
FUSIONS_PER_BLOCK = 4
NODES_PER_BLOCK = 12
OPSET = 18


def fusion_blocks(op, value, bias, scale, weight, block_count: int):
    """Builds four fusible subgraphs per block."""
    if block_count <= 0:
        raise ValueError("block_count must be positive.")
    for _ in range(block_count):
        value = op.Mul(op.Add(value, bias), scale)
        value = op.Add(op.MatMul(value, weight), bias)
        value = op.Relu(op.Add(value, bias))
        value = op.Log(op.Exp(op.Neg(op.Sqrt(op.Abs(op.Sub(value, bias))))))
    return value


def build_light(block_count: int):
    """Builds and exports the fusion workload with onnx-light."""
    builder = GraphBuilder("fusion")
    builder.set_opset_version("", OPSET)
    value = builder.inp("X", TensorProto.FLOAT, ["N", 16])
    bias = builder.init(numpy.ones((1, 16), dtype=numpy.float32), name="bias")
    scale = builder.init(numpy.array(0.5, dtype=numpy.float32), name="scale")
    weight = builder.init(numpy.eye(16, dtype=numpy.float32), name="weight")
    value = fusion_blocks(builder.op, value, bias, scale, weight, block_count)
    builder.out(value, TensorProto.FLOAT, ["N", 16])
    return builder.to_onnx("model")


def build_onnxscript(block_count: int):
    """Builds and exports the fusion workload with onnxscript."""
    graph = onnx_ir.Graph(
        inputs=[], outputs=[], nodes=[], opset_imports={"": OPSET}, name="fusion"
    )
    builder = onnxscript.GraphBuilder(graph)
    value = builder.input("X", dtype=onnx_ir.DataType.FLOAT, shape=["N", 16])
    bias = builder.initializer(
        onnx_ir.tensor(numpy.ones((1, 16), dtype=numpy.float32)), name="bias"
    )
    scale = builder.initializer(
        onnx_ir.tensor(numpy.array(0.5, dtype=numpy.float32)), name="scale"
    )
    weight = builder.initializer(
        onnx_ir.tensor(numpy.eye(16, dtype=numpy.float32)), name="weight"
    )
    value = fusion_blocks(builder.op, value, bias, scale, weight, block_count)
    value.type = onnx_ir.TensorType(onnx_ir.DataType.FLOAT)
    value.shape = onnx_ir.Shape(["N", 16])
    builder.add_output(value, None)
    return onnx_ir.to_proto(onnx_ir.Model(graph, ir_version=10))


class ChainFusionPattern(PatternOptimization):
    """Fuses one fixed operator chain into a custom benchmark node."""

    def __init__(self, name, op_types, fused_op, replacement_inputs):
        super().__init__(priority=1, name=name)
        self.op_types = op_types
        self.fused_op = fused_op
        self.replacement_inputs = replacement_inputs

    def fast_op_type(self):
        """Returns the candidate root operator."""
        return {self.op_types[-1]}

    def match(self, graph, node):
        """Matches the chain and requires exclusive intermediate values."""
        matched = [node]
        current = node
        for op_type in reversed(self.op_types[:-1]):
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
        inputs = [
            nodes[node_index].input[input_index]
            for node_index, input_index in self.replacement_inputs
        ]
        return [
            oh.make_node(
                self.fused_op, inputs, list(nodes[-1].output), domain="onnx_light.benchmark"
            )
        ]


def light_patterns():
    """Returns the onnx-light benchmark patterns."""
    return [
        ChainFusionPattern(
            "AddMulFusion", ("Add", "Mul"), "FusedAddMul", ((0, 0), (0, 1), (1, 1))
        ),
        ChainFusionPattern(
            "MatMulAddFusion", ("MatMul", "Add"), "FusedMatMulAdd", ((0, 0), (0, 1), (1, 1))
        ),
        ChainFusionPattern("AddReluFusion", ("Add", "Relu"), "FusedAddRelu", ((0, 0), (0, 1))),
        ChainFusionPattern(
            "SixNodeFusion",
            ("Sub", "Abs", "Sqrt", "Neg", "Exp", "Log"),
            "FusedSixNode",
            ((0, 0), (0, 1)),
        ),
    ]


def _add_mul_pattern(op, x, bias, scale):
    return op.Mul(op.Add(x, bias), scale)


def _fused_add_mul(op, x, bias, scale):
    return op.FusedAddMul(x, bias, scale, _domain="onnx_light.benchmark", _version=1)


def _matmul_add_pattern(op, x, weight, bias):
    return op.Add(op.MatMul(x, weight), bias)


def _fused_matmul_add(op, x, weight, bias):
    return op.FusedMatMulAdd(x, weight, bias, _domain="onnx_light.benchmark", _version=1)


def _add_relu_pattern(op, x, bias):
    return op.Relu(op.Add(x, bias))


def _fused_add_relu(op, x, bias):
    return op.FusedAddRelu(x, bias, _domain="onnx_light.benchmark", _version=1)


def _six_node_pattern(op, x, bias):
    return op.Log(op.Exp(op.Neg(op.Sqrt(op.Abs(op.Sub(x, bias))))))


def _fused_six_node(op, x, bias):
    return op.FusedSixNode(x, bias, _domain="onnx_light.benchmark", _version=1)


ONNXSCRIPT_RULES = rewriter.RewriteRuleSet(
    [
        rewriter.RewriteRule(_add_mul_pattern, _fused_add_mul, name="AddMulFusion"),
        rewriter.RewriteRule(_matmul_add_pattern, _fused_matmul_add, name="MatMulAddFusion"),
        rewriter.RewriteRule(_add_relu_pattern, _fused_add_relu, name="AddReluFusion"),
        rewriter.RewriteRule(_six_node_pattern, _fused_six_node, name="SixNodeFusion"),
    ]
)


def check_models(block_count: int) -> None:
    """Checks that both builders produce the same unfused operator sequence."""
    light = build_light(block_count)
    scripted = build_onnxscript(block_count)
    light_ops = [node.op_type for node in light.graph.node]
    scripted_ops = [node.op_type for node in scripted.graph.node]
    assert light_ops == scripted_ops
    assert len(light_ops) == block_count * NODES_PER_BLOCK


def measure_construction(build, block_count: int, repeats: int = 3) -> float:
    """Returns median model construction and export time in milliseconds."""
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        model = build(block_count)
        samples.append((time.perf_counter() - start) * 1000)
        del model
    return statistics.median(samples)


def measure_light_fusion(model, repeats: int = 3) -> float:
    """Returns median onnx-light indexing, matching, and rewriting time."""
    samples = []
    expected = len(model.graph.node) // NODES_PER_BLOCK * FUSIONS_PER_BLOCK
    for _ in range(repeats):
        builder = OptimizationGraphBuilder(model)
        gc.collect()
        start = time.perf_counter()
        rewrites = GraphGraph(builder, light_patterns()).optimize()
        samples.append((time.perf_counter() - start) * 1000)
        assert len(rewrites) == expected
    return statistics.median(samples)


def measure_onnxscript_fusion(model, repeats: int = 3) -> float:
    """Returns median onnxscript matching and rewriting time."""
    samples = []
    expected = len(model.graph.node) // NODES_PER_BLOCK * FUSIONS_PER_BLOCK
    for _ in range(repeats):
        ir_model = onnx_ir.from_proto(model)
        gc.collect()
        start = time.perf_counter()
        count = ONNXSCRIPT_RULES.apply_to_model(ir_model)
        samples.append((time.perf_counter() - start) * 1000)
        assert count == expected
    return statistics.median(samples)


def plot_benchmark(results):
    """Plots model construction and pattern-fusion timings separately."""
    figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(11, 4), sharex=True)
    for axis, metric, title in (
        (axes[0], "construction", "Model construction and export"),
        (axes[1], "fusion", "Pattern fusion only"),
    ):
        for implementation in ("onnx-light", "onnxscript"):
            rows = [row for row in results if row["implementation"] == implementation]
            axis.plot(
                [row["patterns"] for row in rows],
                [row[metric] for row in rows],
                "o-",
                label=implementation,
            )
        axis.set_title(title)
        axis.set_xlabel("number of fused patterns")
        axis.set_ylabel("median time (ms)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    print("patterns  implementation  construction (ms)  fusion (ms)")
    results = []
    for count in BLOCK_COUNTS:
        check_models(count)
        light = build_light(count)
        scripted = build_onnxscript(count)
        pattern_count = count * FUSIONS_PER_BLOCK
        for name, build, fusion in (
            ("onnx-light", build_light, measure_light_fusion(light)),
            ("onnxscript", build_onnxscript, measure_onnxscript_fusion(scripted)),
        ):
            construction = measure_construction(build, count)
            results.append(
                {
                    "patterns": pattern_count,
                    "implementation": name,
                    "construction": construction,
                    "fusion": fusion,
                }
            )
            print(f"{pattern_count:8}  {name:14}  {construction:17.2f}  {fusion:11.2f}")
    plot_benchmark(results)
    matplotlib.pyplot.show()
