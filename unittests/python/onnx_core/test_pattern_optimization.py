# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests Python graph-pattern optimization and extension points."""

from __future__ import annotations

import gc
import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx_lib import parser

optim = import_or_skip("onnx_light.onnx_core.optimization")


def _double_neg_model():
    """Builds a model containing two consecutive Neg nodes."""
    return parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 18]>\n'
        "agraph (float[2] x) => (float[2] y) {\n"
        "  middle = Neg(x)\n"
        "  y = Neg(middle)\n"
        "}\n"
    )


class NegNegPattern(optim.PatternOptimization):
    """Replaces two consecutive Neg nodes with Identity."""

    def __init__(self, priority: int = 1):
        super().__init__(priority=priority, name="NegNeg")

    def fast_op_type(self):
        return {"Neg"}

    def match(self, graph, node):
        previous = graph.node_before(node.input[0])
        if previous is None or previous.op_type != "Neg":
            return self.no_match(node, "the input is not produced by Neg")
        return self.result([previous, node], insert_at=node)

    def apply(self, graph, nodes):
        del graph
        previous, node = nodes
        return [helper.make_node("Identity", [previous.input[0]], list(node.output))]


class TestPatternOptimization(ExtTestCase):
    def test_python_pattern_rewrites_and_reports_rejections(self):
        model = _double_neg_model()
        optimized, rewrites, report = optim.optimize(model, [NegNegPattern()], report=True)

        self.assertEqual([node.op_type for node in optimized.graph.node], ["Identity"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(rewrites[0].pattern_name, "NegNeg")
        self.assertEqual(report.rewrites, 1)
        self.assertEqual(report.patterns[0].matches, 1)
        self.assertGreaterEqual(
            sum(item.occurrences for item in report.patterns[0].no_matches), 1
        )

    def test_replay_reproduces_python_rewrite(self):
        model = _double_neg_model()
        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, [NegNegPattern()])
        rewrites = graph.optimize()
        optimized = builder.build_graph()
        replayed = optim.replay(model, rewrites)

        self.assertEqual(replayed.SerializeToString(), optimized.SerializeToString())

    def test_optimizer_retains_python_pattern(self):
        model = _double_neg_model()
        builder = optim.GraphBuilder(model)
        pattern = NegNegPattern()
        graph = optim.GraphGraph(builder, [pattern])
        del pattern
        gc.collect()

        rewrites = graph.optimize()
        self.assertEqual(len(rewrites), 1)

    def test_python_exception_propagates(self):
        class FailingPattern(NegNegPattern):
            def match(self, graph, node):
                del graph, node
                raise RuntimeError("matcher failed")

        builder = optim.GraphBuilder(_double_neg_model())
        graph = optim.GraphGraph(builder, [FailingPattern()])
        with self.assertRaisesRegex(RuntimeError, "matcher failed"):
            graph.optimize()

    def test_standard_cast_pattern_is_selectable(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[2] x) => (float[2] y) {\n"
            "  y = Cast <to=1> (x)\n"
            "}\n"
        )
        optimized, rewrites = optim.optimize(model, ["Cast"])

        self.assertEqual(optim.registered_pattern_names()[0], "Cast")
        self.assertIsInstance(optim.CastPattern(), optim.PatternOptimization)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Identity"])
        self.assertEqual(rewrites[0].pattern_name, "Cast")

    def test_python_pattern_runs_recursively_in_subgraph(self):
        then_branch = helper.make_graph(
            [
                helper.make_node("Neg", ["X"], ["middle"]),
                helper.make_node("Neg", ["middle"], ["then_out"]),
            ],
            "then_branch",
            [],
            [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
        )
        else_branch = helper.make_graph(
            [helper.make_node("Identity", ["X"], ["else_out"])],
            "else_branch",
            [],
            [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "If", ["cond"], ["Y"], then_branch=then_branch, else_branch=else_branch
                )
            ],
            "main",
            [
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [2]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        _, rewrites, report = optim.optimize(model, [NegNegPattern()], report=True)

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(rewrites[0].graph_path, ["then_branch"])
        self.assertTrue(any(item.graph_path == ["then_branch"] for item in report.subgraphs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
