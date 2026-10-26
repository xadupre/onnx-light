# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests optimization callbacks without requiring operator kernels."""

import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
from onnx_light.onnx_lib import parser
from onnx_light.onnx_py import _onnxpycore
from onnx_light.onnx_core import optimization as optim


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

    def __init__(self, name="NegNeg"):
        super().__init__(priority=1, name=name)

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
        return [oh.make_node("Identity", [previous.input[0]], list(node.output))]


class TestOptimizationCallback(ExtTestCase):
    def test_callback_accepts_or_rejects_native_matches(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[2,3] x) => (float[2,3] y) {\n"
            "  middle = Transpose <perm: ints = [1, 0]> (x)\n"
            "  y = Transpose <perm: ints = [1, 0]> (middle)\n"
            "}\n"
        )
        for graph_type in (optim.GraphGraph, _onnxpycore.builder.GraphGraph):
            for report in (False, True):
                for decision in (None, True, False):
                    with self.subTest(graph_type=graph_type, report=report, decision=decision):
                        builder = optim.GraphBuilder(model)
                        graph = graph_type(builder, [optim.TransposeTransposePattern()])
                        observed = []

                        def callback(scope, match):
                            self.assertIs(scope, graph)
                            self.assertTrue(match.matched)
                            self.assertEqual(match.pattern.name, "TransposeTranspose")
                            self.assertEqual([scope.position(n) for n in match.nodes], [0, 1])
                            self.assertTrue(scope.has_shape("y"))
                            observed.append([n.op_type for n in match.nodes])
                            return decision

                        result = graph.optimize(report=report, callback=callback)
                        rewrites = result[0] if report else result
                        self.assertEqual(observed, [["Transpose", "Transpose"]])
                        self.assertEqual(len(rewrites.rewrites), int(decision is not False))
                        self.assertEqual(
                            [n.op_type for n in builder.build_graph().node],
                            ["Transpose", "Transpose"] if decision is False else ["Identity"],
                        )
                        if report:
                            self.assertEqual(
                                result[1].patterns[0].matches, int(decision is not False)
                            )
                        replayed = optim.apply_rewritings(model, rewrites)
                        self.assertEqual(
                            replayed.SerializeToString(),
                            builder.build_graph().SerializeToString(),
                        )

    def test_rejection_does_not_reserve_nodes_for_other_patterns(self):
        builder = optim.GraphBuilder(_double_neg_model())
        graph = optim.GraphGraph(
            builder, [NegNegPattern(name="Rejected"), NegNegPattern(name="Accepted")]
        )
        observed = []

        def callback(scope, match):
            del scope
            observed.append(match.pattern.name)
            return match.pattern.name != "Rejected"

        rewrites, report = graph.optimize(report=True, callback=callback)
        self.assertEqual(observed, ["Rejected", "Accepted"])
        self.assertEqual([r.pattern_name for r in rewrites.rewrites], ["Accepted"])
        self.assertEqual([p.matches for p in report.patterns], [0, 1])
        self.assertEqual([n.op_type for n in builder.build_graph().node], ["Identity"])

    def test_callback_runs_in_nested_graphs(self):
        branches = {}
        for name in ("then_branch", "else_branch"):
            template = _double_neg_model().graph
            branches[name] = oh.make_graph(template.node, name, [], template.output)
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("If", ["cond"], ["out"], **branches)],
                "main",
                [
                    oh.make_tensor_value_info("cond", TensorProto.BOOL, []),
                    oh.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                ],
                [oh.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        builder = optim.GraphBuilder(model)
        observed = []

        def callback(scope, match):
            self.assertEqual(match.pattern.name, "NegNeg")
            observed.append(scope.name)
            return scope.name == "then_branch"

        rewrites = optim.GraphGraph(builder, [NegNegPattern()]).optimize(callback=callback)
        self.assertEqual(set(observed), {"then_branch", "else_branch"})
        self.assertEqual(len(observed), 2)
        self.assertEqual([list(r.graph_path) for r in rewrites.rewrites], [["then_branch"]])
        self.assertEqual(
            optim.apply_rewritings(model, rewrites).SerializeToString(),
            builder.build_graph().SerializeToString(),
        )

    def test_callback_exceptions_propagate_before_application(self):
        builder = optim.GraphBuilder(_double_neg_model())
        before = builder.build_graph().SerializeToString()

        def callback(scope, match):
            del scope, match
            raise ValueError("callback failure")

        graph = optim.GraphGraph(builder, [NegNegPattern()])
        with self.assertRaisesRegex(ValueError, "callback failure"):
            graph.optimize(callback=callback)
        self.assertEqual(builder.build_graph().SerializeToString(), before)
        self.assertEqual(len(graph.optimize().rewrites), 1)

    def test_callback_validation_and_cleanup_only(self):
        builder = optim.GraphBuilder(_double_neg_model())
        graph = optim.GraphGraph(builder, patterns=False)
        with self.assertRaisesRegex(TypeError, "callback must be callable"):
            graph.optimize(callback=42)
        observed = []
        self.assertEqual(
            len(graph.optimize(callback=lambda *args: observed.append(args)).rewrites), 0
        )
        self.assertEqual(observed, [])
        graph = optim.GraphGraph(builder, [NegNegPattern()])
        self.assertEqual(
            len(
                graph.optimize(max_iter=0, callback=lambda *args: observed.append(args)).rewrites
            ),
            0,
        )
        self.assertEqual(observed, [])
        self.assertEqual(len(graph.optimize(callback=None).rewrites), 1)


if __name__ == "__main__":
    unittest.main()
