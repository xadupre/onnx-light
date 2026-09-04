# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests Python graph-pattern optimization and extension points."""

from __future__ import annotations

import gc
import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
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
        return [oh.make_node("Identity", [previous.input[0]], list(node.output))]


class TestPatternOptimization(ExtTestCase):
    def test_python_pattern_rewrites_and_reports_rejections(self):
        model = _double_neg_model()
        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, [NegNegPattern()])
        rewrites, report = graph.optimize(report=True)
        optimized = builder.to_onnx("model")

        self.assertEqual([node.op_type for node in optimized.graph.node], ["Identity"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(rewrites[0].pattern_name, "NegNeg")
        self.assertEqual(
            repr(rewrites[0]),
            "LocalRewriting("
            "pattern=NegNeg, graph_path=<root>, matched_nodes=2, added_nodes=1)",
        )
        self.assertEqual(str(rewrites[0]), repr(rewrites[0]))
        details = rewrites[0].to_detailed_string()
        self.assertIn("  graph_path: <root>\n", details)
        self.assertIn("  matched_nodes:\n    positions: [0, 1]\n", details)
        self.assertIn(
            "  added_nodes:\n    nodes: [Identity(outputs=[y])]\n    positions: [0]\n", details
        )
        self.assertIn("  initializers:\n", details)
        self.assertIn("  value_renames: []\n", details)
        self.assertIn("  timings:\n    match_time_ns: ", details)
        self.assertEqual(report.rewrites, 1)
        pattern_report = next(item for item in report.patterns if item.pattern_name == "NegNeg")
        self.assertEqual(pattern_report.matches, 1)
        self.assertGreaterEqual(sum(item.occurrences for item in pattern_report.no_matches), 1)
        self.assertIn(
            "the input is not produced by Neg",
            {item.reason for item in pattern_report.no_matches},
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

    def test_global_pattern_registration(self):
        optim.register_pattern(NegNegPattern())
        try:
            builder = optim.GraphBuilder(_double_neg_model())
            graph = optim.GraphGraph(builder)
            rewrites = graph.optimize()
            self.assertIn("NegNeg", optim.registered_pattern_names())
            self.assertTrue(any(rewrite.pattern_name == "NegNeg" for rewrite in rewrites))
        finally:
            optim.unregister_pattern("NegNeg")

    def test_builder_pattern_registration(self):
        builder = optim.GraphBuilder(_double_neg_model())
        builder.register_pattern(NegNegPattern())

        graph = optim.GraphGraph(builder, use_global_patterns=False)
        rewrites = graph.optimize()

        self.assertEqual(builder.registered_pattern_names(), ("NegNeg",))
        self.assertEqual(rewrites[0].pattern_name, "NegNeg")

    def test_graph_pattern_registration(self):
        builder = optim.GraphBuilder(_double_neg_model())
        graph = optim.GraphGraph(builder, [NegNegPattern()], use_global_patterns=False)

        rewrites = graph.optimize()
        self.assertEqual(rewrites[0].pattern_name, "NegNeg")

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
        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, optim.standard_patterns(["Cast"]))
        rewrites = graph.optimize()
        optimized = builder.to_onnx("model")

        self.assertEqual(optim.registered_pattern_names()[0], "Cast")
        self.assertIsInstance(optim.CastPattern(), optim.PatternOptimization)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Identity"])
        self.assertEqual(rewrites[0].pattern_name, "Cast")

    def test_standard_clip_clip_pattern_is_selectable(self):
        graph = oh.make_graph(
            [
                oh.make_node("Clip", ["x", "mn"], ["x1"]),
                oh.make_node("Clip", ["x1", "", "mx"], ["y"]),
            ],
            "agraph",
            [
                oh.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                oh.make_tensor_value_info("mn", TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info("mx", TensorProto.FLOAT, [1]),
            ],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, optim.standard_patterns(["ClipClip"]))
        rewrites = graph.optimize()
        optimized = builder.to_onnx("model")

        self.assertIn("ClipClip", optim.registered_pattern_names())
        self.assertIsInstance(optim.ClipClipPattern(), optim.PatternOptimization)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Clip"])
        self.assertEqual(list(optimized.graph.node[0].input), ["x", "mn", "mx"])
        self.assertEqual(rewrites[0].pattern_name, "ClipClip")

    def test_standard_gather_to_slice_pattern_rewrites_vector_singleton(self):
        graph = oh.make_graph(
            [oh.make_node("Gather", ["x", "idx"], ["y"])],
            "agraph",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [5])],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            initializer=[oh.make_tensor("idx", TensorProto.INT64, [1], [3])],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, optim.standard_patterns(["GatherToSlice"]))
        rewrites = graph.optimize()
        optimized = builder.to_onnx("model")

        self.assertIn("GatherToSlice", optim.registered_pattern_names())
        self.assertIsInstance(optim.GatherToSlicePattern(), optim.PatternOptimization)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Slice"])
        self.assertEqual(optimized.graph.node[0].input[0], "x")
        self.assertEqual(optimized.graph.node[0].output[0], "y")
        self.assertEqual(rewrites[0].pattern_name, "GatherToSlice")

    def test_standard_gather_to_slice_pattern_rewrites_scalar_index_with_squeeze(self):
        graph = oh.make_graph(
            [oh.make_node("Gather", ["x", "idx"], ["y"])],
            "agraph",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [5])],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, [])],
            initializer=[oh.make_tensor("idx", TensorProto.INT64, [], [2])],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, optim.standard_patterns(["GatherToSlice"]))
        graph.optimize()
        optimized = builder.to_onnx("model")

        op_types = [node.op_type for node in optimized.graph.node]
        self.assertEqual(op_types, ["Slice", "Squeeze"])
        self.assertEqual(optimized.graph.node[0].input[0], "x")
        self.assertEqual(optimized.graph.node[-1].output[0], "y")

    def test_structural_patterns_are_selectable(self):
        names = set(optim.registered_pattern_names())
        self.assertIn("SliceElimination", names)
        self.assertIn("PadPadFusion", names)
        self.assertIn("ReluClipFusion", names)
        self.assertIsInstance(optim.SliceEliminationPattern(), optim.PatternOptimization)
        self.assertIsInstance(optim.PadPadFusionPattern(), optim.PatternOptimization)
        self.assertIsInstance(optim.ReluClipFusionPattern(), optim.PatternOptimization)

    def test_tensor_layout_algebra_patterns_are_selectable(self):
        names = (
            "ConcatReshape",
            "Reshape",
            "ReduceReshape",
            "Reshape2Of3",
            "ReshapeReshapeBinary",
            "ReshapeReshape",
            "ReshapeSqueeze",
            "ShapeBasedEditDistanceReshape",
            "ShapeBasedReshapeIsSqueeze",
            "ShapedBasedReshape",
            "StaticConcatReshape",
            "UnsqueezeOrSqueezeReshape",
            "UnsqueezeReshape",
            "MulUnsqueezeUnsqueeze",
            "SqueezeAdd",
            "SqueezeBinaryUnsqueeze",
            "SwapUnsqueezeTranspose",
            "TransposeEqualReshape",
            "TransposeReshapeTranspose",
            "MulMulMulScalar",
            "SwitchOrderBinary",
            "SwapRangeAddScalar",
            "ReduceArgTopK",
            "ReduceSumNormalize",
            "Sub1Mul",
            "SwapUnary",
            "SameChildren",
            "SameChildrenFromInput",
            "ShapeBasedIdentity",
            "ShapeBasedSameChildren",
            "ShapeBasedShapeShapeAdd",
            "GemmSumFusion",
            "GemmTranspose",
            "MatMulAdd",
            "MatMulBatchNormalizationFusion",
            "MatMulReshape2Of3",
            "MatMulScaleFusion",
            "MulMulMatMul",
            "ReshapeMatMulReshape",
            "ShapeBasedMatMulToMul",
            "SwitchReshapeActivation",
            "TransposeMatMul",
            "TransposeReshapeMatMul",
            "BatchNormalization",
            "BatchNormalizationTraining",
            "CastLayerNormalizationCast",
            "LayerNormalization",
            "LayerNormalizationScale",
            "RMSNormalization",
            "RMSNormalizationMul",
            "Gelu",
            "LeakyRelu",
            "MaxRelu",
            "SoftmaxCrossEntropyLossCast",
            "RotaryEmbedding",
            "RotaryConcatPart",
            "FunctionCausalMask",
            "FunctionCausalMaskMulAdd",
            "FunctionCosSinCache",
            "FunctionHalfRotaryEmbedding",
            "FunctionAttention",
            "FunctionAttentionGQA",
            "AttentionGQA",
        )
        registered = optim.standard_pattern_names()
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, registered)
                self.assertIsInstance(
                    optim.standard_patterns([name])[0], optim.PatternOptimization
                )
                self.assertIsInstance(
                    getattr(optim, f"{name}Pattern")(), optim.PatternOptimization
                )

    def test_python_pattern_runs_recursively_in_subgraph(self):
        then_branch = oh.make_graph(
            [
                oh.make_node("Neg", ["X"], ["middle"]),
                oh.make_node("Neg", ["middle"], ["then_out"]),
            ],
            "then_branch",
            [],
            [oh.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
        )
        else_branch = oh.make_graph(
            [oh.make_node("Identity", ["X"], ["else_out"])],
            "else_branch",
            [],
            [oh.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
        )
        graph = oh.make_graph(
            [
                oh.make_node(
                    "If", ["cond"], ["Y"], then_branch=then_branch, else_branch=else_branch
                )
            ],
            "main",
            [
                oh.make_tensor_value_info("cond", TensorProto.BOOL, []),
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])

        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, [NegNegPattern()])
        rewrites, report = graph.optimize(report=True)

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(rewrites[0].graph_path, ["then_branch"])
        self.assertIn("graph_path=then_branch", repr(rewrites[0]))
        self.assertIn("  graph_path: then_branch\n", rewrites[0].to_detailed_string())
        self.assertTrue(any(item.graph_path == ["then_branch"] for item in report.subgraphs))

    def test_render_rst_standard_patterns_table_lists_every_pattern(self):
        table = optim.render_rst_standard_patterns_table()

        self.assertIn(".. list-table::", table)
        self.assertIn(":header-rows: 1", table)
        self.assertIn("Class / registered name", table)
        self.assertIn("Candidate roots", table)
        self.assertIn("Transformation", table)
        for name in optim.standard_pattern_names():
            self.assertIn(f"``{name}``", table)

    def test_render_rst_standard_patterns_table_reflects_metadata(self):
        table = optim.render_rst_standard_patterns_table()

        for pattern in optim.standard_patterns():
            self.assertIn(f":class:`{type(pattern).__name__}`", table)
            for op in pattern.fast_op_type():
                self.assertIn(f"``{op}``", table)


if __name__ == "__main__":
    unittest.main(verbosity=2)
