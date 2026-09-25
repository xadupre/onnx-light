# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests Python graph-pattern optimization and extension points."""

from __future__ import annotations

import gc
import re
import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
from onnx_light.onnx_lib import parser
from onnx_light.onnx_core.shape_inference import Device

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

    def __init__(self, priority: int = 1, name: str = "NegNeg", device=Device.kUndefined):
        super().__init__(priority=priority, name=name, device=device)

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


class TestPatternSelection(ExtTestCase):
    def setUp(self):
        super().setUp()
        original = optim.registered_patterns()
        self.addCleanup(self.restore_patterns, original)
        optim.clear_registered_patterns()
        self.neutral = NegNegPattern(name="Neutral")
        self.cpu = NegNegPattern(name="CPUOnly", device=Device.kCPU)
        self.gpu = NegNegPattern(name="GPUOnly", device=Device.kGPU0)
        for pattern in (self.neutral, self.cpu, self.gpu):
            optim.register_pattern(pattern)
        self.builder = optim.GraphBuilder(_double_neg_model())

    def restore_patterns(self, original):
        """Restores the process-global registry after a selector test."""
        optim.clear_registered_patterns()
        for pattern in original:
            optim.register_pattern(pattern)

    def test_default_excludes_device_patterns_even_with_builder_device(self):
        for device in (Device.kUndefined, Device.kCPU, Device.kGPU0):
            with self.subTest(device=device):
                self.builder.device = device
                graph = optim.GraphGraph(self.builder)
                self.assertEqual(graph.patterns, [self.neutral])
                self.assertEqual(self.builder.device, device)

    def test_false_and_empty_list_disable_global_and_builder_patterns(self):
        self.builder.register_pattern(NegNegPattern(name="Local"))
        for selector in (False, []):
            with self.subTest(selector=selector):
                graph = optim.GraphGraph(self.builder, patterns=selector)
                self.assertEqual(graph.patterns, [])
                self.assertEqual(graph.optimize(), [])
                self.assertEqual(
                    [node.op_type for node in self.builder.build_graph().node], ["Neg", "Neg"]
                )

    def test_device_includes_generic_and_matching_patterns_and_sets_target(self):
        for device, specific in ((Device.kCPU, self.cpu), (Device.kGPU0, self.gpu)):
            with self.subTest(device=device):
                builder = optim.GraphBuilder(_double_neg_model())
                graph = optim.GraphGraph(builder, patterns=device)
                self.assertEqual(graph.patterns, [self.neutral, specific])
                self.assertEqual(builder.device, device)
                self.assertEqual(
                    optim.GraphGraph(builder, patterns=device).patterns, graph.patterns
                )

    def test_device_conflict_is_rejected_without_modifying_builder(self):
        for existing, requested in ((Device.kCPU, Device.kGPU0), (Device.kGPU0, Device.kCPU)):
            with self.subTest(existing=existing):
                self.builder.device = existing
                original = self.builder.build_graph().SerializeToString()
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    optim.GraphGraph(self.builder, patterns=requested)
                self.assertEqual(self.builder.device, existing)
                self.assertEqual(self.builder.build_graph().SerializeToString(), original)

    def test_device_is_visible_to_matcher(self):
        observed = []

        class CPUOnlyPattern(NegNegPattern):
            def match(self, graph, node):
                observed.append(graph.builder.device)
                return super().match(graph, node)

        optim.clear_registered_patterns()
        optim.register_pattern(CPUOnlyPattern(name="CPUOnly", device=Device.kCPU))
        graph = optim.GraphGraph(self.builder, patterns=Device.kCPU)
        self.assertEqual([rewrite.pattern_name for rewrite in graph.optimize()], ["CPUOnly"])
        self.assertTrue(observed)
        self.assertEqual(set(observed), {Device.kCPU})

    def test_device_selection_checks_subgraphs_before_changing_target(self):
        inherited = self.builder.make_subgraph("inherited")
        conflicting = self.builder.make_subgraph("conflicting")
        conflicting.device = Device.kGPU0
        with self.assertRaisesRegex(ValueError, "conflicts"):
            optim.GraphGraph(self.builder, patterns=Device.kCPU)
        self.assertEqual(self.builder.device, Device.kUndefined)
        self.assertEqual(inherited.device, Device.kUndefined)
        self.assertEqual(conflicting.device, Device.kGPU0)
        optim.GraphGraph(self.builder, patterns=Device.kGPU0)
        self.assertEqual(self.builder.device, Device.kGPU0)
        self.assertEqual(inherited.device, Device.kGPU0)
        self.assertEqual(conflicting.device, Device.kGPU0)

    def test_regex_uses_fullmatch_and_can_select_device_patterns(self):
        for selector in (r"(CPU|GPU)Only", re.compile(r"(cpu|gpu)only", re.IGNORECASE)):
            with self.subTest(selector=selector):
                graph = optim.GraphGraph(self.builder, patterns=selector)
                self.assertEqual(graph.patterns, [self.cpu, self.gpu])
                self.assertEqual(self.builder.device, Device.kUndefined)
        self.assertEqual(optim.GraphGraph(self.builder, patterns="Only").patterns, [])
        self.assertEqual(optim.GraphGraph(self.builder, patterns="CPUOnly").patterns, [self.cpu])
        self.assertEqual(
            optim.GraphGraph(self.builder, patterns=".*").patterns,
            [self.neutral, self.cpu, self.gpu],
        )

    def test_explicit_list_is_exclusive_and_resolves_local_names(self):
        local = NegNegPattern(name="Local")
        self.builder.register_pattern(local)
        graph = optim.GraphGraph(self.builder, patterns=["GPUOnly", "Local", self.cpu])
        self.assertEqual(graph.patterns, [self.gpu, local, self.cpu])
        self.assertEqual(self.builder.device, Device.kUndefined)
        self.assertEqual(
            optim.GraphGraph(self.builder, patterns=["CPUOnly"]).patterns, [self.cpu]
        )

    def test_builder_overrides_global_before_selection(self):
        replacement = NegNegPattern(name="Neutral", device=Device.kCPU)
        self.builder.register_pattern(replacement)
        self.assertEqual(optim.GraphGraph(self.builder).patterns, [])
        self.assertEqual(
            optim.GraphGraph(self.builder, patterns="Neutral").patterns, [replacement]
        )
        self.assertEqual(
            optim.GraphGraph(self.builder, patterns=Device.kCPU).patterns, [replacement, self.cpu]
        )

    def test_explicit_instances_override_names_and_retain_ownership(self):
        replacement = NegNegPattern(name="Neutral", priority=3)
        graph = optim.GraphGraph(self.builder, patterns=["Neutral", replacement])
        self.assertEqual(graph.patterns, [replacement])
        del replacement
        gc.collect()
        self.assertEqual([rewrite.pattern_name for rewrite in graph.optimize()], ["Neutral"])

    def test_iterables_and_unregistered_standard_names(self):
        graph = optim.GraphGraph(self.builder, patterns=(pattern for pattern in [self.cpu]))
        self.assertEqual(graph.patterns, [self.cpu])
        graph = optim.GraphGraph(self.builder, patterns=["Cast"])
        self.assertEqual([pattern.name for pattern in graph.patterns], ["Cast"])
        self.assertEqual(graph.patterns[0].device, Device.kUndefined)

    def test_invalid_selectors_fail_explicitly(self):
        for selector in (True, 0, 1, object(), [None], [False], [Device.kCPU], re.compile(b".*")):
            with self.subTest(selector=selector), self.assertRaises(TypeError):
                optim.GraphGraph(self.builder, patterns=selector)
        with self.assertRaisesRegex(ValueError, "concrete device"):
            optim.GraphGraph(self.builder, patterns=Device.kUndefined)
        with self.assertRaises(re.error):
            optim.GraphGraph(self.builder, patterns="[")
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            optim.GraphGraph(self.builder, patterns=["DoesNotExist"])
        self.assertEqual(self.builder.device, Device.kUndefined)


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

    def test_initializer_only_cleanup_reports_and_replays(self):
        for with_node in (False, True):
            with self.subTest(with_node=with_node):
                nodes = [oh.make_node("Neg", ["live"], ["y"])] if with_node else []
                output_name = "y" if with_node else "live"
                model = oh.make_model(
                    oh.make_graph(
                        nodes,
                        "initializer_cleanup",
                        [],
                        [oh.make_tensor_value_info(output_name, TensorProto.FLOAT, [2])],
                        [
                            oh.make_tensor("live", TensorProto.FLOAT, [2], [1, 2]),
                            oh.make_tensor("unused", TensorProto.FLOAT, [2], [3, 4]),
                        ],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                    ir_version=10,
                )
                builder = optim.GraphBuilder(model)
                graph = optim.GraphGraph(builder, patterns=False)
                rewrites, report = graph.optimize(report=True)
                optimized = builder.build_graph()

                self.assertEqual(
                    [rewrite.pattern_name for rewrite in rewrites], ["RemoveUnusedNodes"]
                )
                self.assertEqual(report.rewrites, 1)
                self.assertEqual(len(report.patterns), 0)
                self.assertEqual(len(optimized.node), len(nodes))
                self.assertEqual([value.name for value in optimized.initializer], ["live"])
                self.assertEqual(
                    optimized.initializer[0].SerializeToString(),
                    model.graph.initializer[0].SerializeToString(),
                )
                self.assertEqual(
                    optim.replay(model, rewrites).SerializeToString(),
                    optimized.SerializeToString(),
                )
                self.assertEqual(len(graph.optimize()), 0)

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

        graph = optim.GraphGraph(builder)
        rewrites = graph.optimize()

        self.assertEqual(builder.registered_pattern_names(), ("NegNeg",))
        self.assertEqual(rewrites[0].pattern_name, "NegNeg")

    def test_graph_pattern_registration(self):
        builder = optim.GraphBuilder(_double_neg_model())
        graph = optim.GraphGraph(builder, [NegNegPattern()])

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

        for device in (Device.kUndefined, Device.kCPU, Device.kGPU0):
            with self.subTest(device=device):
                observed = []

                class DeviceNegNegPattern(NegNegPattern):
                    def match(self, graph, node):
                        observed.append(graph.builder.device)
                        return super().match(graph, node)

                builder = optim.GraphBuilder(model)
                pattern = DeviceNegNegPattern(device=device)
                builder.register_pattern(pattern)
                graph = optim.GraphGraph(
                    builder, patterns=[pattern] if device == Device.kUndefined else device
                )
                rewrites, report = graph.optimize(report=True)

                self.assertEqual(set(observed), {device})
                self.assertEqual(len(rewrites), 1)
                self.assertEqual(rewrites[0].graph_path, ["then_branch"])
                self.assertIn("graph_path=then_branch", repr(rewrites[0]))
                self.assertIn("  graph_path: then_branch\n", rewrites[0].to_detailed_string())
                self.assertTrue(
                    any(item.graph_path == ["then_branch"] for item in report.subgraphs)
                )

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
