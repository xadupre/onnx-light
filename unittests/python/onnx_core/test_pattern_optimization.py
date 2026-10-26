# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests Python graph-pattern optimization and extension points."""

from __future__ import annotations

import gc
import re
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light import onnx
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import ModelProto, TensorProto
from onnx_light.onnx.external_data_helper import load_external_data_for_model
from onnx_light.onnx_lib import parser
from onnx_light.onnx_op import GetAllOnnxOpSchemasWithHistory
from onnx_light.onnx_py import _onnxpycore, _onnxpypatterns
from onnx_light.onnx_core.graph_builder import ConstantFoldingOptions, OptimizationMode, ProtoKind
from onnx_light.onnx_core.shape_inference import Device

optim = import_or_skip("onnx_light.onnx_core.optimization")
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


def _double_neg_model():
    """Builds a model containing two consecutive Neg nodes."""
    return parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 18]>\n'
        "agraph (float[2] x) => (float[2] y) {\n"
        "  middle = Neg(x)\n"
        "  y = Neg(middle)\n"
        "}\n"
    )


def _transpose_weight_model():
    """Builds a model with a deferred constant transpose."""
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Transpose", ["weight"], ["weight_identity"], perm=[0, 1]),
                oh.make_node("Transpose", ["weight_identity"], ["transposed"], perm=[1, 0]),
                oh.make_node("Add", ["x", "transposed"], ["y"]),
            ],
            "transpose_weight",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [3, 2])],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
            initializer=[
                onh.from_array(numpy.arange(6, dtype=numpy.float32).reshape(2, 3), "weight")
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
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
    def test_standard_pattern_sets_overlap_without_changing_defaults(self):
        pattern_sets = optim.standard_pattern_sets()
        all_names = set(pattern_sets["all"])

        self.assertEqual(
            pattern_sets,
            {
                name: tuple(names)
                for name, names in _onnxpypatterns.registered_pattern_sets().items()
            },
        )
        self.assertEqual(set(pattern_sets), {"all", "L1", "L2", "Layer"})
        self.assertEqual(
            optim.standard_pattern_set_names(), _onnxpypatterns.registered_pattern_set_names()
        )
        self.assertEqual(set(optim.standard_pattern_set_names()), set(pattern_sets))
        self.assertEqual(
            all_names,
            set(pattern_sets["L1"]) | set(pattern_sets["L2"]) | set(pattern_sets["Layer"]),
        )
        self.assertTrue(set(pattern_sets["L1"]).isdisjoint(pattern_sets["L2"]))
        self.assertEqual(pattern_sets["Layer"], ("RepeatedLayer",))
        self.assertTrue(
            {"ConvAddFusion", "MatMulAdd", "LayerNormalization"}.issubset(pattern_sets["L1"])
        )
        self.assertNotIn("RepeatedLayer", optim.standard_pattern_names())
        self.assertEqual(
            [pattern.name for pattern in optim.standard_patterns(pattern_sets["Layer"])],
            ["RepeatedLayer"],
        )

    def test_pattern_sets_can_be_extended_and_created(self):
        custom = NegNegPattern(name="CustomNegNeg")
        optim.register_pattern(custom)
        self.addCleanup(optim.unregister_pattern, "CustomNegNeg")

        optim.register_pattern_set("custom", ["CustomNegNeg"])
        self.addCleanup(optim.unregister_pattern_set, "custom")
        optim.extend_pattern_set("custom", ["GatherToSlice", "CustomNegNeg"])

        pattern_sets = optim.standard_pattern_sets()
        self.assertIn("custom", optim.standard_pattern_set_names())
        self.assertEqual(pattern_sets["custom"], ("CustomNegNeg", "GatherToSlice"))
        self.assertIn("CustomNegNeg", pattern_sets["all"])
        self.assertIn("GatherToSlice", pattern_sets["L2"])
        self.assertIs(optim.standard_patterns(pattern_sets["custom"])[0], custom)
        self.assertTrue(optim.unregister_pattern_set("custom"))

    def test_pattern_listing_defaults_and_explicit_opt_in(self):
        optim.reset_registered_patterns()
        active_names = optim.registered_pattern_names()
        default_names = optim.standard_pattern_names()
        all_names = optim.standard_pattern_names(include_opt_in=True)
        for list_names in (
            _onnxpypatterns.registered_pattern_names,
            _onnxpycore.builder.registered_pattern_names,
            optim.standard_pattern_names,
        ):
            with self.subTest(list_names=list_names):
                self.assertEqual(list_names(), default_names)
                self.assertEqual(list_names(include_opt_in=False), default_names)
                self.assertIn("Cast", default_names)
                self.assertNotIn("RepeatedLayer", default_names)
                self.assertEqual(list_names(include_opt_in=True), all_names)
                self.assertIn("RepeatedLayer", all_names)
                self.assertEqual(
                    [name for name in all_names if name in default_names], default_names
                )
                self.assertEqual(list_names(), default_names)
                self.assertEqual(optim.registered_pattern_names(), active_names)
                self.assertNotIn(
                    "RepeatedLayer", [pattern.name for pattern in optim.standard_patterns()]
                )
                for graph_type in (optim.GraphGraph, _onnxpycore.builder.GraphGraph):
                    graph = graph_type(self.builder)
                    self.assertNotIn(
                        "RepeatedLayer", [pattern.name for pattern in graph.patterns]
                    )
                self.assertIsInstance(
                    optim.standard_patterns(["RepeatedLayer"])[0], optim.RepeatedLayerPattern
                )

    def test_repeated_layer_is_explicit_only(self):
        optim.reset_registered_patterns()
        self.assertNotIn("RepeatedLayer", optim.standard_pattern_names())
        self.assertNotIn("RepeatedLayer", optim.registered_pattern_names())
        self.assertIn("RepeatedLayer", optim.standard_pattern_names(include_opt_in=True))
        self.assertNotIn("RepeatedLayer", [pattern.name for pattern in optim.standard_patterns()])
        self.assertIsInstance(
            optim.standard_patterns(["RepeatedLayer"])[0], optim.RepeatedLayerPattern
        )

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
                self.assertEqual(len(graph.optimize().rewrites), 0)
                self.assertEqual(
                    [node.op_type for node in self.builder.build_graph().node], ["Neg", "Neg"]
                )

    def test_remove_unused_option_preserves_dead_end(self):
        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[2] x) => (float[2] y) {\n"
            "  dead = Add(x, x)\n"
            "  y = Neg(x)\n"
            "}\n"
        )
        for remove_unused in (False, True):
            with self.subTest(remove_unused=remove_unused):
                builder = optim.GraphBuilder(model)
                self.assertEqual(len(builder.to_model().graph.node), 2)
                rewrites = optim.GraphGraph(builder, patterns=False).optimize(
                    remove_unused=remove_unused
                )
                self.assertEqual(len(builder.to_model().graph.node), 1 if remove_unused else 2)
                self.assertEqual(len(rewrites.rewrites), 1 if remove_unused else 0)

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
        self.assertEqual(
            [rewrite.pattern_name for rewrite in graph.optimize().rewrites], ["CPUOnly"]
        )
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
        self.assertEqual(
            [rewrite.pattern_name for rewrite in graph.optimize().rewrites], ["Neutral"]
        )

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
    def test_constant_identity_folding_shares_raw_storage_and_honors_limit(self):
        values = numpy.array([1, 2], dtype=numpy.float32)
        builder = optim.GraphBuilder("sharing", opsets=18)
        builder.make_initializer(onh.from_array(values, "weight"))
        builder.make_node("Identity", ["weight"], ["Y"])
        builder.make_output("Y")
        canonical = builder.to_model()
        pointer = numpy.from_dlpack(canonical.graph.initializer[0]).ctypes.data
        options = ConstantFoldingOptions()
        options.max_element_count = 1
        self.assertEqual(builder.constant_fold(options), 0)
        self.assertEqual(len(builder.to_model().graph.node), 1)
        options.max_element_count = -1
        self.assertEqual(builder.constant_fold(options), 1)
        folded = next(t for t in builder.to_model().graph.initializer if t.name == "Y")
        actual = numpy.from_dlpack(folded)
        self.assertEqual(actual.ctypes.data, pointer)
        self.assertEqualArray(actual, values)

    def test_python_pattern_explicitly_supports_local_function_calls(self):
        class LocalNegNegPattern(NegNegPattern):
            def supports_local_functions(self, node):
                return node.domain == "local"

        model = _double_neg_model()
        model.opset_import.append(oh.make_opsetid("local", 1))
        model.functions.append(
            oh.make_function(
                "local",
                "Neg",
                ["X"],
                ["Y"],
                [oh.make_node("Neg", ["X"], ["Y"])],
                [oh.make_opsetid("", 18)],
            )
        )
        for node in model.graph.node:
            node.domain = "local"

        for pattern, expected in (
            (NegNegPattern(), ["Neg", "Neg"]),
            (LocalNegNegPattern(), ["Identity"]),
        ):
            with self.subTest(pattern=type(pattern).__name__):
                builder = optim.GraphBuilder(model)
                optim.GraphGraph(builder, [pattern]).optimize()
                self.assertEqual(
                    [node.op_type for node in builder.to_onnx().graph.node], expected
                )

        model.functions[0].domain = ""
        model.functions[0].node[0].op_type = "Identity"
        for node in model.graph.node:
            node.domain = ""
        builder = optim.GraphBuilder(model)
        optim.GraphGraph(builder, [LocalNegNegPattern()]).optimize()
        self.assertEqual([node.op_type for node in builder.to_onnx().graph.node], ["Neg", "Neg"])

    def test_stale_batched_match_is_revalidated_before_apply(self):
        class OneShotNegNegPattern(NegNegPattern):
            def __init__(self):
                super().__init__()
                self.active = True

            def match(self, graph, node):
                if not self.active:
                    return self.no_match(node, "the pattern has already been applied")
                return super().match(graph, node)

            def apply(self, graph, nodes):
                assert self.active
                self.active = False
                return super().apply(graph, nodes)

        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18]>\n'
            "agraph (float[2] x, float[2] z) => (float[2] y1, float[2] y2) {\n"
            "  middle1 = Neg(x)\n"
            "  y1 = Neg(middle1)\n"
            "  middle2 = Neg(z)\n"
            "  y2 = Neg(middle2)\n"
            "}\n"
        )
        builder = optim.GraphBuilder(model)

        rewrites = optim.GraphGraph(builder, [OneShotNegNegPattern()]).optimize()

        self.assertEqual([rewrite.pattern_name for rewrite in rewrites.rewrites], ["NegNeg"])
        self.assertEqual(
            [node.op_type for node in builder.to_onnx().graph.node], ["Identity", "Neg", "Neg"]
        )

    def test_python_pattern_rewrites_and_reports_rejections(self):
        model = _double_neg_model()
        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, [NegNegPattern()])
        rewrites, report = graph.optimize(report=True)
        optimized = builder.to_onnx()

        self.assertEqual([node.op_type for node in optimized.graph.node], ["Identity"])
        self.assertIsInstance(rewrites, optim.RewritingProto)
        self.assertEqual(len(rewrites.rewrites), 1)
        rewrite = rewrites.rewrites[0]
        self.assertIsInstance(rewrite, optim.LocalRewritingProto)
        self.assertEqual(rewrite.pattern_name, "NegNeg")
        self.assertEqual(list(rewrite.matched_nodes), [0, 1])
        self.assertEqual(list(rewrite.added_nodes_positions), [0])
        self.assertFalse(hasattr(rewrite, "iteration"))
        self.assertFalse(hasattr(rewrite, "match_time_ns"))
        parsed = optim.RewritingProto()
        parsed.ParseFromString(rewrites.SerializeToString())
        self.assertEqual(parsed.SerializeToString(), rewrites.SerializeToString())
        self.assertEqual(report.rewrites, 1)
        pattern_report = next(item for item in report.patterns if item.pattern_name == "NegNeg")
        self.assertEqual(pattern_report.matches, 1)
        self.assertGreaterEqual(sum(item.occurrences for item in pattern_report.no_matches), 1)
        self.assertIn(
            "the input is not produced by Neg",
            {item.reason for item in pattern_report.no_matches},
        )

    def test_parallel_request_keeps_python_pattern_callbacks_sequential(self):
        code = textwrap.dedent("""
            import threading
            import unittest
            from onnx_light.onnx import TensorProto
            from onnx_light.onnx_core.optimization import (
                GraphBuilder,
                GraphGraph,
                PatternOptimization,
                TransposeTransposePattern,
            )

            callback_threads = []
            checks = unittest.TestCase()

            class ThreadTrackingPattern(PatternOptimization):
                def __init__(self):
                    super().__init__(priority=0, name="ThreadTracking")

                def fast_op_type(self):
                    return {"Add"}

                def match(self, graph, node):
                    del graph
                    callback_threads.append(threading.get_ident())
                    return self.no_match(node, "thread check")

            for mixed in (False, True):
                callback_threads.clear()
                builder = GraphBuilder("parallel_python_callbacks", opsets=18)
                for index in range(64):
                    input_name = f"x{index}"
                    builder.make_input(input_name, TensorProto.FLOAT, [2, 3])
                    output = f"y{index}"
                    builder.make_node("Add", [input_name, input_name], outputs=[output])
                    if mixed:
                        for prefix in ("middle", "transposed"):
                            transposed = f"{prefix}{index}"
                            builder.make_node(
                                "Transpose",
                                [output],
                                outputs=[transposed],
                                attributes={"perm": [1, 0]},
                            )
                            output = transposed
                    builder.make_output(output)
                patterns = [ThreadTrackingPattern()]
                if mixed:
                    patterns.append(TransposeTransposePattern())
                graph = GraphGraph(builder, patterns)
                calling_thread = threading.get_ident()
                plan, report = graph.optimize(max_iter=1, report=True, parallel=True)
                checks.assertEqual(len(callback_threads), 64)
                checks.assertEqual(set(callback_threads), {calling_thread})
                statistics = {item.pattern_name: item for item in report.patterns}
                checks.assertEqual(statistics["ThreadTracking"].attempts, 64)
                if mixed:
                    checks.assertEqual(statistics["TransposeTranspose"].attempts, 128)
                    checks.assertEqual(statistics["TransposeTranspose"].matches, 64)
                    checks.assertEqual(len(plan.rewrites), 64)
        """)
        for options in ([], ["-O"]):
            with self.subTest(options=options):
                result = subprocess.run(
                    [sys.executable, *options, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_parallel_native_matching_preserves_optimization_callback_order(self):
        code = textwrap.dedent("""
            import threading
            import unittest
            from onnx_light.onnx import TensorProto
            from onnx_light.onnx_core.optimization import (
                GraphBuilder,
                GraphGraph,
                TransposeTransposePattern,
            )

            checks = unittest.TestCase()
            for parallel in (False, True):
                builder = GraphBuilder("parallel_native_callbacks", opsets=18)
                for index in range(32):
                    input_name = f"x{index}"
                    builder.make_input(input_name, TensorProto.FLOAT, [2, 3])
                    builder.make_node(
                        "Transpose",
                        [input_name],
                        outputs=[f"middle{index}"],
                        attributes={"perm": [1, 0]},
                    )
                    builder.make_node(
                        "Transpose",
                        [f"middle{index}"],
                        outputs=[f"y{index}"],
                        attributes={"perm": [1, 0]},
                    )
                    builder.make_output(f"y{index}")

                calling_thread = threading.get_ident()
                observed = []

                def callback(graph, match):
                    checks.assertEqual(threading.get_ident(), calling_thread)
                    position = graph.position(match.nodes[0])
                    observed.append(position)
                    return position % 4 == 0

                graph = GraphGraph(builder, [TransposeTransposePattern()])
                plan, report = graph.optimize(
                    max_iter=1, report=True, callback=callback, parallel=parallel
                )
                checks.assertEqual(observed, list(range(0, 64, 2)))
                # Rewrite indices refer to the graph after preceding edits.
                matched_nodes = [list(r.matched_nodes) for r in plan.rewrites]
                checks.assertEqual(matched_nodes, [
                    [index, index + 1] for index in range(0, 48, 3)
                ])
                checks.assertEqual(report.patterns[0].matches, 16)
        """)
        for options in ([], ["-O"]):
            with self.subTest(options=options):
                result = subprocess.run(
                    [sys.executable, *options, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_replay_reproduces_python_rewrite(self):
        model = _double_neg_model()
        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, [NegNegPattern()])
        rewrites = graph.optimize()
        optimized = builder.build_graph()
        replayed = optim.apply_rewritings(model, rewrites)

        self.assertEqual(replayed.SerializeToString(), optimized.SerializeToString())

    def test_named_variants_model_map_and_explicit_selection(self):
        value = oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        model = oh.make_model(oh.make_graph([], "canonical", [value], [value]))
        model.add_rewriting("first", optim.RewritingProto())
        second = optim.RewritingProto()
        second.metadata_props.add(key="target", value="second")
        model.rewritings["second"] = second
        loaded = ModelProto()
        loaded.ParseFromString(model.SerializeToString())
        self.assertEqual(set(loaded.rewritings), {"first", "second"})
        self.assertEqual(loaded.rewritings["second"].metadata_props[0].value, "second")
        self.assertEqual(loaded.graph.SerializeToString(), model.graph.SerializeToString())
        constructed = ModelProto(rewritings={"second": second})
        self.assertEqual(
            constructed.rewritings["second"].SerializeToString(), second.SerializeToString()
        )
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            loaded.add_rewriting("first", optim.RewritingProto())
        with self.assertRaisesRegex(RuntimeError, "empty"):
            loaded.add_rewriting("", optim.RewritingProto())
        builder = optim.GraphBuilder(loaded)
        self.assertEqual(set(builder.to_model().rewritings), {"first", "second"})
        selected = builder.apply_rewritings("second").to_model()
        self.assertEqual(len(selected.rewritings), 0)
        self.assertEqual(len(selected.graph.node), 0)
        self.assertEqual(len(builder.apply_rewritings(name="first").to_model().rewritings), 0)
        with self.assertRaisesRegex(ValueError, "unknown variant"):
            builder.apply_rewritings("missing")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            builder.add_rewriting("first", optim.RewritingProto())
        builder.make_input(oh.make_tensor_value_info("unused", TensorProto.FLOAT, [2]))
        self.assertEqual(len(builder.to_model().rewritings), 0)
        retained = loaded.rewritings["second"]
        loaded.clear_rewritings()
        self.assertFalse(loaded.HasField("rewritings"))
        self.assertEqual(retained.metadata_props[0].value, "second")

    def test_to_onnx_optimization_exports_keep_the_canonical_builder(self):
        builder = optim.GraphBuilder(_double_neg_model())
        builder.register_pattern(NegNegPattern())
        builder.add_rewriting("other", optim.RewritingProto())
        canonical = builder.to_onnx()
        before = canonical.SerializeToString()
        optimized = builder.to_onnx(optimize=OptimizationMode.kOptimize)
        bundled = builder.to_onnx(optimize=OptimizationMode.kRewritings)
        self.assertEqual(
            builder.to_onnx(optimize=OptimizationMode.kNone).SerializeToString(), before
        )
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Identity"])
        self.assertFalse(optimized.rewritings)
        self.assertEqual(bundled.graph.SerializeToString(), canonical.graph.SerializeToString())
        self.assertEqual(set(bundled.rewritings), {"other", "optimized"})
        self.assertEqual(bundled.rewritings["optimized"].rewrites[0].pattern_name, "NegNeg")
        loaded = ModelProto()
        loaded.ParseFromString(bundled.SerializeToString())
        replayed = optim.GraphBuilder(loaded).apply_rewritings("optimized").to_standard_model()
        self.assertEqual(replayed.SerializeToString(), optimized.SerializeToString())
        self.assertEqual(builder.to_onnx().SerializeToString(), before)
        self.assertEqual(builder.registered_pattern_names(), ("NegNeg",))

    def test_to_onnx_bundle_keeps_positions_after_identity_cleanup_and_shape_hoisting(self):
        for op_type in ("Shape", "Size"):
            with self.subTest(op_type=op_type):
                model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node("Relu", ["x"], ["p"]),
                            oh.make_node("Cast", ["z"], ["q"], to=TensorProto.FLOAT),
                            oh.make_node("Identity", ["p"], ["a"]),
                            oh.make_node(op_type, ["a"], ["s"]),
                        ],
                        "shape_order",
                        [
                            oh.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2]),
                            oh.make_tensor_value_info("z", TensorProto.FLOAT, ["N", 2]),
                        ],
                        [
                            oh.make_tensor_value_info("q", TensorProto.FLOAT, ["N", 2]),
                            oh.make_tensor_value_info(
                                "s", TensorProto.INT64, [2] if op_type == "Shape" else []
                            ),
                        ],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                )
                builder = optim.GraphBuilder(model)
                builder.register_pattern(optim.CastPattern())
                optimized = builder.to_onnx(optimize=OptimizationMode.kOptimize)
                bundled = builder.to_onnx(optimize=OptimizationMode.kRewritings)
                self.assertEqual(
                    [node.op_type for node in bundled.graph.node], ["Relu", op_type, "Cast"]
                )
                self.assertEqual(
                    list(bundled.rewritings["optimized"].rewrites[0].matched_nodes), [2]
                )
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "bundle.onnx"
                    onnx.save(bundled, path)
                    loaded = onnx.load(path)
                replayed = (
                    optim.GraphBuilder(loaded).apply_rewritings("optimized").to_standard_model()
                )
                self.assertEqual(
                    replayed.graph.SerializeToString(), optimized.graph.SerializeToString()
                )
                feeds = {
                    "x": numpy.array([[-1, 2], [3, -4]], dtype=numpy.float32),
                    "z": numpy.ones((2, 2), dtype=numpy.float32),
                }
                for actual, expected in zip(
                    ReferenceEvaluator(replayed).run(None, feeds),
                    ReferenceEvaluator(model).run(None, feeds),
                ):
                    self.assertEqualArray(actual, expected)

    def test_to_onnx_drops_old_plans_when_function_canonicalization_changes_reference(self):
        model = _double_neg_model()
        model.functions.append(
            oh.make_function(
                "local",
                "ForwardedNeg",
                ["arg"],
                ["result"],
                [
                    oh.make_node("Identity", ["arg"], ["forwarded"]),
                    oh.make_node("Neg", ["forwarded"], ["result"]),
                ],
                opset_imports=[oh.make_opsetid("", 18)],
            )
        )
        model.opset_import.append(oh.make_opsetid("local", 1))
        builder = optim.GraphBuilder(model)
        builder.register_pattern(NegNegPattern())
        builder.add_rewriting("other", optim.RewritingProto())
        source = builder.to_onnx()
        bundled = builder.to_onnx(optimize=OptimizationMode.kRewritings)
        self.assertEqual(bundled.graph.SerializeToString(), source.graph.SerializeToString())
        self.assertEqual(set(bundled.rewritings), {"optimized"})
        self.assertEqual([node.op_type for node in bundled.functions[0].node], ["Neg"])
        self.assertEqual(set(builder.to_onnx().rewritings), {"other"})

    def test_to_onnx_exports_relocate_external_weights_and_replay_recipes(self):
        model = _transpose_weight_model()
        feeds = {"x": numpy.ones((3, 2), dtype=numpy.float32)}
        expected = ReferenceEvaluator(model).run(None, feeds)[0]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            target = Path(directory) / "target"
            source.mkdir()
            target.mkdir()
            onnx.save(model, source / "model.onnx", save_as_external_data=True, size_threshold=0)
            builder = optim.GraphBuilder(
                onnx.load(source / "model.onnx", load_external_data=True)
            )
            builder.register_pattern(optim.TransposeTransposePattern(priority=-1))
            before = builder.to_onnx().SerializeToString()
            optimized = builder.to_onnx(optimize=OptimizationMode.kOptimize)
            bundled = builder.to_onnx(optimize=OptimizationMode.kRewritings)
            edits = bundled.rewritings["optimized"].rewrites
            self.assertTrue(any(edit.initializer_calls for edit in edits))
            self.assertFalse(any(edit.added_initializers for edit in edits))
            self.assertTrue(
                any(
                    node.op_type == "Transpose"
                    for function in bundled.rewritings["optimized"].added_functions
                    for node in function.node
                )
            )
            for name, exported in (("plain", optimized), ("bundle", bundled)):
                path = target / f"{name}.onnx"
                onnx.save(exported, path, save_as_external_data=True, size_threshold=0)
                self.assertEqual(builder.to_onnx().SerializeToString(), before)
            (source / "model.onnx").unlink()
            (source / "model.onnx.data").unlink()
            plain = onnx.load(target / "plain.onnx", load_external_data=True)
            restored = onnx.load(target / "bundle.onnx", load_external_data=True)
            self.assertEqual(list(restored.rewritings), ["optimized"])
            replayed = (
                optim.GraphBuilder(restored).apply_rewritings("optimized").to_standard_model()
            )
            for exported in (plain, restored, replayed):
                numpy.testing.assert_array_equal(
                    ReferenceEvaluator(exported).run(None, feeds)[0], expected
                )
            self.assertEqual(
                [node.op_type for node in replayed.graph.node],
                [node.op_type for node in plain.graph.node],
            )
            self.assertEqual(
                onh.to_array(replayed.graph.initializer[0]).tolist(),
                onh.to_array(plain.graph.initializer[0]).tolist(),
            )

    def test_rewritten_weights_are_new_but_canonical_payloads_are_shared(self):
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                values = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
                model = _transpose_weight_model()
                if borrowed:
                    model.graph.initializer[0]._set_raw_data_from_buffer(values)
                original = numpy.from_dlpack(model.graph.initializer[0]).ctypes.data
                builder = optim.GraphBuilder(model)
                builder.register_pattern(optim.TransposeTransposePattern(priority=-1))
                optimized = builder.to_onnx(optimize=OptimizationMode.kOptimize)
                bundled = builder.to_onnx(optimize=OptimizationMode.kRewritings)
                self.assertEqual(
                    numpy.from_dlpack(bundled.graph.initializer[0]).ctypes.data, original
                )
                self.assertNotEqual(
                    numpy.from_dlpack(optimized.graph.initializer[0]).ctypes.data, original
                )
                numpy.testing.assert_array_equal(
                    onh.to_array(optimized.graph.initializer[0]), values.T
                )
                replayed = (
                    optim.GraphBuilder(bundled).apply_rewritings("optimized").to_standard_model()
                )
                numpy.testing.assert_array_equal(
                    onh.to_array(replayed.graph.initializer[0]), values.T
                )
                optimized.graph.initializer[0].raw_data = bytes(values.nbytes)
                numpy.testing.assert_array_equal(
                    onh.to_array(bundled.graph.initializer[0]), values
                )
                numpy.testing.assert_array_equal(onh.to_array(model.graph.initializer[0]), values)
                model.graph.initializer[0].raw_data = bytes(values.nbytes)
                numpy.testing.assert_array_equal(
                    onh.to_array(bundled.graph.initializer[0]), values
                )
                del model, builder
                gc.collect()
                numpy.testing.assert_array_equal(
                    onh.to_array(bundled.graph.initializer[0]), values
                )

    def test_to_onnx_optimization_options_validate_before_matching(self):
        builder = optim.GraphBuilder(_double_neg_model())
        before = builder.to_onnx().SerializeToString()
        for option in (None, False, True, 0, 1, "rewritings", "unknown", ProtoKind.kModel):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(TypeError, "OptimizationMode"),
            ):
                builder.to_onnx(optimize=option)
            self.assertEqual(builder.to_onnx().SerializeToString(), before)
        for kind in (ProtoKind.kGraph, ProtoKind.kFunction):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "kModel"):
                builder.to_onnx(kind, optimize=OptimizationMode.kRewritings)
        builder.add_rewriting("optimized", optim.RewritingProto())
        with self.assertRaisesRegex(ValueError, "already exists"):
            builder.to_onnx(optimize=OptimizationMode.kRewritings)
        builder_model = builder.to_onnx()
        builder_model.clear_rewritings()
        self.assertEqual(builder_model.SerializeToString(), before)

    def test_to_onnx_bundle_externalizes_replacement_tensor_payloads(self):
        class LiteralNegNegPattern(NegNegPattern):
            def apply(self, graph, nodes):
                return [
                    oh.make_node(
                        "Constant",
                        [],
                        ["zero"],
                        value=onh.from_array(numpy.zeros(2, dtype=numpy.float32)),
                    ),
                    oh.make_node("Add", [nodes[0].input[0], "zero"], list(nodes[1].output)),
                ]

        builder = optim.GraphBuilder(_double_neg_model())
        builder.register_pattern(LiteralNegNegPattern())
        bundled = builder.to_onnx(optimize=OptimizationMode.kRewritings)
        self.assertFalse(bundled.graph.initializer)
        values = numpy.array([4, -5], dtype=numpy.float32)
        default_tensor = onh.from_array(values, "default_weight")
        default_graph = oh.make_graph(
            [],
            "default_graph",
            [],
            [oh.make_tensor_value_info("default_weight", TensorProto.FLOAT, [2])],
            initializer=[default_tensor],
        )
        plan = bundled.rewritings["optimized"]
        plan.added_functions.append(
            oh.make_function(
                "local",
                "Defaults",
                [],
                ["default_weight"],
                [oh.make_node("Constant", [], ["default_weight"], value=default_tensor)],
                opset_imports=[oh.make_opsetid("", 18)],
                attribute_protos=[
                    oh.make_attribute("tensor", default_tensor),
                    oh.make_attribute("tensors", [default_tensor, default_tensor]),
                    oh.make_attribute("graphs", [default_graph, default_graph]),
                ],
            )
        )
        bundled.rewritings["optimized"] = plan
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            target = Path(directory) / "target"
            source.mkdir()
            target.mkdir()
            path = source / "bundle.onnx"
            onnx.save(
                bundled,
                path,
                save_as_external_data=True,
                size_threshold=0,
                convert_attribute=True,
            )
            self.assertGreater((source / "bundle.onnx.data").stat().st_size, 0)
            deferred = onnx.load(path, load_external_data=False)
            load_external_data_for_model(deferred, str(source))
            onnx.save(
                deferred, target / "deferred.onnx", save_as_external_data=True, size_threshold=0
            )
            onnx.align_external_data_streaming(
                str(path),
                str(target / "aligned.onnx"),
                str(target / "aligned.data"),
                alignment=64,
                chunk_size=3,
            )
            path.unlink()
            (source / "bundle.onnx.data").unlink()
            for name in ("deferred.onnx", "aligned.onnx"):
                with self.subTest(name=name):
                    restored = onnx.load(target / name, load_external_data=True)
                    attributes = (
                        restored.rewritings["optimized"].added_functions[-1].attribute_proto
                    )
                    tensors = [
                        attributes[0].t,
                        *attributes[1].tensors,
                        *(graph.initializer[0] for graph in attributes[2].graphs),
                    ]
                    for tensor in tensors:
                        numpy.testing.assert_array_equal(onh.to_array(tensor), values)
                    replayed = (
                        optim.GraphBuilder(restored).apply_rewritings("optimized").to_model()
                    )
                    x = numpy.array([2, -3], dtype=numpy.float32)
                    numpy.testing.assert_array_equal(
                        ReferenceEvaluator(replayed).run(None, {"x": x})[0], x
                    )

    def test_to_onnx_optimization_preserves_nested_device_targets(self):
        builder = optim.GraphBuilder("root", opsets=18)
        builder.make_input(oh.make_tensor_value_info("condition", TensorProto.BOOL, []))
        builder.make_input(oh.make_tensor_value_info("x", TensorProto.FLOAT, [2]))
        observed = []

        class CPUOnlyNegNegPattern(NegNegPattern):
            def match(self, graph, node):
                observed.append(graph.builder.device)
                if graph.builder.device != Device.kCPU:
                    return self.no_match(node, "requires a CPU target")
                return super().match(graph, node)

        builder.register_pattern(CPUOnlyNegNegPattern())
        for name, device in (("gpu", Device.kGPU0), ("cpu", Device.kCPU)):
            body = builder.make_subgraph(name)
            body.device = device
            body.make_node("Neg", ["x"], ["middle"])
            body.make_node("Neg", ["middle"], ["result"])
            body.make_output("result")
        builder.make_node(
            "If",
            ["condition"],
            ["y"],
            attributes={"then_branch_ref": "gpu", "else_branch_ref": "cpu"},
        )
        builder.make_output("y")
        before = builder.to_onnx().SerializeToString()
        for option in (OptimizationMode.kOptimize, OptimizationMode.kRewritings):
            observed.clear()
            exported = builder.to_onnx(optimize=option)
            self.assertIn(Device.kGPU0, observed)
            self.assertIn(Device.kCPU, observed)
            if option is OptimizationMode.kRewritings:
                self.assertTrue(exported.rewritings["optimized"].rewrites)
                self.assertTrue(
                    all(
                        edit.graph_path == ["cpu"]
                        for edit in exported.rewritings["optimized"].rewrites
                    )
                )
            self.assertEqual(builder.device, Device.kUndefined)
            self.assertEqual(builder.to_onnx().SerializeToString(), before)

    def test_to_onnx_optimized_function_preserves_constant_values(self):
        values = numpy.array([1, 2], dtype=numpy.float32)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Constant", [], ["weight"], value=onh.from_array(values)),
                    oh.make_node("Add", ["x", "weight"], ["y"]),
                ],
                "F",
                [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        builder = optim.GraphBuilder(model)
        before = builder.to_onnx().SerializeToString()
        for mode in (OptimizationMode.kNone, OptimizationMode.kOptimize):
            with self.subTest(mode=mode):
                function = builder.to_onnx(ProtoKind.kFunction, domain="local", optimize=mode)
                self.assertEqual([node.op_type for node in function.node], ["Constant", "Add"])
                numpy.testing.assert_array_equal(
                    onh.to_array(function.node[0].attribute[0].t), values
                )
                called = oh.make_model(
                    oh.make_graph(
                        [oh.make_node("F", ["x"], ["y"], domain="local")],
                        "call",
                        list(model.graph.input),
                        list(model.graph.output),
                    ),
                    functions=[function],
                    opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("local", 1)],
                )
                x = numpy.array([3, -4], dtype=numpy.float32)
                numpy.testing.assert_array_equal(
                    ReferenceEvaluator(called).run(None, {"x": x})[0], x + values
                )
                self.assertEqual(builder.to_onnx().SerializeToString(), before)

    def test_to_onnx_no_op_optimization_and_other_proto_kinds(self):
        value = oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        model = oh.make_model(oh.make_graph([], "no_op", [value], [value]))
        model.add_rewriting("other", optim.RewritingProto())
        builder = optim.GraphBuilder(model)
        before = builder.to_onnx().SerializeToString()
        plain = builder.to_onnx(optimize=OptimizationMode.kOptimize)
        self.assertFalse(plain.rewritings)
        bundled = builder.to_onnx(optimize=OptimizationMode.kRewritings)
        self.assertFalse(bundled.rewritings["optimized"].rewrites)
        self.assertFalse(
            optim.GraphBuilder(bundled).apply_rewritings("optimized").to_model().rewritings
        )
        self.assertEqual(
            len(builder.to_onnx(ProtoKind.kGraph, optimize=OptimizationMode.kOptimize).node), 0
        )
        function = builder.to_onnx(
            ProtoKind.kFunction, domain="local", optimize=OptimizationMode.kOptimize
        )
        self.assertEqual(function.domain, "local")
        self.assertEqual(builder.to_onnx().SerializeToString(), before)

    def test_to_onnx_optimization_preserves_schema_lookup_and_propagates_errors(self):
        schema_calls = []

        def schema_lookup(op_type):
            schema_calls.append(op_type)
            return GetAllOnnxOpSchemasWithHistory(op_type, False)

        builder = optim.GraphBuilder(_double_neg_model(), schema_lookup=schema_lookup)
        builder.register_pattern(NegNegPattern())
        schema_calls.clear()
        builder.to_onnx(optimize=OptimizationMode.kRewritings)
        self.assertIn("Neg", schema_calls)

        class FailingPattern(NegNegPattern):
            def match(self, graph, node):
                raise RuntimeError("export matcher failed")

        builder = optim.GraphBuilder(_double_neg_model(), schema_lookup=None)
        builder.register_pattern(FailingPattern())
        before = builder.to_onnx().SerializeToString()
        for option in (OptimizationMode.kOptimize, OptimizationMode.kRewritings):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(RuntimeError, "export matcher failed"),
            ):
                builder.to_onnx(optimize=option)
        self.assertEqual(builder.to_onnx().SerializeToString(), before)

    def test_rewriting_proto_binary_roundtrip_all_fields(self):
        plan = optim.RewritingProto()
        edit = plan.rewrites.add()
        edit.pattern_name = "Serialized"
        edit.graph_path.extend(["branch", "leaf"])
        edit.matched_nodes.extend([0, 2**64 - 1])
        edit.removed_initializers.extend([1])
        edit.added_nodes.append(oh.make_node("Identity", ["x"], ["y"]))
        edit.added_nodes_positions.extend([0])
        edit.added_initializers.append(oh.make_tensor("literal", TensorProto.FLOAT, [1], [2]))
        edit.added_initializer_positions.extend([1])
        edit.metadata_props.add().key = "local"
        edit.initializer_calls.append(oh.make_node("Prepare", ["weight"], ["prepared"]))
        edit.computed_initializer_positions.extend([2])
        plan.required_opset_imports.append(oh.make_opsetid("custom", 1))
        plan.added_functions.add().name = "Prepare"
        plan.metadata_props.add().key = "collection"
        parsed = optim.RewritingProto()
        parsed.ParseFromString(plan.SerializeToString())
        self.assertEqual(parsed.SerializeToString(), plan.SerializeToString())
        local = optim.LocalRewritingProto()
        local.ParseFromString(edit.SerializeToString())
        self.assertEqual(list(local.matched_nodes), [0, 2**64 - 1])

    def test_rewriting_proto_rejects_retired_renaming_field(self):
        legacy = b"\x4a\x0a\x0a\x03old\x12\x03new"
        plan_bytes = b"\x0a" + bytes([len(legacy)]) + legacy
        entry = b"\x0a\x04test\x12" + bytes([len(plan_bytes)]) + plan_bytes
        model_bytes = b"\xca\x3e" + bytes([len(entry)]) + entry
        for cls, data in (
            (optim.LocalRewritingProto, legacy),
            (optim.RewritingProto, plan_bytes),
            (ModelProto, model_bytes),
        ):
            with (
                self.subTest(proto=cls.__name__),
                self.assertRaisesRegex(ValueError, "retired field 9"),
            ):
                cls().ParseFromString(data)
        with self.assertRaisesRegex(RuntimeError, "invalid serialized proto data"):
            optim.LocalRewritingProto().ParseFromString(legacy.decode("ascii"))

    def test_inverse_restores_empty_initializer_and_output_annotations(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Concat", ["x", "z", "empty"], ["y"], axis=0)],
                "inverse_annotations",
                [
                    oh.make_tensor_value_info("x", TensorProto.INT64, [2]),
                    oh.make_tensor_value_info("z", TensorProto.INT64, [3]),
                ],
                [oh.make_tensor_value_info("y", TensorProto.INT64, ["size"])],
                [onh.from_array(numpy.array([], dtype=numpy.int64), name="empty")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        builder = optim.GraphBuilder(model)
        canonical = builder.build_graph()
        plan = optim.GraphGraph(builder, ["ConcatEmpty"]).optimize()
        self.assertEqual(list(builder.build_graph().node[0].input), ["x", "z"])
        self.assertFalse(builder.build_graph().initializer)
        restored = builder.undo_rewritings(plan).build_graph()
        self.assertEqual(
            restored.output[0].SerializeToString(), canonical.output[0].SerializeToString()
        )
        self.assertEqual(
            restored.node[0].SerializeToString(), canonical.node[0].SerializeToString()
        )
        self.assertEqual([value.name for value in restored.initializer], ["empty"])
        numpy.testing.assert_array_equal(
            onh.to_array(restored.initializer[0]), onh.to_array(canonical.initializer[0])
        )

    def test_canonicalize_and_inverse_restore_only_captured_edits(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Identity", ["x"], ["forwarded"]),
                    oh.make_node("Neg", ["forwarded"], ["middle"]),
                    oh.make_node("Neg", ["middle"], ["y"]),
                ],
                "inverse",
                [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        reference = optim.GraphBuilder(model)
        self.assertEqual(reference.remove_identity_nodes(), 1)
        canonical = reference.build_graph().SerializeToString()
        builder = optim.GraphBuilder(model)
        graph = optim.GraphGraph(builder, [NegNegPattern()])
        plan = graph.optimize()
        optimized = builder.build_graph().SerializeToString()
        self.assertEqual(
            builder.undo_rewritings(plan).build_graph().SerializeToString(), canonical
        )
        self.assertEqual(builder.build_graph().SerializeToString(), optimized)
        loaded = optim.RewritingProto()
        loaded.ParseFromString(plan.SerializeToString())
        self.assertEqual(
            builder.undo_rewritings(loaded).build_graph().SerializeToString(), canonical
        )
        self.assertFalse(any(name.startswith("inverse_") for name in dir(plan)))
        self.assertFalse(any(name.startswith("inverse_") for name in dir(plan.rewrites[0])))
        exported = builder.to_model()
        loaded_model = ModelProto()
        loaded_model.ParseFromString(exported.SerializeToString())
        reloaded = optim.GraphBuilder(loaded_model)
        with self.assertRaisesRegex(ValueError, "in-memory"):
            reloaded.undo_rewritings(loaded)
        invalid = optim.RewritingProto()
        invalid.ParseFromString(plan.SerializeToString())
        invalid.rewrites[0].pattern_name = "mismatch"
        with self.assertRaises(ValueError):
            builder.undo_rewritings(invalid)
        self.assertEqual(builder.build_graph().SerializeToString(), optimized)

    def test_replay_rejects_malformed_preparation_fields(self):
        model = _double_neg_model()
        for field in (
            "required_opset_imports",
            "added_functions",
            "initializer_calls",
            "computed_initializer_positions",
        ):
            with self.subTest(field=field):
                builder = optim.GraphBuilder(model)
                plan = optim.GraphGraph(builder, [NegNegPattern()]).optimize()
                target = (
                    plan
                    if field in ("required_opset_imports", "added_functions")
                    else plan.rewrites[0]
                )
                values = getattr(target, field)
                if field == "computed_initializer_positions":
                    values.append(0)
                else:
                    values.add()
                with self.assertRaises(ValueError):
                    optim.apply_rewritings_to_model(model, plan)

    def test_model_replay_installs_functions_and_imports(self):
        model = _double_neg_model()
        original = model.SerializeToString()
        builder = optim.GraphBuilder(model)
        plan = optim.GraphGraph(builder, [NegNegPattern()]).optimize()
        function = oh.make_function(
            "local.test",
            "Negate",
            ["arg"],
            ["result"],
            [oh.make_node("Neg", ["arg"], ["result"])],
            [oh.make_opsetid("", 18)],
        )
        plan.added_functions.append(function)
        plan.required_opset_imports.append(oh.make_opsetid("local.test", 1))
        plan.rewrites[0].added_nodes[0].op_type = "Negate"
        plan.rewrites[0].added_nodes[0].domain = "local.test"
        loaded = optim.RewritingProto()
        loaded.ParseFromString(plan.SerializeToString())
        result = optim.apply_rewritings_to_model(model, loaded)
        self.assertEqual(model.SerializeToString(), original)
        self.assertEqual(result.functions[0].name, "Negate")
        self.assertEqual(result.graph.node[0].op_type, "Negate")
        self.assertIn(("local.test", 1), [(i.domain, i.version) for i in result.opset_import])
        rewritten = optim.GraphBuilder(model).apply_rewritings(loaded)
        self.assertEqual(rewritten.build_graph().node[0].op_type, "Negate")
        self.assertEqual(rewritten.to_model().functions[0].name, "Negate")

    def test_optimizer_retains_python_pattern(self):
        model = _double_neg_model()
        builder = optim.GraphBuilder(model)
        pattern = NegNegPattern()
        graph = optim.GraphGraph(builder, [pattern])
        del pattern
        gc.collect()

        rewrites = graph.optimize()
        self.assertEqual(len(rewrites.rewrites), 1)

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
                    [rewrite.pattern_name for rewrite in rewrites.rewrites], ["RemoveUnusedNodes"]
                )
                self.assertEqual(report.rewrites, 1)
                self.assertEqual(len(report.patterns), 0)
                self.assertEqual(len(optimized.node), len(nodes))
                self.assertEqual([value.name for value in optimized.initializer], ["live"])
                self.assertEqual(
                    optimized.initializer[0].SerializeToString(),
                    model.graph.initializer[0].SerializeToString(),
                )
                loaded = optim.RewritingProto()
                loaded.ParseFromString(rewrites.SerializeToString())
                self.assertEqual(
                    optim.apply_rewritings(model, loaded).SerializeToString(),
                    optimized.SerializeToString(),
                )
                self.assertEqual(len(graph.optimize().rewrites), 0)

    def test_global_pattern_registration(self):
        optim.register_pattern(NegNegPattern())
        try:
            builder = optim.GraphBuilder(_double_neg_model())
            graph = optim.GraphGraph(builder)
            rewrites = graph.optimize()
            self.assertIn("NegNeg", optim.registered_pattern_names())
            self.assertTrue(
                any(rewrite.pattern_name == "NegNeg" for rewrite in rewrites.rewrites)
            )
        finally:
            optim.unregister_pattern("NegNeg")

    def test_builder_pattern_registration(self):
        builder = optim.GraphBuilder(_double_neg_model())
        builder.register_pattern(NegNegPattern())

        graph = optim.GraphGraph(builder)
        rewrites = graph.optimize()

        self.assertEqual(builder.registered_pattern_names(), ("NegNeg",))
        self.assertEqual(rewrites.rewrites[0].pattern_name, "NegNeg")

    def test_graph_pattern_registration(self):
        builder = optim.GraphBuilder(_double_neg_model())
        graph = optim.GraphGraph(builder, [NegNegPattern()])

        rewrites = graph.optimize()
        self.assertEqual(rewrites.rewrites[0].pattern_name, "NegNeg")

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
        optimized = builder.to_onnx()

        self.assertEqual(optim.registered_pattern_names()[0], "Cast")
        self.assertIsInstance(optim.CastPattern(), optim.PatternOptimization)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Identity"])
        self.assertEqual(rewrites.rewrites[0].pattern_name, "Cast")

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
        optimized = builder.to_onnx()

        self.assertIn("ClipClip", optim.registered_pattern_names())
        self.assertIsInstance(optim.ClipClipPattern(), optim.PatternOptimization)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Clip"])
        self.assertEqual(list(optimized.graph.node[0].input), ["x", "mn", "mx"])
        self.assertEqual(rewrites.rewrites[0].pattern_name, "ClipClip")

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
        optimized = builder.to_onnx()

        self.assertIn("GatherToSlice", optim.registered_pattern_names())
        self.assertIsInstance(optim.GatherToSlicePattern(), optim.PatternOptimization)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Slice"])
        self.assertEqual(optimized.graph.node[0].input[0], "x")
        self.assertEqual(optimized.graph.node[0].output[0], "y")
        self.assertEqual(rewrites.rewrites[0].pattern_name, "GatherToSlice")

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
        optimized = builder.to_onnx()

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
        self.assertNotIn("TransposeToInitializer", registered)
        self.assertFalse(hasattr(optim, "TransposeToInitializerPattern"))

    def test_transpose_folding_options_are_forwarded(self):
        graph = oh.make_graph(
            [
                oh.make_node("Transpose", ["weight"], ["weight_identity"], perm=[0, 1]),
                oh.make_node("Transpose", ["weight_identity"], ["y"], perm=[1, 0]),
            ],
            "g",
            [],
            [oh.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
            initializer=[oh.make_tensor("weight", TensorProto.FLOAT, [2, 3], range(6))],
        )
        builder = optim.GraphBuilder(
            oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        )
        options = ConstantFoldingOptions()
        options.enabled = False
        optim.GraphGraph(builder, ["TransposeTranspose"]).optimize(folding_options=options)
        self.assertEqual([node.op_type for node in builder.build_graph().node], ["Transpose"])

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
                self.assertEqual(len(rewrites.rewrites), 1)
                self.assertEqual(rewrites.rewrites[0].graph_path, ["then_branch"])
                self.assertTrue(
                    any(item.graph_path == ["then_branch"] for item in report.subgraphs)
                )
                loaded = optim.RewritingProto()
                loaded.ParseFromString(rewrites.SerializeToString())
                self.assertEqual(
                    optim.apply_rewritings(model, loaded).SerializeToString(),
                    builder.build_graph().SerializeToString(),
                )

    def test_replay_keeps_unnamed_subgraph_paths_after_parent_shift(self):
        then_branch = oh.make_graph(
            [
                oh.make_node("Neg", ["x"], ["middle"]),
                oh.make_node("Neg", ["middle"], ["then_out"]),
            ],
            "",
            [],
            [oh.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
        )
        else_branch = oh.make_graph(
            [oh.make_node("Identity", ["x"], ["else_out"])],
            "",
            [],
            [oh.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
        )
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Neg", ["x"], ["dead"]),
                    oh.make_node(
                        "If", ["cond"], ["y"], then_branch=then_branch, else_branch=else_branch
                    ),
                ],
                "main",
                [
                    oh.make_tensor_value_info("cond", TensorProto.BOOL, []),
                    oh.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                ],
                [oh.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        builder = optim.GraphBuilder(model)
        plan = optim.GraphGraph(builder, [NegNegPattern()]).optimize()
        self.assertEqual(
            [edit.pattern_name for edit in plan.rewrites], ["NegNeg", "RemoveUnusedNodes"]
        )
        reordered = optim.RewritingProto(rewrites=[plan.rewrites[1], plan.rewrites[0]])
        loaded = optim.RewritingProto()
        loaded.ParseFromString(reordered.SerializeToString())
        self.assertEqual(
            optim.apply_rewritings(model, loaded).SerializeToString(),
            builder.build_graph().SerializeToString(),
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
