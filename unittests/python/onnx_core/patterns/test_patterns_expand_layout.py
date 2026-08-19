# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests expand, transpose, squeeze, and unsqueeze graph patterns."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto, shape_inference
from onnx_light.ext_test_case import import_or_skip

from onnx_light.onnx_core import optimization

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")

REQUESTED_PATTERNS = (
    "Expand",
    "ExpandBroadcast",
    "ExpandSwap",
    "ExpandUnsqueezeExpand",
    "SwapExpandUnsqueeze",
    "SwapExpandReshape",
    "ShapeBasedStaticExpand",
    "ShapeBasedExpandBroadcast",
    "ShapeBasedExpandBroadcastMatMul",
    "ShapeBasedExpandSwap",
    "ShapeBasedExpandCastWhereSwap",
    "ShapeBasedConcatExpand",
    "TransposeTranspose",
    "TransposeGather",
    "ShapeTranspose",
    "UnsqueezeShape",
    "UnsqueezeUnsqueeze",
    "SqueezeUnsqueeze",
    "SwapUnsqueezeTranspose",
    "TransposeEqualReshape",
    "TransposeReshapeTranspose",
    "MulUnsqueezeUnsqueeze",
    "SqueezeAdd",
    "SqueezeBinaryUnsqueeze",
)


def _initializer(name, values, dtype):
    """Creates an initializer."""
    return onh.from_array(np.array(values, dtype=dtype), name=name)


def _make_model(nodes, inputs, outputs, initializers=(), value_info=(), opset=18):
    """Creates a model for one pattern test."""
    graph = oh.make_graph(
        nodes, "pattern_test", inputs, outputs, list(initializers), value_info=list(value_info)
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model.ir_version = 10
    return model


def _optimize(model, pattern, run_shape_inference=True):
    """Runs one isolated graph pattern."""
    work = shape_inference.infer_shapes(model) if run_shape_inference else model
    builder = optimization.GraphBuilder(work)
    graph = optimization.GraphGraph(builder, [pattern], use_global_patterns=False)
    rewrites = graph.optimize()
    return builder.to_onnx("model"), rewrites


def _ops(model):
    """Returns the model operator sequence."""
    return [node.op_type for node in model.graph.node]


def _initializer_values(model, name):
    """Returns an initializer as a NumPy array."""
    for tensor in model.graph.initializer:
        if tensor.name == name:
            return onh.to_array(tensor)
    raise AssertionError(f"Initializer {name!r} was not found.")


def _attribute_ints(node, name):
    """Returns an integer-list attribute."""
    for attribute in node.attribute:
        if attribute.name == name:
            return list(attribute.ints)
    raise AssertionError(f"Attribute {name!r} was not found.")


def _attribute_int(node, name):
    """Returns an integer attribute."""
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute.i
    raise AssertionError(f"Attribute {name!r} was not found.")


class TestExpandLayoutPatterns(unittest.TestCase):
    def assertPatternRewritten(self, rewrites, pattern):
        """Checks that the requested isolated pattern rewrote the graph."""
        self.assertIn(pattern, [rewrite.pattern_name for rewrite in rewrites])

    def assertPatternNotRewritten(self, rewrites, pattern):
        """Checks that the requested isolated pattern rejected the graph."""
        self.assertNotIn(pattern, [rewrite.pattern_name for rewrite in rewrites])

    def assertEquivalent(self, model, optimized, feeds):
        """Checks numerical equivalence with the onnx-light evaluator."""
        expected = ReferenceEvaluator(model).run(None, feeds)
        actual = ReferenceEvaluator(optimized).run(None, feeds)
        self.assertEqual(len(expected), len(actual))
        for expected_value, actual_value in zip(expected, actual):
            if np.issubdtype(expected_value.dtype, np.inexact):
                np.testing.assert_allclose(
                    expected_value, actual_value, rtol=1e-5, atol=1e-5, equal_nan=True
                )
            else:
                np.testing.assert_array_equal(expected_value, actual_value)

    def test_requested_patterns_are_registered(self):
        registered = set(optimization.standard_pattern_names())
        self.assertEqual([], sorted(set(REQUESTED_PATTERNS) - registered))

    def test_expand_execution_and_initializer_cleanup(self):
        model = _make_model(
            [
                oh.make_node("Reshape", ["X", "shape1"], ["xu1"]),
                oh.make_node("Expand", ["xu1", "expand_shape"], ["xm1"]),
                oh.make_node("Cast", ["Y"], ["xm2"], to=TensorProto.FLOAT),
                oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 8]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 8, 5]),
            ],
            [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3, 4, 5])],
            [
                _initializer("shape1", [1, 4, 8], np.int64),
                _initializer("expand_shape", [1, 4, 8], np.int64),
                _initializer("shape3", [2, 3, 4, 5], np.int64),
            ],
        )
        feeds = {
            "X": np.arange(32, dtype=np.float32).reshape(4, 8) / 32,
            "Y": np.arange(240, dtype=np.float32).reshape(2, 3, 8, 5) / 240,
        }

        optimized, rewrites = _optimize(model, "Expand")

        self.assertPatternRewritten(rewrites, "Expand")
        self.assertEqual(["Reshape", "Cast", "MatMul", "Reshape"], _ops(optimized))
        self.assertEqual([1, 4, 8], _initializer_values(optimized, "shape1").tolist())
        self.assertEqual([2, 3, 4, 5], _initializer_values(optimized, "shape3").tolist())
        self.assertEquivalent(model, optimized, feeds)

    def test_expand_no_match_when_shape_changes(self):
        model = _make_model(
            [oh.make_node("Expand", ["X", "shape"], ["Y"])],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
            [_initializer("shape", [2, 3], np.int64)],
        )

        optimized, rewrites = _optimize(model, "Expand")

        self.assertPatternNotRewritten(rewrites, "Expand")
        self.assertEqual(["Expand"], _ops(optimized))
        self.assertEquivalent(
            model, optimized, {"X": np.arange(3, dtype=np.float32).reshape(1, 3)}
        )

    def test_expand_broadcast_left_and_right(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                mul_inputs = ["expanded", "Y"] if side == "left" else ["Y", "expanded"]
                model = _make_model(
                    [
                        oh.make_node("Expand", ["X", "shape"], ["expanded"]),
                        oh.make_node("Mul", mul_inputs, ["Z"]),
                    ],
                    [
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 1]),
                        oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 6]),
                    ],
                    [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 4, 6])],
                    [_initializer("shape", [2, 4, 6], np.int64)],
                )
                feeds = {
                    "X": np.arange(4, dtype=np.float32).reshape(1, 4, 1),
                    "Y": np.arange(48, dtype=np.float32).reshape(2, 4, 6) + 0.5,
                }

                optimized, rewrites = _optimize(model, "ExpandBroadcast")

                self.assertPatternRewritten(rewrites, "ExpandBroadcast")
                self.assertEqual(["Mul"], _ops(optimized))
                expected_inputs = ["X", "Y"] if side == "left" else ["Y", "X"]
                self.assertEqual(expected_inputs, list(optimized.graph.node[0].input))
                self.assertEquivalent(model, optimized, feeds)

    def test_expand_broadcast_rejects_incompatible_or_shared_expand(self):
        cases = (
            ("incompatible", [1, 3], [2, 1], ["Z"]),
            ("shared", [2, 1], [2, 3], ["Z", "expanded"]),
        )
        for name, x_shape, y_shape, outputs in cases:
            with self.subTest(name=name):
                model = _make_model(
                    [
                        oh.make_node("Expand", ["X", "shape"], ["expanded"]),
                        oh.make_node("Mul", ["expanded", "Y"], ["Z"]),
                    ],
                    [
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
                        oh.make_tensor_value_info("Y", TensorProto.FLOAT, y_shape),
                    ],
                    [
                        oh.make_tensor_value_info(output, TensorProto.FLOAT, [2, 3])
                        for output in outputs
                    ],
                    [_initializer("shape", [2, 3], np.int64)],
                )

                optimized, rewrites = _optimize(model, "ExpandBroadcast")

                self.assertPatternNotRewritten(rewrites, "ExpandBroadcast")
                self.assertEqual(["Expand", "Mul"], _ops(optimized))

    def test_expand_swap_upstream_operator_variants(self):
        cases = (
            ("Exp", [], {}),
            ("Pow", ["power"], {"power": _initializer("power", [2.0], np.float32)}),
            ("Cast", [], {"to": TensorProto.FLOAT16}),
        )
        for op_type, extra_inputs, configuration in cases:
            with self.subTest(op_type=op_type):
                initializers = [_initializer("shape", [3, 1, 1], np.int64)]
                attributes = {}
                if op_type == "Pow":
                    initializers.append(configuration["power"])
                elif op_type == "Cast":
                    attributes["to"] = configuration["to"]
                model = _make_model(
                    [
                        oh.make_node("Expand", ["X", "shape"], ["expanded"]),
                        oh.make_node(op_type, ["expanded", *extra_inputs], ["Z"], **attributes),
                    ],
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5, 7])],
                    [
                        oh.make_tensor_value_info(
                            "Z",
                            TensorProto.FLOAT16 if op_type == "Cast" else TensorProto.FLOAT,
                            [3, 5, 7],
                        )
                    ],
                    initializers,
                )
                feeds = {"X": np.arange(35, dtype=np.float32).reshape(1, 5, 7) / 35}

                optimized, rewrites = _optimize(model, "ExpandSwap")

                self.assertPatternRewritten(rewrites, "ExpandSwap")
                self.assertEqual([op_type, "Expand"], _ops(optimized))
                self.assertEqual("X", optimized.graph.node[0].input[0])
                self.assertEqual("shape", optimized.graph.node[1].input[1])
                if op_type == "Pow":
                    self.assertEqual("power", optimized.graph.node[0].input[1])
                if op_type == "Cast":
                    self.assertEqual(
                        TensorProto.FLOAT16, _attribute_int(optimized.graph.node[0], "to")
                    )
                self.assertEquivalent(model, optimized, feeds)

    def test_expand_swap_rejects_binary_and_shared_consumers(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                nodes = [
                    oh.make_node("Expand", ["X", "shape"], ["expanded"]),
                    oh.make_node(
                        "Add" if not shared else "Exp",
                        ["expanded", "Y"] if not shared else ["expanded"],
                        ["Z"],
                    ),
                ]
                outputs = [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5, 7])]
                if shared:
                    nodes.append(oh.make_node("Identity", ["expanded"], ["other"]))
                    outputs.append(
                        oh.make_tensor_value_info("other", TensorProto.FLOAT, [3, 5, 7])
                    )
                model = _make_model(
                    nodes,
                    [
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5, 7]),
                        *(
                            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5, 7])]
                            if not shared
                            else []
                        ),
                    ],
                    outputs,
                    [_initializer("shape", [3, 1, 1], np.int64)],
                )

                optimized, rewrites = _optimize(model, "ExpandSwap")

                self.assertPatternNotRewritten(rewrites, "ExpandSwap")
                self.assertEqual(
                    ["Expand", "Add" if not shared else "Exp", *(["Identity"] if shared else [])],
                    _ops(optimized),
                )

    def test_swap_expand_unsqueeze_and_generated_shape(self):
        for axes, expected_shape in (([1], [3, 1, 1, 1]), ([-1], [3, 1, 1, 1])):
            with self.subTest(axes=axes):
                model = _make_model(
                    [
                        oh.make_node("Expand", ["X", "shape"], ["expanded"]),
                        oh.make_node("Unsqueeze", ["expanded", "axes"], ["Y"]),
                    ],
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5, 7])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                    [
                        _initializer("shape", [3, 1, 1], np.int64),
                        _initializer("axes", axes, np.int64),
                    ],
                )
                feeds = {"X": np.arange(35, dtype=np.float32).reshape(1, 5, 7)}

                optimized, rewrites = _optimize(model, "SwapExpandUnsqueeze")

                self.assertPatternRewritten(rewrites, "SwapExpandUnsqueeze")
                self.assertEqual(["Unsqueeze", "Expand"], _ops(optimized))
                generated = optimized.graph.node[1].input[1]
                self.assertEqual(
                    expected_shape, _initializer_values(optimized, generated).tolist()
                )
                self.assertEquivalent(model, optimized, feeds)

    def test_swap_expand_unsqueeze_no_match_cases(self):
        cases = ("dynamic_axes", "shared_expand", "old_opset")
        for case in cases:
            with self.subTest(case=case):
                inputs = [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5, 7])]
                initializers = [_initializer("shape", [3, 1, 1], np.int64)]
                if case == "dynamic_axes":
                    inputs.append(oh.make_tensor_value_info("axes", TensorProto.INT64, [1]))
                else:
                    initializers.append(_initializer("axes", [1], np.int64))
                nodes = [
                    oh.make_node("Expand", ["X", "shape"], ["expanded"]),
                    oh.make_node("Unsqueeze", ["expanded", "axes"], ["Y"]),
                ]
                outputs = [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)]
                if case == "shared_expand":
                    nodes.append(oh.make_node("Identity", ["expanded"], ["other"]))
                    outputs.append(
                        oh.make_tensor_value_info("other", TensorProto.FLOAT, [3, 5, 7])
                    )
                model = _make_model(
                    nodes, inputs, outputs, initializers, opset=11 if case == "old_opset" else 18
                )

                optimized, rewrites = _optimize(model, "SwapExpandUnsqueeze")

                self.assertPatternNotRewritten(rewrites, "SwapExpandUnsqueeze")
                self.assertEqual(
                    ["Expand", "Unsqueeze", *(["Identity"] if case == "shared_expand" else [])],
                    _ops(optimized),
                )

    def test_expand_unsqueeze_expand_fusion_and_initializer(self):
        model = _make_model(
            [
                oh.make_node("Expand", ["X", "shape1"], ["expanded"]),
                oh.make_node("Unsqueeze", ["expanded", "axes"], ["unsqueezed"]),
                oh.make_node("Expand", ["unsqueezed", "shape2"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [5, 2, 3, 4])],
            [
                _initializer("shape1", [5, 3, 4], np.int64),
                _initializer("axes", [1], np.int64),
                _initializer("shape2", [5, 2, 3, 4], np.int64),
            ],
        )
        feeds = {"X": np.arange(12, dtype=np.float32).reshape(1, 3, 4)}

        optimized, rewrites = _optimize(model, "ExpandUnsqueezeExpand")

        self.assertPatternRewritten(rewrites, "ExpandUnsqueezeExpand")
        self.assertEqual(["Unsqueeze", "Expand"], _ops(optimized))
        generated = optimized.graph.node[1].input[1]
        self.assertEqual([5, 2, 3, 4], _initializer_values(optimized, generated).tolist())
        self.assertEquivalent(model, optimized, feeds)

    def test_expand_unsqueeze_expand_rejects_bad_topology_and_shared_usage(self):
        cases = ("missing_second_expand", "rank_mismatch", "shared_unsqueeze")
        for case in cases:
            with self.subTest(case=case):
                axes = [1, 2] if case == "rank_mismatch" else [1]
                nodes = [
                    oh.make_node("Expand", ["X", "shape1"], ["expanded"]),
                    oh.make_node("Unsqueeze", ["expanded", "axes"], ["unsqueezed"]),
                ]
                initializers = [
                    _initializer("shape1", [5, 3, 4], np.int64),
                    _initializer("axes", axes, np.int64),
                ]
                outputs = []
                if case != "missing_second_expand":
                    nodes.append(oh.make_node("Expand", ["unsqueezed", "shape2"], ["Y"]))
                    initializers.append(_initializer("shape2", [5, 2, 3, 4], np.int64))
                    outputs.append(oh.make_tensor_value_info("Y", TensorProto.FLOAT, None))
                else:
                    outputs.append(
                        oh.make_tensor_value_info("unsqueezed", TensorProto.FLOAT, None)
                    )
                if case == "shared_unsqueeze":
                    nodes.append(oh.make_node("Identity", ["unsqueezed"], ["other"]))
                    outputs.append(oh.make_tensor_value_info("other", TensorProto.FLOAT, None))
                model = _make_model(
                    nodes,
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4])],
                    outputs,
                    initializers,
                )

                optimized, rewrites = _optimize(model, "ExpandUnsqueezeExpand")

                self.assertPatternNotRewritten(rewrites, "ExpandUnsqueezeExpand")
                self.assertEqual([node.op_type for node in nodes], _ops(optimized))

    def test_swap_expand_reshape_upstream_topology(self):
        model = _make_model(
            [
                oh.make_node("Shape", ["X"], ["batch"], start=0, end=1),
                oh.make_node("Concat", ["batch", "ones"], ["shape"], axis=0),
                oh.make_node("Expand", ["weight", "shape"], ["expanded"]),
                oh.make_node("Reshape", ["expanded", "reshape_shape"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 4])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", 1, 4])],
            [
                _initializer("ones", [1, 1], np.int64),
                _initializer("reshape_shape", [0, 1, -1], np.int64),
                _initializer(
                    "weight",
                    np.array([2, 3, 4, 5], dtype=np.float32).reshape(1, 4, 1),
                    np.float32,
                ),
            ],
        )
        feeds = {"X": np.arange(12, dtype=np.float32).reshape(3, 4)}

        optimized, rewrites = _optimize(model, "SwapExpandReshape")

        self.assertPatternRewritten(rewrites, "SwapExpandReshape")
        self.assertEqual(["Shape", "Concat", "Expand"], _ops(optimized))
        reshaped_weight = optimized.graph.node[2].input[0]
        self.assertEqual((1, 1, 4), _initializer_values(optimized, reshaped_weight).shape)
        self.assertEqual("shape", optimized.graph.node[2].input[1])
        self.assertEquivalent(model, optimized, feeds)

    def test_swap_expand_reshape_no_match_when_target_is_not_specialized(self):
        model = _make_model(
            [
                oh.make_node("Expand", ["X", "expand_shape"], ["expanded"]),
                oh.make_node("Reshape", ["expanded", "reshape_shape"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2, 1])],
            [
                _initializer("expand_shape", [3, 2, 1], np.int64),
                _initializer("reshape_shape", [0, 1, -1], np.int64),
            ],
        )

        optimized, rewrites = _optimize(model, "SwapExpandReshape")

        self.assertPatternNotRewritten(rewrites, "SwapExpandReshape")
        self.assertEqual(["Expand", "Reshape"], _ops(optimized))

    def test_shape_based_static_expand_upstream_graph(self):
        model = _make_model(
            [
                oh.make_node("Shape", ["X"], ["prefix"], start=0, end=-1),
                oh.make_node("Concat", ["prefix", "two"], ["target"], axis=0),
                oh.make_node("Expand", ["X", "target"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, "d", 1])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, "d", 2])],
            [_initializer("two", [2], np.int64)],
        )
        feeds = {"X": np.arange(72, dtype=np.float32).reshape(2, 3, 12, 1)}

        optimized, rewrites = _optimize(
            model, "ShapeBasedStaticExpand", run_shape_inference=False
        )

        self.assertPatternRewritten(rewrites, "ShapeBasedStaticExpand")
        self.assertEqual(["Expand"], _ops(optimized))
        target = optimized.graph.node[0].input[1]
        self.assertEqual([1, 1, 1, 2], _initializer_values(optimized, target).tolist())
        self.assertEquivalent(model, optimized, feeds)

    def test_shape_based_static_expand_rejects_symbolic_changed_dimension(self):
        model = _make_model(
            [oh.make_node("Expand", ["X", "target"], ["Y"])],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 1]),
                oh.make_tensor_value_info("target", TensorProto.INT64, [2]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["b", 4])],
        )

        optimized, rewrites = _optimize(
            model, "ShapeBasedStaticExpand", run_shape_inference=False
        )

        self.assertPatternNotRewritten(rewrites, "ShapeBasedStaticExpand")
        self.assertEqual(["Expand"], _ops(optimized))

    def test_shape_based_expand_broadcast_upstream_graph(self):
        model = _make_model(
            [
                oh.make_node("Reshape", ["X", "column_shape"], ["column"]),
                oh.make_node("Reshape", ["X", "row_shape"], ["row"]),
                oh.make_node("Shape", ["X"], ["shape"]),
                oh.make_node("Concat", ["shape", "shape"], ["target"], axis=0),
                oh.make_node("Expand", ["column", "target"], ["columns"]),
                oh.make_node("Expand", ["row", "target"], ["rows"]),
                oh.make_node("Add", ["columns", "rows"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["d"])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["d", "d"])],
            [
                _initializer("column_shape", [-1, 1], np.int64),
                _initializer("row_shape", [1, -1], np.int64),
            ],
        )
        feeds = {"X": np.arange(3, dtype=np.float32)}

        optimized, rewrites = _optimize(model, "ShapeBasedExpandBroadcast")

        self.assertPatternRewritten(rewrites, "ShapeBasedExpandBroadcast")
        self.assertEqual(["Reshape", "Reshape", "Add"], _ops(optimized))
        self.assertEqual(["column", "row"], list(optimized.graph.node[-1].input))
        self.assertEquivalent(model, optimized, feeds)

    def test_shape_based_expand_broadcast_preserves_shared_expand(self):
        model = _make_model(
            [
                oh.make_node("Shape", ["Y"], ["target"]),
                oh.make_node("Expand", ["X", "target"], ["expanded"]),
                oh.make_node("Mul", ["expanded", "Y"], ["Z"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, "b"]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", "b"]),
            ],
            [
                oh.make_tensor_value_info("Z", TensorProto.FLOAT, ["a", "b"]),
                oh.make_tensor_value_info("expanded", TensorProto.FLOAT, ["a", "b"]),
            ],
        )
        feeds = {
            "X": np.arange(3, dtype=np.float32).reshape(1, 3),
            "Y": np.arange(6, dtype=np.float32).reshape(2, 3) + 1,
        }

        optimized, rewrites = _optimize(model, "ShapeBasedExpandBroadcast")

        self.assertPatternRewritten(rewrites, "ShapeBasedExpandBroadcast")
        self.assertEqual(["Shape", "Expand", "Mul"], _ops(optimized))
        self.assertEqual(["X", "Y"], list(optimized.graph.node[-1].input))
        self.assertEquivalent(model, optimized, feeds)

    def test_shape_based_expand_broadcast_rejects_incompatible_symbols(self):
        model = _make_model(
            [
                oh.make_node("Shape", ["Y"], ["target"]),
                oh.make_node("Expand", ["X", "target"], ["expanded"]),
                oh.make_node("Add", ["expanded", "Y"], ["Z"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, ["m", "b"]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", "b"]),
            ],
            [oh.make_tensor_value_info("Z", TensorProto.FLOAT, ["a", "b"])],
        )

        optimized, rewrites = _optimize(
            model, "ShapeBasedExpandBroadcast", run_shape_inference=False
        )

        self.assertPatternNotRewritten(rewrites, "ShapeBasedExpandBroadcast")
        self.assertEqual(["Shape", "Expand", "Add"], _ops(optimized))

    def test_shape_based_expand_broadcast_matmul(self):
        model = _make_model(
            [
                oh.make_node("Shape", ["X"], ["batch"], start=0, end=1),
                oh.make_node("Shape", ["Y"], ["matrix"], start=1),
                oh.make_node("Concat", ["batch", "matrix"], ["target"], axis=0),
                oh.make_node("Expand", ["Y", "target"], ["expanded"]),
                oh.make_node("MatMul", ["X", "expanded"], ["Z"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 3, 4]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 5]),
            ],
            [oh.make_tensor_value_info("Z", TensorProto.FLOAT, ["a", 3, 5])],
        )
        feeds = {
            "X": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "Y": np.arange(20, dtype=np.float32).reshape(1, 4, 5),
        }

        optimized, rewrites = _optimize(model, "ShapeBasedExpandBroadcastMatMul")

        self.assertPatternRewritten(rewrites, "ShapeBasedExpandBroadcastMatMul")
        self.assertEqual(["MatMul"], _ops(optimized))
        self.assertEqual(["X", "Y"], list(optimized.graph.node[0].input))
        self.assertEquivalent(model, optimized, feeds)

    def test_shape_based_expand_broadcast_matmul_no_match(self):
        model = _make_model(
            [
                oh.make_node("Expand", ["Y", "target"], ["expanded"]),
                oh.make_node("MatMul", ["X", "expanded"], ["Z"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 2, 3]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["m", 3, 4]),
                oh.make_tensor_value_info("target", TensorProto.INT64, [3]),
            ],
            [oh.make_tensor_value_info("Z", TensorProto.FLOAT, ["a", 2, 4])],
        )

        optimized, rewrites = _optimize(
            model, "ShapeBasedExpandBroadcastMatMul", run_shape_inference=False
        )

        self.assertPatternNotRewritten(rewrites, "ShapeBasedExpandBroadcastMatMul")
        self.assertEqual(["Expand", "MatMul"], _ops(optimized))

    def test_shape_based_expand_swap_upstream_variants(self):
        cases = (
            ("one", False, False),
            ("one_with_leading_batch", True, False),
            ("two", True, True),
        )
        for name, leading_batch, two_expands in cases:
            with self.subTest(name=name):
                nodes = [oh.make_node("Reshape", ["X", "column_shape"], ["column"])]
                initializers = [
                    _initializer("column_shape", [-1, 1], np.int64),
                    _initializer("four", [4], np.int64),
                ]
                if two_expands:
                    nodes.append(oh.make_node("Reshape", ["X", "row_shape"], ["row"]))
                    initializers.append(_initializer("row_shape", [1, -1], np.int64))
                nodes.extend(
                    [
                        oh.make_node("Shape", ["X"], ["shape"]),
                        oh.make_node(
                            "Concat",
                            ["four", "shape", "shape"] if leading_batch else ["shape", "shape"],
                            ["target"],
                            axis=0,
                        ),
                        oh.make_node("Expand", ["column", "target"], ["columns"]),
                    ]
                )
                if two_expands:
                    nodes.append(oh.make_node("Expand", ["row", "target"], ["rows"]))
                    right = "rows"
                else:
                    initializers.append(_initializer("scalar", [3.0], np.float32))
                    right = "scalar"
                nodes.append(oh.make_node("Add", ["columns", right], ["Y"]))
                output_shape = [4, "d", "d"] if leading_batch else ["d", "d"]
                model = _make_model(
                    nodes,
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["d"])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
                    initializers,
                )
                feeds = {"X": np.arange(3, dtype=np.float32)}

                optimized, rewrites = _optimize(model, "ShapeBasedExpandSwap")

                self.assertPatternRewritten(rewrites, "ShapeBasedExpandSwap")
                self.assertEqual("Expand", optimized.graph.node[-1].op_type)
                self.assertEqual("Add", optimized.graph.node[-2].op_type)
                self.assertEqual("target", optimized.graph.node[-1].input[1])
                self.assertEquivalent(model, optimized, feeds)

    def test_shape_based_expand_swap_preserves_shared_input_expand(self):
        model = _make_model(
            [
                oh.make_node("Expand", ["X", "target"], ["expanded"]),
                oh.make_node("Add", ["expanded", "scalar"], ["Y"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 1]),
                oh.make_tensor_value_info("target", TensorProto.INT64, [2]),
            ],
            [
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4]),
                oh.make_tensor_value_info("expanded", TensorProto.FLOAT, [4, 4]),
            ],
            [_initializer("scalar", [1.0], np.float32)],
        )
        feeds = {
            "X": np.arange(4, dtype=np.float32).reshape(4, 1),
            "target": np.array([4, 4], dtype=np.int64),
        }

        optimized, rewrites = _optimize(model, "ShapeBasedExpandSwap", run_shape_inference=False)

        self.assertPatternRewritten(rewrites, "ShapeBasedExpandSwap")
        self.assertEqual(["Expand", "Add", "Expand"], _ops(optimized))
        self.assertEquivalent(model, optimized, feeds)

    def test_shape_based_expand_swap_rejects_redundant_expand(self):
        model = _make_model(
            [
                oh.make_node("Expand", ["X", "target"], ["expanded"]),
                oh.make_node("Add", ["expanded", "scalar"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])],
            [_initializer("target", [4, 4], np.int64), _initializer("scalar", [1.0], np.float32)],
        )

        optimized, rewrites = _optimize(model, "ShapeBasedExpandSwap")

        self.assertPatternNotRewritten(rewrites, "ShapeBasedExpandSwap")
        self.assertEqual(["Expand", "Add"], _ops(optimized))

    def test_shape_based_expand_cast_where_upstream_branches(self):
        for expanded_branch in (1, 2):
            with self.subTest(expanded_branch=expanded_branch):
                where_inputs = (
                    ["condition", "expanded", "constant"]
                    if expanded_branch == 1
                    else ["condition", "constant", "expanded"]
                )
                model = _make_model(
                    [
                        oh.make_node("Shape", ["X"], ["batch"], start=0, end=1),
                        oh.make_node("Concat", ["batch", "batch", "one"], ["target"], axis=0),
                        oh.make_node("Expand", ["X", "target"], ["expanded"]),
                        oh.make_node("Cast", ["expanded"], ["condition"], to=TensorProto.BOOL),
                        oh.make_node("Where", where_inputs, ["Y"]),
                    ],
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["b", 1, "c"])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["b", "b", "c"])],
                    [
                        _initializer("one", [1], np.int64),
                        _initializer("constant", [-np.inf], np.float32),
                    ],
                )
                feeds = {"X": (np.arange(12).reshape(3, 1, 4) % 3).astype(np.float32)}

                optimized, rewrites = _optimize(model, "ShapeBasedExpandCastWhereSwap")

                self.assertPatternRewritten(rewrites, "ShapeBasedExpandCastWhereSwap")
                self.assertEqual(["Shape", "Concat", "Cast", "Where", "Expand"], _ops(optimized))
                self.assertEqual("X", optimized.graph.node[-3].input[0])
                self.assertEqual("target", optimized.graph.node[-1].input[1])
                self.assertEquivalent(model, optimized, feeds)

    def test_shape_based_expand_cast_where_rejects_extra_expand_use(self):
        model = _make_model(
            [
                oh.make_node("Expand", ["X", "target"], ["expanded"]),
                oh.make_node("Cast", ["expanded"], ["condition"], to=TensorProto.BOOL),
                oh.make_node("Where", ["condition", "expanded", "zero"], ["Y"]),
                oh.make_node("Identity", ["expanded"], ["other"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1]),
                oh.make_tensor_value_info("target", TensorProto.INT64, [2]),
            ],
            [
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3]),
                oh.make_tensor_value_info("other", TensorProto.FLOAT, [2, 3]),
            ],
            [_initializer("zero", [0.0], np.float32)],
        )

        optimized, rewrites = _optimize(
            model, "ShapeBasedExpandCastWhereSwap", run_shape_inference=False
        )

        self.assertPatternNotRewritten(rewrites, "ShapeBasedExpandCastWhereSwap")
        self.assertEqual(["Expand", "Cast", "Where", "Identity"], _ops(optimized))

    def test_shape_based_concat_expand_upstream_graph_and_initializer(self):
        model = _make_model(
            [
                oh.make_node("Shape", ["X"], ["first"], start=0, end=1),
                oh.make_node("Concat", ["first", "two"], ["target"], axis=0),
                oh.make_node("Expand", ["X", "target"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 1])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", 2])],
            [_initializer("two", [2], np.int64)],
        )
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(6, 1)}

        optimized, rewrites = _optimize(
            model, "ShapeBasedConcatExpand", run_shape_inference=False
        )

        self.assertPatternRewritten(rewrites, "ShapeBasedConcatExpand")
        self.assertEqual(["Expand"], _ops(optimized))
        target = optimized.graph.node[0].input[1]
        self.assertEqual([1, 2], _initializer_values(optimized, target).tolist())
        self.assertEquivalent(model, optimized, feeds)

    def test_shape_based_concat_expand_rejects_shared_target(self):
        model = _make_model(
            [
                oh.make_node("Shape", ["X"], ["first"], start=0, end=1),
                oh.make_node("Concat", ["first", "two"], ["target"], axis=0),
                oh.make_node("Expand", ["X", "target"], ["Y"]),
                oh.make_node("Identity", ["target"], ["saved_target"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 1])],
            [
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", 2]),
                oh.make_tensor_value_info("saved_target", TensorProto.INT64, [2]),
            ],
            [_initializer("two", [2], np.int64)],
        )

        optimized, rewrites = _optimize(
            model, "ShapeBasedConcatExpand", run_shape_inference=False
        )

        self.assertPatternNotRewritten(rewrites, "ShapeBasedConcatExpand")
        self.assertEqual(["Shape", "Concat", "Expand", "Identity"], _ops(optimized))

    def test_transpose_transpose_upstream_variants(self):
        cases = (
            ("identity_shared", [1, 0, 3, 2], [1, 0, 3, 2], True, ["Transpose", "Identity"]),
            ("merged", [1, 0, 2], [0, 2, 1], False, ["Transpose"]),
        )
        for name, first_perm, second_perm, shared, expected_ops in cases:
            with self.subTest(name=name):
                rank = len(first_perm)
                input_shape = list(range(2, rank + 2))
                second_shape = [input_shape[first_perm[index]] for index in second_perm]
                nodes = [
                    oh.make_node("Transpose", ["X"], ["first"], perm=first_perm),
                    oh.make_node("Transpose", ["first"], ["Y"], perm=second_perm),
                ]
                outputs = [oh.make_tensor_value_info("Y", TensorProto.FLOAT, second_shape)]
                if shared:
                    outputs.append(
                        oh.make_tensor_value_info(
                            "first",
                            TensorProto.FLOAT,
                            [input_shape[index] for index in first_perm],
                        )
                    )
                model = _make_model(
                    nodes,
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
                    outputs,
                )
                feeds = {
                    "X": np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
                }

                optimized, rewrites = _optimize(model, "TransposeTranspose")

                self.assertPatternRewritten(rewrites, "TransposeTranspose")
                self.assertEqual(expected_ops, _ops(optimized))
                if name == "merged":
                    self.assertEqual([1, 2, 0], _attribute_ints(optimized.graph.node[0], "perm"))
                self.assertEquivalent(model, optimized, feeds)

    def test_transpose_transpose_rejects_shared_non_identity_merge(self):
        model = _make_model(
            [
                oh.make_node("Transpose", ["X"], ["first"], perm=[1, 0, 2]),
                oh.make_node("Transpose", ["first"], ["Y"], perm=[0, 2, 1]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 2]),
                oh.make_tensor_value_info("first", TensorProto.FLOAT, [3, 2, 4]),
            ],
        )

        optimized, rewrites = _optimize(model, "TransposeTranspose")

        self.assertPatternNotRewritten(rewrites, "TransposeTranspose")
        self.assertEqual(["Transpose", "Transpose"], _ops(optimized))

    def test_transpose_gather_drop_and_swap_variants(self):
        cases = (
            ("drop", [1, 0, 2, 3], 0, ["Gather"]),
            ("swap", [2, 1, 0], 1, ["Gather", "Transpose"]),
        )
        for name, perm, axis, expected_ops in cases:
            with self.subTest(name=name):
                input_shape = [3, 4, 5, 6][: len(perm)]
                model = _make_model(
                    [
                        oh.make_node("Transpose", ["X"], ["transposed"], perm=perm),
                        oh.make_node("Gather", ["transposed", "index"], ["Y"], axis=axis),
                    ],
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                    [_initializer("index", 1, np.int64)],
                )
                feeds = {
                    "X": np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
                }

                optimized, rewrites = _optimize(model, "TransposeGather")

                self.assertPatternRewritten(rewrites, "TransposeGather")
                self.assertEqual(expected_ops, _ops(optimized))
                self.assertEqual(perm[axis], _attribute_int(optimized.graph.node[0], "axis"))
                if name == "swap":
                    self.assertEqual([1, 0], _attribute_ints(optimized.graph.node[1], "perm"))
                self.assertEquivalent(model, optimized, feeds)

    def test_transpose_gather_reused(self):
        model = _make_model(
            [
                oh.make_node("Transpose", ["X"], ["transposed"], perm=[1, 0, 2, 3]),
                oh.make_node("Gather", ["transposed", "index"], ["Y"], axis=0),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3, 16, 80])],
            [
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 16, 80]),
                oh.make_tensor_value_info("transposed", TensorProto.FLOAT, [3, 4, 16, 80]),
            ],
            [_initializer("index", 1, np.int64)],
        )
        feeds = {"X": np.arange(4 * 3 * 16 * 80, dtype=np.float32).reshape(4, 3, 16, 80)}

        optimized, rewrites = _optimize(model, "TransposeGather")

        self.assertPatternRewritten(rewrites, "TransposeGather")
        self.assertEqual(["Transpose", "Gather"], _ops(optimized))
        transpose, gather = optimized.graph.node
        self.assertEqual(["X"], list(transpose.input))
        self.assertEqual(["transposed"], list(transpose.output))
        self.assertEqual([1, 0, 2, 3], _attribute_ints(transpose, "perm"))
        self.assertEqual(["X", "index"], list(gather.input))
        self.assertEqual(["Y"], list(gather.output))
        self.assertEqual(1, _attribute_int(gather, "axis"))
        self.assertEqual(["Y", "transposed"], [output.name for output in optimized.graph.output])
        self.assertEquivalent(model, optimized, feeds)

    def test_transpose_gather_rejects_vector_index(self):
        model = _make_model(
            [
                oh.make_node("Transpose", ["X"], ["transposed"], perm=[1, 0, 2, 3]),
                oh.make_node("Gather", ["transposed", "index"], ["Y"], axis=0),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5, 6])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            [_initializer("index", [0, 1], np.int64)],
        )

        optimized, rewrites = _optimize(model, "TransposeGather")

        self.assertPatternNotRewritten(rewrites, "TransposeGather")
        self.assertEqual(["Transpose", "Gather"], _ops(optimized))

    def test_shape_transpose_ranges_and_shared_usage(self):
        cases = (
            ("full", {}, [2, 0, 1], False),
            ("range", {"start": 1, "end": -1}, [0], False),
            ("shared", {}, [2, 0, 1], True),
        )
        for name, attributes, expected_indices, shared in cases:
            with self.subTest(name=name):
                nodes = [
                    oh.make_node("Transpose", ["X"], ["transposed"], perm=[2, 0, 1]),
                    oh.make_node("Shape", ["transposed"], ["Y"], **attributes),
                ]
                outputs = [
                    oh.make_tensor_value_info("Y", TensorProto.INT64, [len(expected_indices)])
                ]
                if shared:
                    outputs.append(
                        oh.make_tensor_value_info("transposed", TensorProto.FLOAT, [4, 2, 3])
                    )
                model = _make_model(
                    nodes, [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])], outputs
                )
                feeds = {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}

                optimized, rewrites = _optimize(model, "ShapeTranspose")

                self.assertPatternRewritten(rewrites, "ShapeTranspose")
                self.assertEqual(
                    ["Transpose", "Shape", "Gather"] if shared else ["Shape", "Gather"],
                    _ops(optimized),
                )
                gather = optimized.graph.node[-1]
                self.assertEqual(0, _attribute_int(gather, "axis"))
                self.assertEqual(
                    expected_indices, _initializer_values(optimized, gather.input[1]).tolist()
                )
                self.assertEquivalent(model, optimized, feeds)

    def test_shape_transpose_no_match_without_transpose_parent(self):
        model = _make_model(
            [oh.make_node("Shape", ["X"], ["Y"])],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [oh.make_tensor_value_info("Y", TensorProto.INT64, [3])],
        )

        optimized, rewrites = _optimize(model, "ShapeTranspose")

        self.assertPatternNotRewritten(rewrites, "ShapeTranspose")
        self.assertEqual(["Shape"], _ops(optimized))

    def test_unsqueeze_shape_ranges_and_shared_usage(self):
        cases = (
            ("one_axis", [1], {}, False),
            ("two_axes_range", [0, 2], {"start": 1, "end": 4}, False),
            ("shared", [1], {}, True),
        )
        for name, axes, attributes, shared in cases:
            with self.subTest(name=name):
                nodes = [
                    oh.make_node("Unsqueeze", ["X", "axes"], ["unsqueezed"]),
                    oh.make_node("Shape", ["unsqueezed"], ["Y"], **attributes),
                ]
                output_rank = 3 + len(axes)
                start = attributes.get("start", 0)
                end = attributes.get("end", output_rank)
                outputs = [oh.make_tensor_value_info("Y", TensorProto.INT64, [end - start])]
                if shared:
                    outputs.append(
                        oh.make_tensor_value_info("unsqueezed", TensorProto.FLOAT, None)
                    )
                model = _make_model(
                    nodes,
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
                    outputs,
                    [_initializer("axes", axes, np.int64)],
                )
                feeds = {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}

                optimized, rewrites = _optimize(model, "UnsqueezeShape")

                self.assertPatternRewritten(rewrites, "UnsqueezeShape")
                self.assertEqual("Concat", optimized.graph.node[-1].op_type)
                if shared:
                    self.assertEqual("Unsqueeze", optimized.graph.node[0].op_type)
                initializer_names = {
                    initializer.name for initializer in optimized.graph.initializer
                }
                one_inputs = []
                for input_name in optimized.graph.node[-1].input:
                    if input_name in initializer_names and np.array_equal(
                        _initializer_values(optimized, input_name), np.array([1], dtype=np.int64)
                    ):
                        one_inputs.append(input_name)
                normalized_axes = [axis if axis >= 0 else axis + output_rank for axis in axes]
                expected_ones = sum(start <= axis < end for axis in normalized_axes)
                self.assertEqual(expected_ones, len(one_inputs))
                self.assertEquivalent(model, optimized, feeds)

    def test_unsqueeze_shape_rejects_dynamic_axes(self):
        model = _make_model(
            [
                oh.make_node("Unsqueeze", ["X", "axes"], ["unsqueezed"]),
                oh.make_node("Shape", ["unsqueezed"], ["Y"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
                oh.make_tensor_value_info("axes", TensorProto.INT64, [1]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.INT64, [4])],
        )

        optimized, rewrites = _optimize(model, "UnsqueezeShape")

        self.assertPatternNotRewritten(rewrites, "UnsqueezeShape")
        self.assertEqual(["Unsqueeze", "Shape"], _ops(optimized))

    def test_unsqueeze_unsqueeze_single_axis_upstream_subtests(self):
        feeds = {"X": np.arange(12, dtype=np.float32).reshape(3, 4)}
        for first_axis in range(3):
            for second_axis in range(4):
                with self.subTest(first_axis=first_axis, second_axis=second_axis):
                    model = _make_model(
                        [
                            oh.make_node("Unsqueeze", ["X", "first_axes"], ["first"]),
                            oh.make_node("Unsqueeze", ["first", "second_axes"], ["Y"]),
                        ],
                        [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", "b"])],
                        [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                        [
                            _initializer("first_axes", [first_axis], np.int64),
                            _initializer("second_axes", [second_axis], np.int64),
                        ],
                    )

                    optimized, rewrites = _optimize(model, "UnsqueezeUnsqueeze")

                    self.assertPatternRewritten(rewrites, "UnsqueezeUnsqueeze")
                    self.assertEqual(["Unsqueeze"], _ops(optimized))
                    self.assertEquivalent(model, optimized, feeds)

    def test_unsqueeze_unsqueeze_multi_axis_upstream_subtests(self):
        feeds = {"X": np.arange(12, dtype=np.float32).reshape(3, 4)}
        cases = itertools.product(
            itertools.combinations(range(3), 2), itertools.combinations(range(4), 2)
        )
        for first_axes, second_axes in cases:
            with self.subTest(first_axes=first_axes, second_axes=second_axes):
                model = _make_model(
                    [
                        oh.make_node("Unsqueeze", ["X", "first_axes"], ["first"]),
                        oh.make_node("Unsqueeze", ["first", "second_axes"], ["Y"]),
                    ],
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", "b"])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                    [
                        _initializer("first_axes", first_axes, np.int64),
                        _initializer("second_axes", second_axes, np.int64),
                    ],
                )

                optimized, rewrites = _optimize(model, "UnsqueezeUnsqueeze")

                self.assertPatternRewritten(rewrites, "UnsqueezeUnsqueeze")
                self.assertEqual(["Unsqueeze"], _ops(optimized))
                self.assertEquivalent(model, optimized, feeds)

    def test_unsqueeze_unsqueeze_shared_and_no_match_cases(self):
        shared_model = _make_model(
            [
                oh.make_node("Unsqueeze", ["X", "first_axes"], ["first"]),
                oh.make_node("Unsqueeze", ["first", "second_axes"], ["Y"]),
                oh.make_node("Identity", ["first"], ["other"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
            [
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 1, 1]),
                oh.make_tensor_value_info("other", TensorProto.FLOAT, [2, 3, 1]),
            ],
            [
                _initializer("first_axes", [2], np.int64),
                _initializer("second_axes", [3], np.int64),
            ],
        )
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}

        optimized, rewrites = _optimize(shared_model, "UnsqueezeUnsqueeze")

        self.assertPatternRewritten(rewrites, "UnsqueezeUnsqueeze")
        self.assertEqual(["Unsqueeze", "Unsqueeze", "Identity"], _ops(optimized))
        self.assertEquivalent(shared_model, optimized, feeds)

        dynamic_model = _make_model(
            [
                oh.make_node("Unsqueeze", ["X", "first_axes"], ["first"]),
                oh.make_node("Unsqueeze", ["first", "second_axes"], ["Y"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
                oh.make_tensor_value_info("first_axes", TensorProto.INT64, [1]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
            [_initializer("second_axes", [3], np.int64)],
        )

        dynamic_optimized, dynamic_rewrites = _optimize(dynamic_model, "UnsqueezeUnsqueeze")

        self.assertPatternNotRewritten(dynamic_rewrites, "UnsqueezeUnsqueeze")
        self.assertEqual(["Unsqueeze", "Unsqueeze"], _ops(dynamic_optimized))

    def test_squeeze_unsqueeze_upstream_variants(self):
        cases = (
            ("squeeze_unsqueeze", "Squeeze", [1], "Unsqueeze", [1], [3, 1, 5], "Identity"),
            (
                "squeeze_unsqueeze_two",
                "Squeeze",
                [1, 2],
                "Unsqueeze",
                [1, 2],
                [3, 1, 1, 5],
                "Identity",
            ),
            (
                "unsqueeze_squeeze",
                "Unsqueeze",
                [1, 2],
                "Squeeze",
                [1, 2],
                [3, 1, 1, 5],
                "Identity",
            ),
            ("subset", "Unsqueeze", [1], "Squeeze", [1, 2], [3, 1, 5], "Squeeze"),
        )
        for name, first_op, first_axes, second_op, second_axes, shape, expected_op in cases:
            with self.subTest(name=name):
                model = _make_model(
                    [
                        oh.make_node(first_op, ["X", "first_axes"], ["first"]),
                        oh.make_node(second_op, ["first", "second_axes"], ["Y"]),
                    ],
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                    [
                        _initializer("first_axes", first_axes, np.int64),
                        _initializer("second_axes", second_axes, np.int64),
                    ],
                )
                feeds = {"X": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}

                optimized, rewrites = _optimize(model, "SqueezeUnsqueeze")

                self.assertPatternRewritten(rewrites, "SqueezeUnsqueeze")
                self.assertEqual([expected_op], _ops(optimized))
                self.assertEquivalent(model, optimized, feeds)

    def test_squeeze_without_axes_then_unsqueeze(self):
        model = _make_model(
            [
                oh.make_node("Squeeze", ["X"], ["squeezed"]),
                oh.make_node("Unsqueeze", ["squeezed", "axes"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.INT64, [1])],
            [oh.make_tensor_value_info("Y", TensorProto.INT64, [1])],
            [_initializer("axes", [0], np.int64)],
        )

        optimized, rewrites = _optimize(model, "SqueezeUnsqueeze")

        self.assertPatternRewritten(rewrites, "SqueezeUnsqueeze")
        self.assertEqual(["Identity"], _ops(optimized))
        self.assertEquivalent(model, optimized, {"X": np.array([5], dtype=np.int64)})

    def test_squeeze_unsqueeze_shared_and_disjoint_axes(self):
        shared_model = _make_model(
            [
                oh.make_node("Unsqueeze", ["X", "axes"], ["first"]),
                oh.make_node("Squeeze", ["first", "axes"], ["Y"]),
                oh.make_node("Identity", ["first"], ["other"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
            [
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3]),
                oh.make_tensor_value_info("other", TensorProto.FLOAT, [2, 3, 1]),
            ],
            [_initializer("axes", [2], np.int64)],
        )
        feeds = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}

        optimized, rewrites = _optimize(shared_model, "SqueezeUnsqueeze")

        self.assertPatternRewritten(rewrites, "SqueezeUnsqueeze")
        self.assertEqual(["Unsqueeze", "Identity", "Identity"], _ops(optimized))
        self.assertEquivalent(shared_model, optimized, feeds)

        no_match_model = _make_model(
            [
                oh.make_node("Unsqueeze", ["X", "unsqueeze_axes"], ["first"]),
                oh.make_node("Squeeze", ["first", "squeeze_axes"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1])],
            [
                _initializer("unsqueeze_axes", [2], np.int64),
                _initializer("squeeze_axes", [1], np.int64),
            ],
        )

        no_match_optimized, no_match_rewrites = _optimize(no_match_model, "SqueezeUnsqueeze")

        self.assertPatternNotRewritten(no_match_rewrites, "SqueezeUnsqueeze")
        self.assertEqual(["Unsqueeze", "Squeeze"], _ops(no_match_optimized))

    def test_swap_unsqueeze_transpose_upstream_subtests(self):
        cases = [([axis], [0, 2, 1, 3]) for axis in (0, 1, 2, -1)] + [
            (axes, [0, 2, 1, 4, 3]) for axes in ([0, 1], [1, 2], [0, 2])
        ]
        feeds = {"X": np.arange(84, dtype=np.float32).reshape(4, 3, 7)}
        for axes, perm in cases:
            with self.subTest(axes=axes):
                model = _make_model(
                    [
                        oh.make_node("Unsqueeze", ["X", "axes"], ["unsqueezed"]),
                        oh.make_node("Transpose", ["unsqueezed"], ["Y"], perm=perm),
                    ],
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", "b", "c"])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                    [_initializer("axes", axes, np.int64)],
                )

                optimized, rewrites = _optimize(model, "SwapUnsqueezeTranspose")

                self.assertPatternRewritten(rewrites, "SwapUnsqueezeTranspose")
                self.assertEqual(["Transpose", "Unsqueeze"], _ops(optimized))
                self.assertEquivalent(model, optimized, feeds)

    def test_swap_unsqueeze_transpose_rejects_shared_unsqueeze(self):
        model = _make_model(
            [
                oh.make_node("Unsqueeze", ["X", "axes"], ["unsqueezed"]),
                oh.make_node("Transpose", ["unsqueezed"], ["Y"], perm=[0, 2, 1, 3]),
                oh.make_node("Identity", ["unsqueezed"], ["other"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, None),
                oh.make_tensor_value_info("other", TensorProto.FLOAT, None),
            ],
            [_initializer("axes", [1], np.int64)],
        )

        optimized, rewrites = _optimize(model, "SwapUnsqueezeTranspose")

        self.assertPatternNotRewritten(rewrites, "SwapUnsqueezeTranspose")
        self.assertEqual(["Unsqueeze", "Transpose", "Identity"], _ops(optimized))

    def test_transpose_equal_reshape_upstream_shapes(self):
        for batch in (3, 0):
            with self.subTest(batch=batch):
                model = _make_model(
                    [
                        oh.make_node("Transpose", ["X"], ["transposed"], perm=[0, 2, 1, 3]),
                        oh.make_node("Add", ["transposed", "one"], ["Y"]),
                    ],
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 2, 1, 5])],
                    [oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", 1, 2, 5])],
                    [_initializer("one", [1.0], np.float32)],
                )
                feeds = {"X": np.arange(batch * 10, dtype=np.float32).reshape(batch, 2, 1, 5)}

                optimized, rewrites = _optimize(model, "TransposeEqualReshape")

                self.assertPatternRewritten(rewrites, "TransposeEqualReshape")
                self.assertEqual(["Reshape", "Add"], _ops(optimized))
                shape_name = optimized.graph.node[0].input[1]
                self.assertEqual(
                    [0, 1, 2, 0], _initializer_values(optimized, shape_name).tolist()
                )
                self.assertEquivalent(model, optimized, feeds)

    def test_transpose_equal_reshape_no_match_for_moved_data_axes(self):
        model = _make_model(
            [oh.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1, 3])],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2, 4, 5])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 2, 5])],
        )

        optimized, rewrites = _optimize(model, "TransposeEqualReshape")

        self.assertPatternNotRewritten(rewrites, "TransposeEqualReshape")
        self.assertEqual(["Transpose"], _ops(optimized))

    def test_transpose_reshape_transpose_after(self):
        model = _make_model(
            [
                oh.make_node("Transpose", ["X"], ["first"], perm=[0, 2, 3, 1]),
                oh.make_node("Reshape", ["first", "shape"], ["reshaped"]),
                oh.make_node("Transpose", ["reshaped"], ["Y"], perm=[0, 2, 1]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4, 5])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 20])],
            [_initializer("shape", [2, 20, 3], np.int64)],
        )
        feeds = {"X": np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)}

        optimized, rewrites = _optimize(model, "TransposeReshapeTranspose")

        self.assertPatternRewritten(rewrites, "TransposeReshapeTranspose")
        self.assertEqual(["Transpose", "Transpose", "Reshape"], _ops(optimized))
        self.assertEqual([0, 3, 1, 2], _attribute_ints(optimized.graph.node[1], "perm"))
        shape_name = optimized.graph.node[2].input[1]
        self.assertEqual([2, 3, 20], _initializer_values(optimized, shape_name).tolist())
        self.assertEquivalent(model, optimized, feeds)

    def test_transpose_reshape_transpose_before(self):
        model = _make_model(
            [
                oh.make_node("Transpose", ["X"], ["first"], perm=[0, 2, 3, 1]),
                oh.make_node("Reshape", ["first", "shape"], ["reshaped"]),
                oh.make_node("Transpose", ["reshaped"], ["Y"], perm=[0, 1, 3, 2, 4, 5]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 6, 4, 4])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 2, 2, 2, 6])],
            [_initializer("shape", [2, 2, 2, 2, 2, 6], np.int64)],
        )
        feeds = {"X": np.arange(192, dtype=np.float32).reshape(2, 6, 4, 4)}

        optimized, rewrites = _optimize(model, "TransposeReshapeTranspose")

        self.assertPatternRewritten(rewrites, "TransposeReshapeTranspose")
        self.assertEqual(["Reshape", "Transpose", "Transpose"], _ops(optimized))
        shape_name = optimized.graph.node[0].input[1]
        self.assertEqual([2, 6, 2, 2, 2, 2], _initializer_values(optimized, shape_name).tolist())
        self.assertEqual([0, 2, 3, 4, 5, 1], _attribute_ints(optimized.graph.node[1], "perm"))
        self.assertEquivalent(model, optimized, feeds)

    def test_transpose_reshape_transpose_rejects_dynamic_shape(self):
        model = _make_model(
            [
                oh.make_node("Transpose", ["X"], ["first"], perm=[0, 2, 3, 1]),
                oh.make_node("Reshape", ["first", "shape"], ["reshaped"]),
                oh.make_node("Transpose", ["reshaped"], ["Y"], perm=[0, 2, 1]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4, 5]),
                oh.make_tensor_value_info("shape", TensorProto.INT64, [3]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 20])],
        )

        optimized, rewrites = _optimize(
            model, "TransposeReshapeTranspose", run_shape_inference=False
        )

        self.assertPatternNotRewritten(rewrites, "TransposeReshapeTranspose")
        self.assertEqual(["Transpose", "Reshape", "Transpose"], _ops(optimized))

    def test_mul_unsqueeze_unsqueeze_and_no_match_cases(self):
        positive = _make_model(
            [
                oh.make_node("Unsqueeze", ["X", "axes"], ["left"]),
                oh.make_node("Unsqueeze", ["Y", "axes"], ["right"]),
                oh.make_node("Mul", ["left", "right"], ["Z"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4]),
            ],
            [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4, 1])],
            [_initializer("axes", [2], np.int64)],
        )
        feeds = {
            "X": np.arange(12, dtype=np.float32).reshape(3, 4),
            "Y": np.arange(12, dtype=np.float32).reshape(3, 4) + 1,
        }

        optimized, rewrites = _optimize(positive, "MulUnsqueezeUnsqueeze")

        self.assertPatternRewritten(rewrites, "MulUnsqueezeUnsqueeze")
        self.assertEqual(["Mul", "Unsqueeze"], _ops(optimized))
        self.assertEquivalent(positive, optimized, feeds)

        for case in ("different_axes", "shared"):
            with self.subTest(case=case):
                nodes = [
                    oh.make_node("Unsqueeze", ["X", "left_axes"], ["left"]),
                    oh.make_node("Unsqueeze", ["Y", "right_axes"], ["right"]),
                    oh.make_node("Mul", ["left", "right"], ["Z"]),
                ]
                outputs = [oh.make_tensor_value_info("Z", TensorProto.FLOAT, None)]
                if case == "shared":
                    nodes.append(oh.make_node("Identity", ["left"], ["other"]))
                    outputs.append(oh.make_tensor_value_info("other", TensorProto.FLOAT, None))
                model = _make_model(
                    nodes,
                    [
                        oh.make_tensor_value_info("X", TensorProto.FLOAT, [3]),
                        oh.make_tensor_value_info("Y", TensorProto.FLOAT, [3]),
                    ],
                    outputs,
                    [
                        _initializer("left_axes", [0], np.int64),
                        _initializer(
                            "right_axes", [1] if case == "different_axes" else [0], np.int64
                        ),
                    ],
                )

                no_match, no_match_rewrites = _optimize(model, "MulUnsqueezeUnsqueeze")

                self.assertPatternNotRewritten(no_match_rewrites, "MulUnsqueezeUnsqueeze")
                self.assertEqual(
                    [
                        "Unsqueeze",
                        "Unsqueeze",
                        "Mul",
                        *(["Identity"] if case == "shared" else []),
                    ],
                    _ops(no_match),
                )

    def test_squeeze_add_upstream_axes_variants(self):
        for explicit_axes in (True, False):
            with self.subTest(explicit_axes=explicit_axes):
                squeeze_inputs = ["S1", "axes"] if explicit_axes else ["S1"]
                second_inputs = ["S2", "axes"] if explicit_axes else ["S2"]
                model = _make_model(
                    [
                        oh.make_node("Squeeze", squeeze_inputs, ["left"]),
                        oh.make_node("Squeeze", second_inputs, ["right"]),
                        oh.make_node("Add", ["left", "right"], ["sum"]),
                        oh.make_node("Unsqueeze", ["sum", "axes"], ["Y"]),
                    ],
                    [
                        oh.make_tensor_value_info("S1", TensorProto.INT64, [1]),
                        oh.make_tensor_value_info("S2", TensorProto.INT64, [1]),
                    ],
                    [oh.make_tensor_value_info("Y", TensorProto.INT64, [1])],
                    [_initializer("axes", [0], np.int64)],
                )
                feeds = {"S1": np.array([5], dtype=np.int64), "S2": np.array([7], dtype=np.int64)}

                optimized, rewrites = _optimize(model, "SqueezeAdd")

                self.assertPatternRewritten(rewrites, "SqueezeAdd")
                self.assertEqual(["Add", "Squeeze", "Unsqueeze"], _ops(optimized))
                self.assertEquivalent(model, optimized, feeds)

    def test_squeeze_add_rejects_different_axes(self):
        model = _make_model(
            [
                oh.make_node("Squeeze", ["X", "first_axes"], ["left"]),
                oh.make_node("Squeeze", ["Y", "second_axes"], ["right"]),
                oh.make_node("Add", ["left", "right"], ["Z"]),
            ],
            [
                oh.make_tensor_value_info("X", TensorProto.INT64, [1]),
                oh.make_tensor_value_info("Y", TensorProto.INT64, [1, 1]),
            ],
            [oh.make_tensor_value_info("Z", TensorProto.INT64, None)],
            [
                _initializer("first_axes", [0], np.int64),
                _initializer("second_axes", [1], np.int64),
            ],
        )

        optimized, rewrites = _optimize(model, "SqueezeAdd")

        self.assertPatternNotRewritten(rewrites, "SqueezeAdd")
        self.assertEqual(["Squeeze", "Squeeze", "Add"], _ops(optimized))

    def test_squeeze_binary_unsqueeze_upstream_graph(self):
        model = _make_model(
            [
                oh.make_node("Shape", ["X"], ["dimension"], start=0, end=1),
                oh.make_node("Squeeze", ["dimension"], ["scalar_dimension"]),
                oh.make_node("Div", ["scalar_dimension", "two"], ["quotient"]),
                oh.make_node("Unsqueeze", ["quotient", "zero"], ["Y"]),
            ],
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", "b"])],
            [oh.make_tensor_value_info("Y", TensorProto.INT64, [1])],
            [_initializer("zero", [0], np.int64), _initializer("two", 2, np.int64)],
        )
        feeds = {"X": np.arange(12, dtype=np.float32).reshape(3, 4)}

        optimized, rewrites = _optimize(model, "SqueezeBinaryUnsqueeze")

        self.assertPatternRewritten(rewrites, "SqueezeBinaryUnsqueeze")
        self.assertEqual(["Shape", "Div"], _ops(optimized))
        self.assertEquivalent(model, optimized, feeds)

    def test_squeeze_binary_unsqueeze_rejects_non_scalar_or_shared_values(self):
        for case in ("non_scalar", "shared"):
            with self.subTest(case=case):
                nodes = [
                    oh.make_node("Squeeze", ["X"], ["squeezed"]),
                    oh.make_node("Div", ["squeezed", "two"], ["quotient"]),
                    oh.make_node("Unsqueeze", ["quotient", "zero"], ["Y"]),
                ]
                outputs = [oh.make_tensor_value_info("Y", TensorProto.INT64, [1])]
                if case == "shared":
                    nodes.append(oh.make_node("Identity", ["quotient"], ["other"]))
                    outputs.append(oh.make_tensor_value_info("other", TensorProto.INT64, []))
                model = _make_model(
                    nodes,
                    [oh.make_tensor_value_info("X", TensorProto.INT64, [1])],
                    outputs,
                    [
                        _initializer("zero", [0], np.int64),
                        _initializer("two", [2] if case == "non_scalar" else 2, np.int64),
                    ],
                )

                optimized, rewrites = _optimize(model, "SqueezeBinaryUnsqueeze")

                self.assertPatternNotRewritten(rewrites, "SqueezeBinaryUnsqueeze")
                self.assertEqual(
                    ["Squeeze", "Div", "Unsqueeze", *(["Identity"] if case == "shared" else [])],
                    _ops(optimized),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
