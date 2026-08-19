# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests algebraic graph patterns migrated from YOBX xoptim."""

from __future__ import annotations

import unittest

import numpy as np

from onnx_light.onnx import TensorProto, helper, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core import optimization


def _value(name: str, data_type: int, shape: list[int | str]):
    return helper.make_tensor_value_info(name, data_type, shape)


def _initializer(name: str, value):
    return numpy_helper.from_array(np.asarray(value), name=name)


def _model(nodes, inputs, outputs, initializers=(), *, opset: int = 18, ir_version: int = 10):
    graph = helper.make_graph(
        list(nodes), "algebra-pattern", list(inputs), list(outputs), list(initializers)
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=ir_version
    )


def _range(*shape: int, bias: float = 0.0, dtype=np.float32):
    size = int(np.prod(shape))
    return (np.arange(size, dtype=np.float32).reshape(shape) / max(size, 1) + bias).astype(dtype)


def _op_types(model) -> list[str]:
    return [node.op_type for node in model.graph.node]


def _node_inputs(model) -> list[tuple[str, ...]]:
    return [tuple(node.input) for node in model.graph.node]


def _rewrite_count(rewrites, pattern_name: str) -> int:
    return sum(rewrite.pattern_name == pattern_name for rewrite in rewrites)


def _run(model, feeds):
    return ReferenceEvaluator(model).run(None, feeds)


def _optimize(model, pattern_name: str, *, report: bool = False):
    builder = optimization.GraphBuilder(model)
    pattern = getattr(optimization, f"{pattern_name}Pattern")()
    graph = optimization.GraphGraph(builder, [pattern], use_global_patterns=False)
    result = graph.optimize(report=report)
    return builder.to_onnx("model"), result


class TestPatternsAlgebra(unittest.TestCase):
    def assert_equivalent(self, original, optimized, feeds, *, atol=1e-6):
        expected = _run(original, feeds)
        got = _run(optimized, feeds)
        self.assertEqual(len(expected), len(got))
        for expected_value, got_value in zip(expected, got):
            if np.issubdtype(expected_value.dtype, np.inexact):
                np.testing.assert_allclose(expected_value, got_value, atol=atol, rtol=atol)
            else:
                np.testing.assert_array_equal(expected_value, got_value)

    @staticmethod
    def _mul_mul_mul_model(inner_op: str):
        return _model(
            [
                helper.make_node(inner_op, ["X", "cst1"], ["xc"]),
                helper.make_node(inner_op, ["Y", "cst2"], ["yc"]),
                helper.make_node("Mul", ["xc", "yc"], ["Z"]),
            ],
            [
                _value("X", TensorProto.FLOAT, ["a", "b"]),
                _value("Y", TensorProto.FLOAT, ["a", "b"]),
            ],
            [_value("Z", TensorProto.FLOAT, ["a", "b"])],
            [
                _initializer("cst1", np.array([2], dtype=np.float32)),
                _initializer("cst2", np.array([3], dtype=np.float32)),
            ],
        )

    def test_mul_mul_mul_scalar_variants_and_initializers(self):
        feeds = {"X": _range(3, 3), "Y": _range(3, 3, bias=0.25)}
        for inner_op, expected_ops in (("Mul", ["Mul", "Mul"]), ("Div", ["Mul", "Div"])):
            with self.subTest(inner_op=inner_op):
                model = self._mul_mul_mul_model(inner_op)
                optimized, rewrites = _optimize(model, "MulMulMulScalar")

                self.assertEqual(expected_ops, _op_types(optimized))
                self.assertEqual(3, len(optimized.graph.initializer))
                combined = [
                    initializer
                    for initializer in optimized.graph.initializer
                    if initializer.name.startswith("MulMulMulScalarPattern.cst")
                ]
                self.assertEqual(1, len(combined))
                np.testing.assert_allclose(
                    numpy_helper.to_array(combined[0]), np.array([6], dtype=np.float32)
                )
                self.assertEqual(1, _rewrite_count(rewrites, "MulMulMulScalar"))
                self.assert_equivalent(model, optimized, feeds, atol=1e-5)

    def test_mul_mul_mul_scalar_statistics(self):
        model = self._mul_mul_mul_model("Div")
        optimized, result = _optimize(model, "MulMulMulScalar", report=True)
        rewrites, report = result
        pattern_report = report.patterns[0]

        self.assertEqual(["Mul", "Div"], _op_types(optimized))
        self.assertEqual(1, _rewrite_count(rewrites, "MulMulMulScalar"))
        self.assertEqual(1, report.rewrites)
        self.assertEqual("MulMulMulScalar", pattern_report.pattern_name)
        self.assertEqual(1, pattern_report.matches)
        self.assertGreaterEqual(pattern_report.attempts, 1)
        self.assertGreaterEqual(report.iterations, 1)
        self.assertGreaterEqual(sum(item.occurrences for item in pattern_report.no_matches), 1)

    def test_mul_mul_mul_scalar_shared_inner_output_no_match(self):
        model = _model(
            [
                helper.make_node("Mul", ["X", "cst1"], ["xc"]),
                helper.make_node("Mul", ["Y", "cst2"], ["yc"]),
                helper.make_node("Mul", ["xc", "yc"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, [2, 3]), _value("Y", TensorProto.FLOAT, [2, 3])],
            [_value("Z", TensorProto.FLOAT, [2, 3]), _value("xc", TensorProto.FLOAT, [2, 3])],
            [
                _initializer("cst1", np.array([2], dtype=np.float32)),
                _initializer("cst2", np.array([3], dtype=np.float32)),
            ],
        )
        feeds = {"X": _range(2, 3), "Y": _range(2, 3, bias=0.5)}
        optimized, rewrites = _optimize(model, "MulMulMulScalar")

        self.assertEqual(["Mul", "Mul", "Mul"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "MulMulMulScalar"))
        self.assert_equivalent(model, optimized, feeds)

    @staticmethod
    def _sub1_mul_model(side: str):
        subtracted = "X" if side == "left" else "Y"
        mul_inputs = ["i1", "Y"] if side == "left" else ["X", "i1"]
        return _model(
            [
                helper.make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["one"],
                    value=_initializer("", np.array([1], dtype=np.float32)),
                ),
                helper.make_node("Sub", ["one", subtracted], ["i1"]),
                helper.make_node("Mul", mul_inputs, ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, ["a", 6]), _value("Y", TensorProto.FLOAT, ["a", 6])],
            [_value("Z", TensorProto.FLOAT, ["a", 6])],
            [_initializer("shape", np.array([1, 6], dtype=np.int64))],
        )

    def test_sub1_mul_upstream_constant_of_shape_left_and_right(self):
        feeds = {"X": _range(2, 6), "Y": _range(2, 6, bias=0.5)}
        for side in ("left", "right"):
            with self.subTest(side=side):
                model = self._sub1_mul_model(side)
                optimized, rewrites = _optimize(model, "Sub1Mul")

                self.assertEqual(["Mul", "Sub"], _op_types(optimized))
                self.assertEqual(1, len(optimized.graph.initializer))
                self.assertEqual(1, _rewrite_count(rewrites, "Sub1Mul"))
                self.assert_equivalent(model, optimized, feeds, atol=1e-5)

    def test_sub1_mul_materialized_initializer_left_and_right(self):
        feeds = {"X": _range(2, 6), "Y": _range(2, 6, bias=0.5)}
        for side in ("left", "right"):
            with self.subTest(side=side):
                subtracted = "X" if side == "left" else "Y"
                mul_inputs = ["i1", "Y"] if side == "left" else ["X", "i1"]
                model = _model(
                    [
                        helper.make_node("Sub", ["one", subtracted], ["i1"]),
                        helper.make_node("Mul", mul_inputs, ["Z"]),
                    ],
                    [
                        _value("X", TensorProto.FLOAT, ["a", 6]),
                        _value("Y", TensorProto.FLOAT, ["a", 6]),
                    ],
                    [_value("Z", TensorProto.FLOAT, ["a", 6])],
                    [_initializer("one", np.array([1], dtype=np.float32))],
                )
                optimized, rewrites = _optimize(model, "Sub1Mul")

                self.assertEqual(["Mul", "Sub"], _op_types(optimized))
                self.assertEqual(1, len(optimized.graph.initializer))
                self.assertEqual(1, _rewrite_count(rewrites, "Sub1Mul"))
                self.assert_equivalent(model, optimized, feeds, atol=1e-5)

    def test_sub1_mul_shared_sub_output_no_match(self):
        model = _model(
            [
                helper.make_node("Sub", ["one", "X"], ["one_minus"]),
                helper.make_node("Mul", ["one_minus", "Y"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, [2, 3]), _value("Y", TensorProto.FLOAT, [2, 3])],
            [
                _value("Z", TensorProto.FLOAT, [2, 3]),
                _value("one_minus", TensorProto.FLOAT, [2, 3]),
            ],
            [_initializer("one", np.array([1], dtype=np.float32))],
        )
        feeds = {"X": _range(2, 3), "Y": _range(2, 3, bias=0.25)}
        optimized, rewrites = _optimize(model, "Sub1Mul")

        self.assertEqual(["Sub", "Mul"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "Sub1Mul"))
        self.assert_equivalent(model, optimized, feeds)

    def test_sub1_mul_constant_of_shape_wrong_value_no_match(self):
        model = self._sub1_mul_model("left")
        model.graph.node[0].attribute[0].t.CopyFrom(
            _initializer("", np.array([2], dtype=np.float32))
        )
        feeds = {"X": _range(2, 6), "Y": _range(2, 6, bias=0.5)}
        optimized, rewrites = _optimize(model, "Sub1Mul")

        self.assertEqual(["ConstantOfShape", "Sub", "Mul"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "Sub1Mul"))
        self.assert_equivalent(model, optimized, feeds)

    def test_sub1_mul_constant_of_shape_broadcast_rank_no_match(self):
        model = _model(
            [
                helper.make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["one"],
                    value=_initializer("", np.array([1], dtype=np.float32)),
                ),
                helper.make_node("Sub", ["one", "X"], ["i1"]),
                helper.make_node("Mul", ["i1", "Y"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, [3]), _value("Y", TensorProto.FLOAT, [3])],
            [_value("Z", TensorProto.FLOAT, [2, 1, 3])],
            [_initializer("shape", np.array([2, 1, 3], dtype=np.int64))],
        )
        feeds = {"X": _range(3), "Y": _range(3, bias=0.5)}
        optimized, rewrites = _optimize(model, "Sub1Mul")

        self.assertEqual(["ConstantOfShape", "Sub", "Mul"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "Sub1Mul"))
        self.assert_equivalent(model, optimized, feeds)

    def test_sub1_mul_preserves_external_constant_of_shape_output(self):
        model = self._sub1_mul_model("left")
        model.graph.output.append(_value("one", TensorProto.FLOAT, [1, 6]))
        feeds = {"X": _range(2, 6), "Y": _range(2, 6, bias=0.5)}
        optimized, rewrites = _optimize(model, "Sub1Mul")

        self.assertEqual(["ConstantOfShape", "Mul", "Sub"], _op_types(optimized))
        self.assertEqual(1, _rewrite_count(rewrites, "Sub1Mul"))
        self.assert_equivalent(model, optimized, feeds, atol=1e-5)

    @staticmethod
    def _switch_order_model(side: int, private_method_shapes: bool = False):
        if private_method_shapes:
            outer = [2, 1, 1024]
            broad = [2, 1024, 1024]
            tail = [1024]
        else:
            outer = ["a", 1, 3, 4]
            broad = ["a", 2, 3, 4]
            tail = ["a", 1, 3, 4]
        nodes = [helper.make_node("Add", ["B", "C"], ["bc"])]
        nodes.append(helper.make_node("Add", ["bc", "A"] if side == 0 else ["A", "bc"], ["F"]))
        return _model(
            nodes,
            [
                _value("A", TensorProto.FLOAT, outer),
                _value("B", TensorProto.FLOAT, broad),
                _value("C", TensorProto.FLOAT, tail),
            ],
            [_value("F", TensorProto.FLOAT, broad)],
        )

    def test_switch_order_binary_left_and_right(self):
        feeds = {
            "A": _range(2, 1, 3, 4),
            "B": _range(2, 2, 3, 4, bias=0.25),
            "C": _range(2, 1, 3, 4, bias=0.5),
        }
        for side in (0, 1):
            with self.subTest(side=side):
                model = self._switch_order_model(side)
                original_inputs = _node_inputs(model)
                optimized, rewrites = _optimize(model, "SwitchOrderBinary")

                self.assertEqual(["Add", "Add"], _op_types(optimized))
                self.assertEqual(0, len(optimized.graph.initializer))
                self.assertNotEqual(original_inputs, _node_inputs(optimized))
                self.assertEqual({"A", "C"}, set(optimized.graph.node[0].input))
                self.assertEqual(1, _rewrite_count(rewrites, "SwitchOrderBinary"))
                self.assert_equivalent(model, optimized, feeds)

    def test_switch_order_private_helper_intent_on_graphs(self):
        feeds = {
            "A": np.ones((2, 1, 1024), dtype=np.float32),
            "B": np.ones((2, 1024, 1024), dtype=np.float32),
            "C": np.ones((1024,), dtype=np.float32),
        }
        for side in (0, 1):
            with self.subTest(side=side):
                model = self._switch_order_model(side, private_method_shapes=True)
                optimized, rewrites = _optimize(model, "SwitchOrderBinary")

                self.assertEqual(["Add", "Add"], _op_types(optimized))
                self.assertEqual({"A", "C"}, set(optimized.graph.node[0].input))
                self.assertIn("B", optimized.graph.node[1].input)
                self.assertEqual(1, _rewrite_count(rewrites, "SwitchOrderBinary"))
                self.assert_equivalent(model, optimized, feeds)

    def test_switch_order_binary_shared_inner_output_no_match(self):
        model = self._switch_order_model(0)
        model.graph.output.append(_value("bc", TensorProto.FLOAT, ["a", 2, 3, 4]))
        feeds = {
            "A": _range(2, 1, 3, 4),
            "B": _range(2, 2, 3, 4, bias=0.25),
            "C": _range(2, 1, 3, 4, bias=0.5),
        }
        optimized, rewrites = _optimize(model, "SwitchOrderBinary")

        self.assertEqual(_node_inputs(model), _node_inputs(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "SwitchOrderBinary"))
        self.assert_equivalent(model, optimized, feeds)

    @staticmethod
    def _same_children_model(case: str):
        prefix = [helper.make_node("Add", ["X", "Y"], ["xy"])]
        output_type = TensorProto.FLOAT16
        if case == "two_casts":
            nodes = [
                *prefix,
                helper.make_node("Cast", ["xy"], ["xy1"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["xy"], ["xy2"], to=TensorProto.FLOAT16),
                helper.make_node("Add", ["xy1", "xy2"], ["Z"]),
            ]
            expected_ops = ["Add", "Cast", "Add"]
        elif case == "commutative_add":
            nodes = [
                helper.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["Y"], ["yc"], to=TensorProto.FLOAT16),
                helper.make_node("Add", ["xc", "yc"], ["xy"]),
                helper.make_node("Add", ["yc", "xc"], ["xy2"]),
                helper.make_node("Add", ["xy", "xy2"], ["Z"]),
            ]
            expected_ops = ["Cast", "Cast", "Add", "Add"]
        elif case == "three_casts":
            nodes = [
                *prefix,
                helper.make_node("Cast", ["xy"], ["xy1"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["xy"], ["xy2"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["xy"], ["xy3"], to=TensorProto.FLOAT16),
                helper.make_node("Add", ["xy1", "xy2"], ["xy12"]),
                helper.make_node("Add", ["xy12", "xy3"], ["Z"]),
            ]
            expected_ops = ["Add", "Cast", "Add", "Add"]
        elif case == "two_exp":
            nodes = [
                *prefix,
                helper.make_node("Cast", ["xy"], ["xy1"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["xy"], ["xy2"], to=TensorProto.FLOAT16),
                helper.make_node("Exp", ["xy1"], ["e1"]),
                helper.make_node("Exp", ["xy2"], ["e2"]),
                helper.make_node("Add", ["e1", "e2"], ["Z"]),
            ]
            expected_ops = ["Add", "Cast", "Exp", "Add"]
        elif case == "four_exp":
            nodes = [
                *prefix,
                helper.make_node("Cast", ["xy"], ["xy1"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["xy"], ["xy2"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["xy"], ["xy3"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["xy"], ["xy4"], to=TensorProto.FLOAT16),
                helper.make_node("Exp", ["xy1"], ["e1"]),
                helper.make_node("Exp", ["xy2"], ["e2"]),
                helper.make_node("Exp", ["xy3"], ["e3"]),
                helper.make_node("Exp", ["xy4"], ["e4"]),
                helper.make_node("Add", ["e1", "e2"], ["e12"]),
                helper.make_node("Add", ["e3", "e4"], ["e34"]),
                helper.make_node("Add", ["e12", "e34"], ["Z"]),
            ]
            expected_ops = ["Add", "Cast", "Exp", "Add", "Add"]
        else:
            raise AssertionError(case)
        return (
            _model(
                nodes,
                [
                    _value("X", TensorProto.FLOAT, ["a", 2, 3, 4]),
                    _value("Y", TensorProto.FLOAT, ["a", 1, 3, 4]),
                ],
                [_value("Z", output_type, ["a", 2, 3, 4])],
            ),
            expected_ops,
        )

    def test_same_children_topology_variants(self):
        feeds = {"X": _range(2, 3, 4), "Y": _range(1, 3, 4)}
        for case in ("two_casts", "commutative_add", "three_casts", "two_exp", "four_exp"):
            with self.subTest(case=case):
                model, expected_ops = self._same_children_model(case)
                original_inputs = _node_inputs(model)
                optimized, rewrites = _optimize(model, "SameChildren")

                self.assertEqual(expected_ops, _op_types(optimized))
                self.assertEqual(0, len(optimized.graph.initializer))
                self.assertNotEqual(original_inputs, _node_inputs(optimized))
                self.assertGreaterEqual(_rewrite_count(rewrites, "SameChildren"), 1)
                self.assert_equivalent(model, optimized, feeds, atol=1e-3)

    def test_same_children_from_graph_input(self):
        model = _model(
            [
                helper.make_node("Cast", ["X"], ["xy1"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["X"], ["xy2"], to=TensorProto.FLOAT16),
                helper.make_node("Add", ["xy1", "xy2"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, ["a", 2, 3, 4])],
            [_value("Z", TensorProto.FLOAT16, ["a", 2, 3, 4])],
        )
        feeds = {"X": _range(2, 3, 4)}
        original_inputs = _node_inputs(model)
        optimized, rewrites = _optimize(model, "SameChildrenFromInput")

        self.assertEqual(["Cast", "Add"], _op_types(optimized))
        self.assertEqual(0, len(optimized.graph.initializer))
        self.assertNotEqual(original_inputs, _node_inputs(optimized))
        self.assertEqual(1, _rewrite_count(rewrites, "SameChildrenFromInput"))
        self.assert_equivalent(model, optimized, feeds)

    def test_same_children_many_duplicated(self):
        nodes = [
            helper.make_node("Add", ["X", "one"], ["x1"]),
            helper.make_node("Add", ["x1", "one"], ["x11"]),
            helper.make_node("Add", ["x1", "one"], ["x12"]),
            helper.make_node("Add", ["x11", "one"], ["x111"]),
            helper.make_node("Add", ["x11", "one"], ["x112"]),
            helper.make_node("Add", ["x12", "one"], ["x121"]),
            helper.make_node("Add", ["x12", "one"], ["x122"]),
            helper.make_node("Add", ["x111", "one"], ["x1111"]),
            helper.make_node("Add", ["x111", "one"], ["x1112"]),
            helper.make_node("Add", ["x112", "one"], ["x1121"]),
            helper.make_node("Add", ["x112", "one"], ["x1122"]),
            helper.make_node("Add", ["x121", "one"], ["x1211"]),
            helper.make_node("Add", ["x121", "one"], ["x1212"]),
            helper.make_node("Add", ["x122", "one"], ["x1221"]),
            helper.make_node("Add", ["x122", "one"], ["x1222"]),
            helper.make_node(
                "Sum",
                ["x1111", "x1112", "x1121", "x1122", "x1211", "x1212", "x1221", "x1222"],
                ["Z"],
            ),
        ]
        model = _model(
            nodes,
            [_value("X", TensorProto.FLOAT, ["a", "b", "c"])],
            [_value("Z", TensorProto.FLOAT, ["a", "b", "c"])],
            [_initializer("one", np.array([1], dtype=np.float32))],
        )
        feeds = {"X": _range(2, 3, 4)}
        optimized, rewrites = _optimize(model, "SameChildren")

        self.assertEqual(["Add", "Add", "Add", "Add", "Sum"], _op_types(optimized))
        self.assertEqual(1, len(optimized.graph.initializer))
        self.assertGreaterEqual(_rewrite_count(rewrites, "SameChildren"), 1)
        self.assert_equivalent(model, optimized, feeds)

    def test_same_children_different_attributes_no_match(self):
        model = _model(
            [
                helper.make_node("Identity", ["X"], ["shared"]),
                helper.make_node("Cast", ["shared"], ["left"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["shared"], ["right"], to=TensorProto.DOUBLE),
            ],
            [_value("X", TensorProto.FLOAT, [2, 3])],
            [
                _value("left", TensorProto.FLOAT16, [2, 3]),
                _value("right", TensorProto.DOUBLE, [2, 3]),
            ],
        )
        feeds = {"X": _range(2, 3)}
        optimized, rewrites = _optimize(model, "SameChildren")

        self.assertEqual(["Cast", "Cast"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "SameChildren"))
        self.assert_equivalent(model, optimized, feeds)

    @staticmethod
    def _shape_children_model():
        return _model(
            [
                helper.make_node("Shape", ["X"], ["sh1"]),
                helper.make_node("Shape", ["X"], ["sh2"]),
                helper.make_node("Expand", ["Y", "sh1"], ["y1"]),
                helper.make_node("Expand", ["Y", "sh2"], ["y2"]),
                helper.make_node("Mul", ["y1", "y2"], ["Z"]),
            ],
            [
                _value("X", TensorProto.FLOAT, ["a", 2, 3, 4]),
                _value("Y", TensorProto.FLOAT, ["a", 1, 3, 4]),
            ],
            [_value("Z", TensorProto.FLOAT, ["a", 2, 3, 4])],
        )

    def test_same_children_duplicate_shape_local_deterministic_cleanup(self):
        model = self._shape_children_model()
        feeds = {"X": _range(2, 2, 3, 4), "Y": _range(2, 1, 3, 4)}
        optimized, rewrites = _optimize(model, "SameChildren")

        self.assertEqual(["Shape", "Expand", "Mul"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "SameChildren"))
        self.assertEqual(1, _rewrite_count(rewrites, "RemoveDuplicateNodes"))
        self.assert_equivalent(model, optimized, feeds)

    def test_shape_based_same_children_and_duplicate_shape_cleanup(self):
        feeds = {"X": _range(2, 2, 3, 4), "Y": _range(2, 1, 3, 4)}
        for pattern_name in ("ShapeBasedSameChildren", "ShapeBasedSameChildren", "SameChildren"):
            with self.subTest(pattern_name=pattern_name):
                model = self._shape_children_model()
                original_inputs = _node_inputs(model)
                optimized, rewrites = _optimize(model, pattern_name)

                self.assertEqual(["Shape", "Expand", "Mul"], _op_types(optimized))
                self.assertEqual(0, len(optimized.graph.initializer))
                self.assertNotEqual(original_inputs, _node_inputs(optimized))
                if pattern_name == "SameChildren":
                    self.assertEqual(0, _rewrite_count(rewrites, pattern_name))
                    self.assertEqual(1, _rewrite_count(rewrites, "RemoveDuplicateNodes"))
                else:
                    self.assertGreaterEqual(_rewrite_count(rewrites, pattern_name), 1)
                self.assert_equivalent(model, optimized, feeds)

    def test_shape_based_identity_full_slice(self):
        model = _model(
            [
                helper.make_node("Shape", ["X"], ["N"], start=0, end=1),
                helper.make_node("Slice", ["X", "zero", "N", "zero"], ["Y"]),
            ],
            [_value("X", TensorProto.FLOAT, ["a"])],
            [_value("Y", TensorProto.FLOAT, ["a"])],
            [_initializer("zero", np.array([0], dtype=np.int64))],
        )
        feeds = {"X": np.arange(5, dtype=np.float32)}
        optimized, rewrites = _optimize(model, "ShapeBasedIdentity")

        self.assertEqual(["Identity"], _op_types(optimized))
        self.assertEqual(1, len(optimized.graph.initializer))
        self.assertEqual(1, _rewrite_count(rewrites, "ShapeBasedIdentity"))
        self.assert_equivalent(model, optimized, feeds)

    def test_shape_based_identity_non_unit_steps_no_match(self):
        model = _model(
            [helper.make_node("Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"])],
            [_value("X", TensorProto.FLOAT, [2, 3])],
            [_value("Y", TensorProto.FLOAT, [2, 3])],
            [
                _initializer("starts", np.array([1], dtype=np.int64)),
                _initializer("ends", np.array([-3], dtype=np.int64)),
                _initializer("axes", np.array([0], dtype=np.int64)),
                _initializer("steps", np.array([-1], dtype=np.int64)),
            ],
        )
        feeds = {"X": _range(2, 3)}
        optimized, rewrites = _optimize(model, "ShapeBasedIdentity")

        self.assertEqual(["Slice"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "ShapeBasedIdentity"))
        self.assert_equivalent(model, optimized, feeds)

    def test_shape_based_matmul_to_mul_variants(self):
        feeds = {
            "X": np.arange(5, dtype=np.float32).reshape(1, -1, 1),
            "Y": np.arange(6, dtype=np.float32).reshape(1, 1, -1),
        }
        for transpose, expected_ops, initializer_count in (
            (False, ["Mul"], 0),
            (True, ["Reshape", "Reshape", "Mul"], 2),
        ):
            with self.subTest(transpose=transpose):
                nodes = [helper.make_node("MatMul", ["X", "Y"], ["Zt" if transpose else "Z"])]
                if transpose:
                    nodes.append(helper.make_node("Transpose", ["Zt"], ["Z"], perm=[0, 2, 1]))
                model = _model(
                    nodes,
                    [
                        _value("X", TensorProto.FLOAT, ["a", "b", 1]),
                        _value("Y", TensorProto.FLOAT, ["a", 1, "c"]),
                    ],
                    [
                        _value(
                            "Z",
                            TensorProto.FLOAT,
                            ["a", "c", "b"] if transpose else ["a", "b", "c"],
                        )
                    ],
                )
                optimized, rewrites = _optimize(model, "ShapeBasedMatMulToMul")

                self.assertEqual(expected_ops, _op_types(optimized))
                self.assertEqual(initializer_count, len(optimized.graph.initializer))
                self.assertEqual(1, _rewrite_count(rewrites, "ShapeBasedMatMulToMul"))
                self.assert_equivalent(model, optimized, feeds)

    def test_shape_based_matmul_transpose_shared_output_no_match(self):
        model = _model(
            [
                helper.make_node("MatMul", ["X", "Y"], ["Zt"]),
                helper.make_node("Transpose", ["Zt"], ["Z"], perm=[0, 2, 1]),
            ],
            [
                _value("X", TensorProto.FLOAT, [1, 5, 1]),
                _value("Y", TensorProto.FLOAT, [1, 1, 6]),
            ],
            [
                _value("Z", TensorProto.FLOAT, [1, 6, 5]),
                _value("Zt", TensorProto.FLOAT, [1, 5, 6]),
            ],
        )
        feeds = {
            "X": np.arange(5, dtype=np.float32).reshape(1, -1, 1),
            "Y": np.arange(6, dtype=np.float32).reshape(1, 1, -1),
        }
        optimized, rewrites = _optimize(model, "ShapeBasedMatMulToMul")

        self.assertEqual(["MatMul", "Transpose"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "ShapeBasedMatMulToMul"))
        self.assert_equivalent(model, optimized, feeds)

    def test_swap_unary_exp_and_scalar_mul(self):
        feeds = {"X": _range(1, 2, 3, 4)}
        for unary, initializers in (
            ("Exp", []),
            ("Mul", [_initializer("cst", np.array([2], dtype=np.float32))]),
        ):
            with self.subTest(unary=unary):
                unary_inputs = ["xt"] if unary == "Exp" else ["xt", "cst"]
                model = _model(
                    [
                        helper.make_node("Transpose", ["X"], ["xt"], perm=[0, 2, 1, 3]),
                        helper.make_node(unary, unary_inputs, ["Y"]),
                    ],
                    [_value("X", TensorProto.FLOAT, ["a", "b", "c", "d"])],
                    [_value("Y", TensorProto.FLOAT, ["a", "c", "b", "d"])],
                    initializers,
                )
                optimized, rewrites = _optimize(model, "SwapUnary")

                self.assertEqual([unary, "Transpose"], _op_types(optimized))
                self.assertEqual(len(initializers), len(optimized.graph.initializer))
                self.assertEqual(1, _rewrite_count(rewrites, "SwapUnary"))
                self.assert_equivalent(model, optimized, feeds)

    def test_swap_unary_scalar_and_shared_layout_no_match(self):
        cases = (
            (
                "scalar",
                [_value("Y", TensorProto.FLOAT, [1, 3, 2])],
                [_initializer("cst", np.array(2, dtype=np.float32))],
            ),
            (
                "shared",
                [
                    _value("Y", TensorProto.FLOAT, [1, 3, 2]),
                    _value("xt", TensorProto.FLOAT, [1, 3, 2]),
                ],
                [_initializer("cst", np.array([2], dtype=np.float32))],
            ),
        )
        for case, outputs, initializers in cases:
            with self.subTest(case=case):
                model = _model(
                    [
                        helper.make_node("Transpose", ["X"], ["xt"], perm=[0, 2, 1]),
                        helper.make_node("Mul", ["xt", "cst"], ["Y"]),
                    ],
                    [_value("X", TensorProto.FLOAT, [1, 2, 3])],
                    outputs,
                    initializers,
                )
                feeds = {"X": _range(1, 2, 3)}
                optimized, rewrites = _optimize(model, "SwapUnary")

                self.assertEqual(["Transpose", "Mul"], _op_types(optimized))
                self.assertEqual(0, _rewrite_count(rewrites, "SwapUnary"))
                self.assert_equivalent(model, optimized, feeds)

    def test_swap_range_add_scalar_zero_and_nonzero_start(self):
        for zero_start, expected_ops in (
            (True, ["Squeeze", "Add", "Range"]),
            (False, ["Squeeze", "Add", "Add", "Range"]),
        ):
            with self.subTest(zero_start=zero_start):
                inputs = [
                    _value("END", TensorProto.INT64, []),
                    _value("PLUS", TensorProto.INT64, [1]),
                ]
                initializers = [_initializer("one", np.array(1, dtype=np.int64))]
                feeds = {
                    "END": np.array(10, dtype=np.int64),
                    "PLUS": np.array([40], dtype=np.int64),
                }
                start = "zero"
                if zero_start:
                    initializers.append(_initializer("zero", np.array(0, dtype=np.int64)))
                else:
                    inputs.insert(0, _value("START", TensorProto.INT64, []))
                    feeds["START"] = np.array(2, dtype=np.int64)
                    start = "START"
                model = _model(
                    [
                        helper.make_node("Range", [start, "END", "one"], ["arange"]),
                        helper.make_node("Add", ["arange", "PLUS"], ["Y"]),
                    ],
                    inputs,
                    [_value("Y", TensorProto.INT64, ["n"])],
                    initializers,
                )
                optimized, rewrites = _optimize(model, "SwapRangeAddScalar")

                self.assertEqual(expected_ops, _op_types(optimized))
                self.assertEqual(2 if zero_start else 1, len(optimized.graph.initializer))
                self.assertEqual(1, _rewrite_count(rewrites, "SwapRangeAddScalar"))
                self.assert_equivalent(model, optimized, feeds)

    def test_swap_range_add_scalar_no_match(self):
        model = _model(
            [
                helper.make_node("Range", ["zero", "END", "one"], ["arange"]),
                helper.make_node("Add", ["arange", "PLUS"], ["Y"]),
            ],
            [_value("END", TensorProto.INT64, []), _value("PLUS", TensorProto.INT64, [])],
            [_value("Y", TensorProto.INT64, ["n"])],
            [
                _initializer("one", np.array(1, dtype=np.int64)),
                _initializer("zero", np.array(0, dtype=np.int64)),
            ],
        )
        feeds = {"END": np.array(10, dtype=np.int64), "PLUS": np.array(40, dtype=np.int64)}
        optimized, rewrites = _optimize(model, "SwapRangeAddScalar")

        self.assertEqual(["Range", "Add"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "SwapRangeAddScalar"))
        self.assert_equivalent(model, optimized, feeds)

    @staticmethod
    def _reduce_sum_normalize_model():
        return _model(
            [
                helper.make_node("Cast", ["X"], ["xc"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", ["xc", "axis"], ["red"], keepdims=1),
                helper.make_node("Mul", ["red", "Y"], ["mul"]),
                helper.make_node("Sub", ["xc", "mul"], ["subc"]),
                helper.make_node("Cast", ["subc"], ["Z"], to=TensorProto.FLOAT16),
            ],
            [
                _value("X", TensorProto.FLOAT16, ["a", "b"]),
                _value("Y", TensorProto.FLOAT, ["a", "b"]),
            ],
            [_value("Z", TensorProto.FLOAT16, ["a", "b"])],
            [_initializer("axis", np.array(-1, dtype=np.int64))],
        )

    def test_reduce_sum_normalize(self):
        model = self._reduce_sum_normalize_model()
        feeds = {"X": _range(2, 3, dtype=np.float16), "Y": _range(2, 3, bias=0.25)}
        original_inputs = _node_inputs(model)
        optimized, rewrites = _optimize(model, "ReduceSumNormalize")

        self.assertEqual(["ReduceSum", "Cast", "Mul", "Sub"], _op_types(optimized))
        self.assertEqual(1, len(optimized.graph.initializer))
        self.assertNotEqual(original_inputs, _node_inputs(optimized))
        self.assertEqual(1, _rewrite_count(rewrites, "ReduceSumNormalize"))
        expected = _run(model, feeds)[0]
        reduced = np.sum(feeds["X"], axis=-1, keepdims=True, dtype=np.float16)
        got = (feeds["X"] - reduced * feeds["Y"].astype(np.float16)).astype(np.float16)
        np.testing.assert_allclose(expected, got, atol=1e-3, rtol=1e-3)

    def test_reduce_arg_topk_min_max_keepdims(self):
        feeds = {"X": np.arange(32 * 8, dtype=np.float32).reshape(32, 8)}
        for extremum, keepdims in (("Min", 0), ("Min", 1), ("Max", 0), ("Max", 1)):
            with self.subTest(extremum=extremum, keepdims=keepdims):
                value_shape = [32, 1] if keepdims else [32]
                model = _model(
                    [
                        helper.make_node(
                            f"Reduce{extremum}", ["X", "axis"], ["Y1"], keepdims=keepdims
                        ),
                        helper.make_node(
                            f"Arg{extremum}", ["X"], ["Y2"], axis=1, keepdims=keepdims
                        ),
                    ],
                    [_value("X", TensorProto.FLOAT, ["a", "b"])],
                    [
                        _value("Y1", TensorProto.FLOAT, value_shape),
                        _value("Y2", TensorProto.INT64, value_shape),
                    ],
                    [_initializer("axis", np.array([1], dtype=np.int64))],
                    opset=22,
                    ir_version=11,
                )
                optimized, rewrites = _optimize(model, "ReduceArgTopK")

                self.assertEqual(
                    ["TopK"] if keepdims else ["TopK", "Squeeze", "Squeeze"], _op_types(optimized)
                )
                self.assertEqual(2, len(optimized.graph.initializer))
                self.assertEqual(1, _rewrite_count(rewrites, "ReduceArgTopK"))
                self.assert_equivalent(model, optimized, feeds)

    def test_reduce_arg_topk_different_axes_no_match(self):
        model = _model(
            [
                helper.make_node("ReduceMin", ["X", "axis"], ["Y1"], keepdims=0),
                helper.make_node("ArgMin", ["X"], ["Y2"], axis=0, keepdims=0),
            ],
            [_value("X", TensorProto.FLOAT, ["a", "b"])],
            [_value("Y1", TensorProto.FLOAT, ["a"]), _value("Y2", TensorProto.INT64, ["b"])],
            [_initializer("axis", np.array([1], dtype=np.int64))],
            opset=22,
            ir_version=11,
        )
        feeds = {"X": np.arange(4 * 3, dtype=np.float32).reshape(4, 3)}
        optimized, rewrites = _optimize(model, "ReduceArgTopK")

        self.assertEqual(["ReduceMin", "ArgMin"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "ReduceArgTopK"))
        self.assert_equivalent(model, optimized, feeds)

    def test_where_add_upstream_topology_both_add_orders(self):
        feeds = {"mask": np.array([[True, False, True], [False, True, False]])}
        for where_first in (True, False):
            with self.subTest(where_first=where_first):
                add_inputs = ["fmask", "X"] if where_first else ["X", "fmask"]
                model = _model(
                    [
                        helper.make_node("Where", ["mask", "zero", "inf"], ["fmask"]),
                        helper.make_node("Add", add_inputs, ["Y"]),
                    ],
                    [_value("mask", TensorProto.BOOL, ["a", "b"])],
                    [_value("Y", TensorProto.FLOAT, ["a", "b"])],
                    [
                        _initializer("X", np.array([0.25], dtype=np.float32)),
                        _initializer("zero", np.array([0], dtype=np.float32)),
                        _initializer("inf", np.array([-np.inf], dtype=np.float32)),
                    ],
                )
                optimized, rewrites = _optimize(model, "WhereAdd")

                self.assertEqual(["Where"], _op_types(optimized))
                self.assertEqual([("mask", "X", "inf")], _node_inputs(optimized))
                self.assertEqual(3, len(optimized.graph.initializer))
                self.assertEqual(1, _rewrite_count(rewrites, "WhereAdd"))
                self.assert_equivalent(model, optimized, feeds)

    def test_where_add_upstream_positive_infinity_addend_no_match(self):
        model = _model(
            [
                helper.make_node("Where", ["mask", "zero", "minus_inf"], ["fmask"]),
                helper.make_node("Add", ["fmask", "X"], ["Y"]),
            ],
            [_value("mask", TensorProto.BOOL, [2, 2])],
            [_value("Y", TensorProto.FLOAT, [2, 2])],
            [
                _initializer("X", np.array([np.inf], dtype=np.float32)),
                _initializer("zero", np.array([0], dtype=np.float32)),
                _initializer("minus_inf", np.array([-np.inf], dtype=np.float32)),
            ],
        )
        feeds = {"mask": np.array([[True, False], [False, True]])}
        optimized, rewrites = _optimize(model, "WhereAdd")

        self.assertEqual(["Where", "Add"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "WhereAdd"))
        with np.errstate(invalid="ignore"):
            self.assert_equivalent(model, optimized, feeds)

    def test_where_add_local_branch_factoring_variants(self):
        feeds = {
            "cond": np.array([[True, False], [False, True]]),
            "X": _range(2, 2),
            "Y": _range(2, 2, bias=0.25),
            "Z": _range(2, 2, bias=0.5),
        }
        for then_common_first, else_common_first in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(
                then_common_first=then_common_first, else_common_first=else_common_first
            ):
                then_inputs = ["Z", "X"] if then_common_first else ["X", "Z"]
                else_inputs = ["Z", "Y"] if else_common_first else ["Y", "Z"]
                model = _model(
                    [
                        helper.make_node("Add", then_inputs, ["then"]),
                        helper.make_node("Add", else_inputs, ["else"]),
                        helper.make_node("Where", ["cond", "then", "else"], ["out"]),
                    ],
                    [
                        _value("cond", TensorProto.BOOL, [2, 2]),
                        _value("X", TensorProto.FLOAT, [2, 2]),
                        _value("Y", TensorProto.FLOAT, [2, 2]),
                        _value("Z", TensorProto.FLOAT, [2, 2]),
                    ],
                    [_value("out", TensorProto.FLOAT, [2, 2])],
                )
                optimized, rewrites = _optimize(model, "WhereAdd")

                self.assertEqual(["Where", "Add"], _op_types(optimized))
                self.assertEqual(0, len(optimized.graph.initializer))
                self.assertEqual(1, _rewrite_count(rewrites, "WhereAdd"))
                self.assert_equivalent(model, optimized, feeds)

    def test_where_add_upstream_and_local_no_match_topologies(self):
        upstream_model = _model(
            [
                helper.make_node("Where", ["mask", "one", "inf"], ["fmask"]),
                helper.make_node("Add", ["fmask", "X"], ["Y"]),
            ],
            [_value("X", TensorProto.FLOAT, [2, 2]), _value("mask", TensorProto.BOOL, [2, 2])],
            [_value("Y", TensorProto.FLOAT, [2, 2])],
            [
                _initializer("one", np.array([1], dtype=np.float32)),
                _initializer("inf", np.array([-np.inf], dtype=np.float32)),
            ],
        )
        local_model = _model(
            [
                helper.make_node("Add", ["X", "Z"], ["then"]),
                helper.make_node("Add", ["Y", "W"], ["else"]),
                helper.make_node("Where", ["mask", "then", "else"], ["out"]),
            ],
            [
                _value("mask", TensorProto.BOOL, [2, 2]),
                _value("X", TensorProto.FLOAT, [2, 2]),
                _value("Y", TensorProto.FLOAT, [2, 2]),
                _value("Z", TensorProto.FLOAT, [2, 2]),
                _value("W", TensorProto.FLOAT, [2, 2]),
            ],
            [_value("out", TensorProto.FLOAT, [2, 2])],
        )
        for name, model, feeds, expected_ops in (
            (
                "upstream",
                upstream_model,
                {"X": _range(2, 2), "mask": np.array([[True, False], [False, True]])},
                ["Where", "Add"],
            ),
            (
                "local",
                local_model,
                {
                    "mask": np.array([[True, False], [False, True]]),
                    "X": _range(2, 2),
                    "Y": _range(2, 2, bias=0.25),
                    "Z": _range(2, 2, bias=0.5),
                    "W": _range(2, 2, bias=0.75),
                },
                ["Add", "Add", "Where"],
            ),
        ):
            with self.subTest(name=name):
                optimized, rewrites = _optimize(model, "WhereAdd")
                self.assertEqual(expected_ops, _op_types(optimized))
                self.assertEqual(0, _rewrite_count(rewrites, "WhereAdd"))
                self.assert_equivalent(model, optimized, feeds)

    def test_where_add_shared_branch_output_no_match(self):
        model = _model(
            [
                helper.make_node("Add", ["X", "Z"], ["then"]),
                helper.make_node("Add", ["Y", "Z"], ["else"]),
                helper.make_node("Where", ["mask", "then", "else"], ["out"]),
            ],
            [
                _value("mask", TensorProto.BOOL, [2, 2]),
                _value("X", TensorProto.FLOAT, [2, 2]),
                _value("Y", TensorProto.FLOAT, [2, 2]),
                _value("Z", TensorProto.FLOAT, [2, 2]),
            ],
            [_value("out", TensorProto.FLOAT, [2, 2]), _value("then", TensorProto.FLOAT, [2, 2])],
        )
        feeds = {
            "mask": np.array([[True, False], [False, True]]),
            "X": _range(2, 2),
            "Y": _range(2, 2, bias=0.25),
            "Z": _range(2, 2, bias=0.5),
        }
        optimized, rewrites = _optimize(model, "WhereAdd")

        self.assertEqual(["Add", "Add", "Where"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "WhereAdd"))
        self.assert_equivalent(model, optimized, feeds)

    def test_where_add_upstream_external_use_no_match(self):
        model = _model(
            [
                helper.make_node("Where", ["mask", "zero", "inf"], ["fmask"]),
                helper.make_node("Add", ["fmask", "X"], ["Y"]),
            ],
            [_value("X", TensorProto.FLOAT, [2, 2]), _value("mask", TensorProto.BOOL, [2, 2])],
            [_value("Y", TensorProto.FLOAT, [2, 2]), _value("fmask", TensorProto.FLOAT, [2, 2])],
            [
                _initializer("zero", np.array([0], dtype=np.float32)),
                _initializer("inf", np.array([-np.inf], dtype=np.float32)),
            ],
        )
        feeds = {"X": _range(2, 2), "mask": np.array([[True, False], [False, True]])}
        optimized, rewrites = _optimize(model, "WhereAdd")

        self.assertEqual(["Where", "Add"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "WhereAdd"))
        self.assert_equivalent(model, optimized, feeds)

    def test_not_where_and_shared_not(self):
        feeds = {
            "X": np.array([[True, False], [True, False]]),
            "A": np.array([[1, 2], [3, 4]], dtype=np.int64),
            "B": np.array([[-1, -2], [-3, -4]], dtype=np.int64),
        }
        for shared, expected_ops in ((False, ["Where"]), (True, ["Not", "Where"])):
            with self.subTest(shared=shared):
                outputs = [_value("Y", TensorProto.INT64, [2, 2])]
                if shared:
                    outputs.append(_value("nx", TensorProto.BOOL, [2, 2]))
                model = _model(
                    [
                        helper.make_node("Not", ["X"], ["nx"]),
                        helper.make_node("Where", ["nx", "A", "B"], ["Y"]),
                    ],
                    [
                        _value("X", TensorProto.BOOL, [2, 2]),
                        _value("A", TensorProto.INT64, [2, 2]),
                        _value("B", TensorProto.INT64, [2, 2]),
                    ],
                    outputs,
                )
                optimized, rewrites = _optimize(model, "NotWhere")

                self.assertEqual(expected_ops, _op_types(optimized))
                self.assertEqual(1, _rewrite_count(rewrites, "NotWhere"))
                self.assert_equivalent(model, optimized, feeds)

    def test_not_where_no_match(self):
        model = _model(
            [helper.make_node("Not", ["X"], ["nx"]), helper.make_node("And", ["nx", "X"], ["Y"])],
            [_value("X", TensorProto.BOOL, [2, 2])],
            [_value("Y", TensorProto.BOOL, [2, 2])],
        )
        feeds = {"X": np.array([[True, False], [False, True]])}
        optimized, rewrites = _optimize(model, "NotWhere")

        self.assertEqual(["Not", "And"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "NotWhere"))
        self.assert_equivalent(model, optimized, feeds)

    def test_not_not_and_shared_first_not(self):
        feeds = {"X": np.array([[False, True], [True, False]])}
        for shared, expected_ops in ((False, ["Identity"]), (True, ["Not", "Identity"])):
            with self.subTest(shared=shared):
                outputs = [_value("Y", TensorProto.BOOL, [2, 2])]
                if shared:
                    outputs.append(_value("middle", TensorProto.BOOL, [2, 2]))
                model = _model(
                    [
                        helper.make_node("Not", ["X"], ["middle"]),
                        helper.make_node("Not", ["middle"], ["Y"]),
                    ],
                    [_value("X", TensorProto.BOOL, [2, 2])],
                    outputs,
                    opset=23,
                )
                optimized, rewrites = _optimize(model, "NotNot")

                self.assertEqual(expected_ops, _op_types(optimized))
                self.assertEqual(1, _rewrite_count(rewrites, "NotNot"))
                self.assert_equivalent(model, optimized, feeds)

    def test_not_not_no_match(self):
        model = _model(
            [helper.make_node("Not", ["X"], ["Y"])],
            [_value("X", TensorProto.BOOL, [2, 2])],
            [_value("Y", TensorProto.BOOL, [2, 2])],
            opset=23,
        )
        feeds = {"X": np.array([[False, True], [True, False]])}
        optimized, rewrites = _optimize(model, "NotNot")

        self.assertEqual(["Not"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "NotNot"))
        self.assert_equivalent(model, optimized, feeds)

    def test_unsqueeze_equal_upstream_topology_preserves_rank(self):
        model = _model(
            [
                helper.make_node("Unsqueeze", ["X", "axis"], ["Y"]),
                helper.make_node("Equal", ["X", "m_one"], ["xe"]),
                helper.make_node("Unsqueeze", ["xe", "axis"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, ["a", "b"])],
            [
                _value("Y", TensorProto.FLOAT, ["a", 1, "b"]),
                _value("Z", TensorProto.BOOL, ["a", 1, "b"]),
            ],
            [
                _initializer("axis", np.array([1], dtype=np.int64)),
                _initializer("m_one", np.array([-1], dtype=np.float32)),
            ],
        )
        feeds = {"X": _range(2, 3)}
        optimized, rewrites = _optimize(model, "UnsqueezeEqual")

        self.assertEqual(["Unsqueeze", "Equal"], _op_types(optimized))
        self.assertEqual(2, len(optimized.graph.initializer))
        self.assertEqual(1, _rewrite_count(rewrites, "UnsqueezeEqual"))
        self.assert_equivalent(model, optimized, feeds)

    def test_unsqueeze_equal_upstream_scalar_rank_zero_constant(self):
        model = _model(
            [
                helper.make_node("Unsqueeze", ["X", "axis"], ["Y"]),
                helper.make_node("Equal", ["X", "m_one"], ["xe"]),
                helper.make_node("Unsqueeze", ["xe", "axis"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, [])],
            [_value("Y", TensorProto.FLOAT, [1]), _value("Z", TensorProto.BOOL, [1])],
            [
                _initializer("axis", np.array([0], dtype=np.int64)),
                _initializer("m_one", np.array(-1, dtype=np.float32)),
            ],
        )
        feeds = {"X": np.array(-1, dtype=np.float32)}
        optimized, rewrites = _optimize(model, "UnsqueezeEqual")

        self.assertEqual(["Unsqueeze", "Equal"], _op_types(optimized))
        self.assertEqual(1, _rewrite_count(rewrites, "UnsqueezeEqual"))
        self.assertEqual([(1,), (1,)], [value.shape for value in _run(model, feeds)])
        self.assertEqual([(1,), (1,)], [value.shape for value in _run(optimized, feeds)])
        self.assert_equivalent(model, optimized, feeds)

    def test_unsqueeze_equal_upstream_scalar_rank_one_constant_no_match(self):
        model = _model(
            [
                helper.make_node("Unsqueeze", ["X", "axis"], ["Y"]),
                helper.make_node("Equal", ["X", "m_one"], ["xe"]),
                helper.make_node("Unsqueeze", ["xe", "axis"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, [])],
            [_value("Y", TensorProto.FLOAT, [1]), _value("Z", TensorProto.BOOL, [1, 1])],
            [
                _initializer("axis", np.array([0], dtype=np.int64)),
                _initializer("m_one", np.array([-1], dtype=np.float32)),
            ],
        )
        feeds = {"X": np.array(-1, dtype=np.float32)}
        optimized, rewrites = _optimize(model, "UnsqueezeEqual")

        self.assertEqual(["Unsqueeze", "Equal", "Unsqueeze"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "UnsqueezeEqual"))
        self.assertEqual([(1,), (1, 1)], [value.shape for value in _run(model, feeds)])
        self.assertEqual([(1,), (1, 1)], [value.shape for value in _run(optimized, feeds)])
        self.assert_equivalent(model, optimized, feeds)

    def test_unsqueeze_equal_local_topology_and_no_match(self):
        feeds = {"X": _range(2, 3), "Y": _range(2, 3, bias=0.25)}
        for same_axes, expected_ops in (
            (True, ["Equal", "Unsqueeze"]),
            (False, ["Unsqueeze", "Unsqueeze", "Equal"]),
        ):
            with self.subTest(same_axes=same_axes):
                initializers = [_initializer("axis1", np.array([1], dtype=np.int64))]
                right_axis = "axis1"
                if not same_axes:
                    initializers.append(_initializer("axis0", np.array([0], dtype=np.int64)))
                    right_axis = "axis0"
                model = _model(
                    [
                        helper.make_node("Unsqueeze", ["X", "axis1"], ["xu"]),
                        helper.make_node("Unsqueeze", ["Y", right_axis], ["yu"]),
                        helper.make_node("Equal", ["xu", "yu"], ["Z"]),
                    ],
                    [
                        _value("X", TensorProto.FLOAT, [2, 3]),
                        _value("Y", TensorProto.FLOAT, [2, 3]),
                    ],
                    [_value("Z", TensorProto.BOOL, [2, 1, 3] if same_axes else [2, 2, 3])],
                    initializers,
                )
                optimized, rewrites = _optimize(model, "UnsqueezeEqual")

                self.assertEqual(expected_ops, _op_types(optimized))
                self.assertEqual(
                    1 if same_axes else 0, _rewrite_count(rewrites, "UnsqueezeEqual")
                )
                self.assert_equivalent(model, optimized, feeds)

    def test_unsqueeze_equal_upstream_different_axes_no_match(self):
        model = _model(
            [
                helper.make_node("Unsqueeze", ["X", "axis1"], ["Y"]),
                helper.make_node("Equal", ["X", "m_one"], ["xe"]),
                helper.make_node("Unsqueeze", ["xe", "axis0"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, [2, 3])],
            [_value("Y", TensorProto.FLOAT, [2, 1, 3]), _value("Z", TensorProto.BOOL, [1, 2, 3])],
            [
                _initializer("axis0", np.array([0], dtype=np.int64)),
                _initializer("axis1", np.array([1], dtype=np.int64)),
                _initializer("m_one", np.array([-1], dtype=np.float32)),
            ],
        )
        feeds = {"X": _range(2, 3)}
        optimized, rewrites = _optimize(model, "UnsqueezeEqual")

        self.assertEqual(["Unsqueeze", "Equal", "Unsqueeze"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "UnsqueezeEqual"))
        self.assert_equivalent(model, optimized, feeds)

    def test_shape_based_shape_shape_add_preserves_upstream_no_match(self):
        model = _model(
            [
                helper.make_node("Shape", ["X"], ["sx"]),
                helper.make_node("Shape", ["Y"], ["sy"]),
                helper.make_node("Add", ["sx", "sy"], ["Z"]),
            ],
            [_value("X", TensorProto.FLOAT, [2, 3]), _value("Y", TensorProto.FLOAT, [2, 3])],
            [_value("Z", TensorProto.INT64, [2])],
        )
        feeds = {"X": _range(2, 3), "Y": _range(2, 3, bias=0.5)}
        optimized, result = _optimize(model, "ShapeBasedShapeShapeAdd", report=True)
        rewrites, report = result

        self.assertEqual(["Shape", "Shape", "Add"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "ShapeBasedShapeShapeAdd"))
        self.assertEqual(0, report.rewrites)
        reasons = {
            item.reason
            for pattern_report in report.patterns
            for item in pattern_report.no_matches
        }
        self.assertIn("the upstream Shape plus Shape rewrite is not implemented", reasons)
        self.assert_equivalent(model, optimized, feeds)

    def test_shape_based_shape_shape_add_non_shape_inputs_no_match(self):
        model = _model(
            [helper.make_node("Add", ["X", "Y"], ["Z"])],
            [_value("X", TensorProto.INT64, [2]), _value("Y", TensorProto.INT64, [2])],
            [_value("Z", TensorProto.INT64, [2])],
        )
        feeds = {"X": np.array([2, 3], dtype=np.int64), "Y": np.array([4, 5], dtype=np.int64)}
        optimized, rewrites = _optimize(model, "ShapeBasedShapeShapeAdd")

        self.assertEqual(["Add"], _op_types(optimized))
        self.assertEqual(0, _rewrite_count(rewrites, "ShapeBasedShapeShapeAdd"))
        self.assert_equivalent(model, optimized, feeds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
