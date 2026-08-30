# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests reshape-related graph patterns with the light ONNX implementation."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import TensorProto, checker
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

from onnx_light.onnx_core import optimization

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


class TestPatternsReshape(ExtTestCase):
    @staticmethod
    def _range(*shape: int, bias: float | None = None) -> np.ndarray:
        """Returns deterministic floating-point data with the requested shape."""
        size = int(np.prod(shape))
        value = np.arange(size, dtype=np.float32) / size
        if bias is not None:
            value += bias
        return value.reshape(shape)

    @staticmethod
    def _value_info(name: str, shape: list[int | str]):
        """Creates a floating-point value-info entry."""
        return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    @staticmethod
    def _initializer(name: str, values, dtype=np.int64):
        """Creates an initializer from values."""
        return onh.from_array(np.asarray(values, dtype=dtype), name=name)

    def _optimize_and_check(
        self,
        model,
        feeds: dict[str, np.ndarray],
        patterns: list[str],
        expected_ops: list[str],
        *,
        required_patterns: set[str] | None = None,
        forbidden_patterns: set[str] | None = None,
        expected_initializer_names: set[str] | None = None,
    ):
        """Optimizes a model and checks its topology and numerical equivalence."""
        checker.check_model(model)
        expected = ReferenceEvaluator(model).run(None, feeds)

        builder = optimization.GraphBuilder(model)
        graph = optimization.GraphGraph(builder, patterns, use_global_patterns=False)
        rewrites = graph.optimize()
        optimized = builder.to_onnx("model")
        checker.check_model(optimized)

        self.assertEqual(expected_ops, [node.op_type for node in optimized.graph.node])
        rewrite_names = {rewrite.pattern_name for rewrite in rewrites}
        if required_patterns:
            self.assertTrue(
                required_patterns <= rewrite_names, (required_patterns, rewrite_names)
            )
        if forbidden_patterns:
            self.assertTrue(forbidden_patterns.isdisjoint(rewrite_names), rewrite_names)
        if expected_initializer_names is not None:
            self.assertEqual(
                expected_initializer_names,
                {initializer.name for initializer in optimized.graph.initializer},
            )

        got = ReferenceEvaluator(optimized).run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for expected_value, got_value in zip(expected, got):
            self.assertEqual(expected_value.dtype, got_value.dtype)
            np.testing.assert_allclose(expected_value, got_value, atol=1e-6, rtol=1e-6)
        return optimized, rewrites

    def _make_reshape_matmul_reshape(
        self, *, dynamic: bool = False, keep_intermediate: bool = False
    ):
        input_x_shape = ["D32", "D128"] if dynamic else [32, 128]
        input_y_shape = ["batch", "channel", "D128", "D64"] if dynamic else [3, 5, 128, 64]
        output_shape = ["batch", "channel", "D32", "D64"] if dynamic else [3, 5, 32, 64]
        outputs = [self._value_info("Z", output_shape)]
        if keep_intermediate:
            outputs.append(self._value_info("xm1", [1, 32, 128]))
        graph = oh.make_graph(
            [
                oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                oh.make_node("Unsqueeze", ["xu1", "one"], ["xu2"]),
                oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                oh.make_node("Cast", ["xm2c"], ["xm2"], to=TensorProto.FLOAT),
                oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
            ],
            "reshape_matmul_reshape",
            [self._value_info("X", input_x_shape), self._value_info("Y", input_y_shape)],
            outputs,
            [
                self._initializer("zero", [0]),
                self._initializer("one", [1]),
                self._initializer("shape1", [1, 32, 128]),
                self._initializer("shape2", [15, 128, 64]),
                self._initializer("shape3", [3, 5, 32, 64]),
            ],
        )
        return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10)

    def _check_reshape_matmul_reshape(
        self, *, dynamic: bool = False, keep_intermediate: bool = False, combined: bool = False
    ):
        patterns = ["Cast", "ReshapeMatMulReshape", "UnsqueezeUnsqueeze"]
        if combined:
            patterns.extend(["MatMulReshape2Of3", "ReshapeReshape"])
        expected_ops = (
            ["Unsqueeze", "Reshape", "Reshape", "MatMul", "Reshape"]
            if dynamic
            else (
                ["Unsqueeze", "Reshape", "MatMul"]
                if keep_intermediate
                else ["Unsqueeze", "MatMul"]
            )
        )
        required = {"Cast", "UnsqueezeUnsqueeze"}
        forbidden = None
        if dynamic:
            forbidden = {"ReshapeMatMulReshape"}
        else:
            required.add("ReshapeMatMulReshape")
        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        return self._optimize_and_check(
            self._make_reshape_matmul_reshape(
                dynamic=dynamic, keep_intermediate=keep_intermediate
            ),
            feeds,
            patterns,
            expected_ops,
            required_patterns=required,
            forbidden_patterns=forbidden,
        )

    def test_reshape_matmul_reshape_static(self):
        self._check_reshape_matmul_reshape()

    def test_reshape_matmul_reshape_dynamic_1(self):
        self._check_reshape_matmul_reshape(dynamic=True)

    def test_reshape_matmul_reshape_dynamic_2(self):
        self._check_reshape_matmul_reshape(dynamic=True)

    def test_reshape_matmul_reshape_keep_intermediate(self):
        optimized, _ = self._check_reshape_matmul_reshape(keep_intermediate=True)
        self.assertEqual(["Z", "xm1"], [output.name for output in optimized.graph.output])

    def test_combination_reshape_matmul_reshape_static(self):
        self._check_reshape_matmul_reshape(combined=True)

    def test_combination_reshape_matmul_reshape_dynamic_1(self):
        self._check_reshape_matmul_reshape(dynamic=True, combined=True)

    def test_combination_reshape_matmul_reshape_dynamic_2(self):
        self._check_reshape_matmul_reshape(dynamic=True, combined=True)

    def test_combination_reshape_matmul_reshape_keep_intermediate(self):
        optimized, _ = self._check_reshape_matmul_reshape(keep_intermediate=True, combined=True)
        self.assertEqual(["Z", "xm1"], [output.name for output in optimized.graph.output])

    def test_reshape_same_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Reshape", ["X", "shape"], ["Y"])],
                "reshape_same_shape",
                [self._value_info("X", [2, 3])],
                [self._value_info("Y", [2, 3])],
                [self._initializer("shape", [2, 3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(2, 3)},
            ["Reshape"],
            ["Identity"],
            required_patterns={"Reshape"},
            expected_initializer_names={"shape"},
        )

    def test_reshape_reshape_execution(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "r1"], ["xu1"]),
                    oh.make_node("Reshape", ["xu1", "r2"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2"]),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "reshape_reshape_execution",
                [self._value_info("X", [32, 128]), self._value_info("Y", [3, 5, 128, 64])],
                [self._value_info("Z", [3, 5, 32, 64])],
                [
                    self._initializer("r1", [-1]),
                    self._initializer("r2", [-1, 128]),
                    self._initializer("shape1", [1, 32, 128]),
                    self._initializer("shape2", [15, 128, 64]),
                    self._initializer("shape3", [3, 5, 32, 64]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)},
            ["ReshapeReshape"],
            ["Reshape", "Reshape", "MatMul", "Reshape"],
            required_patterns={"ReshapeReshape"},
        )

    def test_reshape_reshape_3d_shape_based(self):
        feeds = {"X": self._range(2, 3, 8)}
        with self.subTest(pattern="ReshapeReshape"):
            model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                        oh.make_node("Reshape", ["xr", "shape2"], ["xrr"]),
                        oh.make_node("Add", ["xrr", "one"], ["Y"]),
                    ],
                    "reshape_reshape_3d",
                    [self._value_info("X", ["a", "b", "c"])],
                    [self._value_info("Y", ["a", "b", "c"])],
                    [
                        self._initializer("shape1", [0, 0, 2, -1]),
                        self._initializer("shape2", [2, 3, 8]),
                        self._initializer("one", [1], np.float32),
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=10,
            )
            self._optimize_and_check(
                model,
                feeds,
                ["ReshapeReshape"],
                ["Reshape", "Add"],
                required_patterns={"ReshapeReshape"},
            )
        with self.subTest(pattern="ShapedBasedReshape"):
            model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Reshape", ["X", "shape"], ["xr"]),
                        oh.make_node("Add", ["xr", "one"], ["Y"]),
                    ],
                    "shape_based_reshape",
                    [self._value_info("X", ["a", "b", "c"])],
                    [self._value_info("Y", ["a", "b", "c"])],
                    [
                        self._initializer("shape", [0, 0, -1]),
                        self._initializer("one", [1], np.float32),
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=10,
            )
            self._optimize_and_check(
                model,
                feeds,
                ["ShapedBasedReshape"],
                ["Add"],
                required_patterns={"ShapedBasedReshape"},
            )

    def test_reshape_reshape_zero(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape"], ["Xs"], name="R1"),
                    oh.make_node("Reshape", ["Xs", "shape0"], ["Y"], name="R2"),
                ],
                "reshape_reshape_zero",
                [
                    self._value_info("X", ["a", "b", "c"]),
                    oh.make_tensor_value_info("shape", TensorProto.INT64, [4]),
                ],
                [self._value_info("Y", ["d", "e", "f", "g"])],
                [self._initializer("shape0", [0, 1, -1, 0])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(2, 4, 128), "shape": np.array([2, 8, 4, 16], dtype=np.int64)},
            ["ReshapeReshape"],
            ["Gather", "Gather", "Concat", "Reshape"],
            required_patterns={"ReshapeReshape"},
        )

    def test_reshape_reshape_zero_from_positive(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xr"]),
                    oh.make_node("Reshape", ["xr", "shape2"], ["Y"]),
                ],
                "reshape_reshape_zero_positive",
                [self._value_info("X", [2, 3, 16, 8])],
                [self._value_info("Y", [2, 3, 16, 1, 8])],
                [
                    self._initializer("shape1", [2, 3, 16, 8, 1]),
                    self._initializer("shape2", [0, 0, 0, 1, 8]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(2, 3, 16, 8)},
            ["ReshapeReshape"],
            ["Reshape"],
            required_patterns={"ReshapeReshape"},
        )

    def test_reshape_reshape_three(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["s1"]),
                    oh.make_node("Reshape", ["s1", "shape2"], ["s2"]),
                    oh.make_node("Reshape", ["s2", "shape3"], ["Y"]),
                ],
                "reshape_reshape_three",
                [self._value_info("X", ["a", "b", 8])],
                [self._value_info("Y", ["d", 8])],
                [
                    self._initializer("shape1", [4, 7, 7, 8]),
                    self._initializer("shape2", [4, 49, 8]),
                    self._initializer("shape3", [196, 8]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(4, 49, 8)},
            ["ReshapeReshape"],
            ["Reshape"],
            required_patterns={"ReshapeReshape"},
        )

    def test_reshape_reshape_single(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xs"]),
                    oh.make_node("Reshape", ["xs", "shape2"], ["Y"]),
                ],
                "reshape_reshape_single",
                [self._value_info("X", ["a", "b", 1, 16, 80])],
                [self._value_info("Y", ["ab", 1280])],
                [
                    self._initializer("shape1", [0, 0, 1, 16, 80]),
                    self._initializer("shape2", [-1, 1280]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(1, 3, 1, 16, 80)},
            ["ReshapeReshape"],
            ["Reshape"],
            required_patterns={"ReshapeReshape"},
        )

    def _make_2of3_model(self, op_type: str, topology: str, *, keep_intermediate: bool = False):
        if op_type == "Mul":
            input_x_shape = [2, 3, 4]
            input_y_shape = [2, 3, 4] if topology in {"three", "both"} else [3, 8]
            reshaped_x_shape = [-1, 8]
            reshaped_y_shape = [3, -1]
            output_shape = [2, 3, 4] if topology != "both" else [3, 8]
            inner_shape = [3, 8]
        else:
            input_x_shape = [2, 2, 3, 4] if topology != "right" else [2, 2, 4, 3]
            input_y_shape = (
                [2, 2, 4, 3]
                if topology in {"three", "both"}
                else [4, 4, 3] if topology == "left" else [4, 3, 4]
            )
            reshaped_x_shape = [-1, 3, 4] if topology != "right" else [-1, 4, 3]
            reshaped_y_shape = [-1, 4, 3]
            output_shape = [2, 2, 3, 3] if topology != "both" else [4, 3, 3]
            inner_shape = [4, 3, 3]

        nodes = []
        initializers = []
        left_name = "X"
        right_name = "Y"
        if topology in {"three", "left", "right", "both"}:
            if topology != "right":
                nodes.append(oh.make_node("Reshape", ["X", "shape1"], ["xr"]))
                initializers.append(self._initializer("shape1", reshaped_x_shape))
                left_name = "xr"
            else:
                nodes.append(oh.make_node("Reshape", ["X", "shape1"], ["xr"]))
                initializers.append(self._initializer("shape1", reshaped_x_shape))
                right_name = "xr"
        if topology in {"three", "both"}:
            nodes.append(oh.make_node("Reshape", ["Y", "shape2"], ["yr"]))
            initializers.append(self._initializer("shape2", reshaped_y_shape))
            right_name = "yr"
        if topology == "right":
            left_name = "Y"

        inner_output = "inner" if topology != "both" else "Z"
        nodes.append(oh.make_node(op_type, [left_name, right_name], [inner_output]))
        if topology != "both":
            nodes.append(oh.make_node("Reshape", [inner_output, "shape3"], ["Z"]))
            initializers.append(self._initializer("shape3", output_shape))

        outputs = [self._value_info("Z", output_shape)]
        if keep_intermediate:
            outputs.append(self._value_info("inner", inner_shape))
        graph = oh.make_graph(
            nodes,
            f"{op_type.lower()}_reshape_2of3_{topology}",
            [self._value_info("X", input_x_shape), self._value_info("Y", input_y_shape)],
            outputs,
            initializers,
        )
        return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10)

    def _check_2of3(self, op_type: str, topology: str, *, keep_intermediate: bool = False):
        pattern = "Reshape2Of3" if op_type == "Mul" else "MatMulReshape2Of3"
        model = self._make_2of3_model(op_type, topology, keep_intermediate=keep_intermediate)
        if op_type == "Mul":
            feeds = {
                "X": self._range(2, 3, 4),
                "Y": self._range(2, 3, 4) if topology in {"three", "both"} else self._range(3, 8),
            }
        else:
            feeds = {
                "X": self._range(*([2, 2, 4, 3] if topology == "right" else [2, 2, 3, 4])),
                "Y": self._range(
                    *(
                        [2, 2, 4, 3]
                        if topology in {"three", "both"}
                        else [4, 4, 3] if topology == "left" else [4, 3, 4]
                    )
                ),
            }
        if topology == "three" and not keep_intermediate:
            expected_ops = [op_type]
        elif topology == "three":
            expected_ops = [op_type, "Reshape"]
        elif topology == "both" and op_type == "MatMul":
            expected_ops = ["Reshape", "Reshape", op_type]
        elif topology == "both":
            expected_ops = [op_type, "Reshape"]
        else:
            expected_ops = ["Reshape", op_type]
        return self._optimize_and_check(
            model, feeds, [pattern], expected_ops, required_patterns={pattern}
        )

    def test_mul_reshape_2of3_static_3(self):
        self._check_2of3("Mul", "three")

    def test_mul_reshape_2of3_static_3_keep(self):
        optimized, _ = self._check_2of3("Mul", "three", keep_intermediate=True)
        self.assertEqual(["Z", "inner"], [output.name for output in optimized.graph.output])

    def test_mul_reshape_2of3_static_2_left(self):
        self._check_2of3("Mul", "left")

    def test_mul_reshape_2of3_static_2_right(self):
        self._check_2of3("Mul", "right")

    def test_mul_reshape_2of3_static_2_left_right(self):
        self._check_2of3("Mul", "both")

    def test_matmul_reshape_2of3_static_3(self):
        self._check_2of3("MatMul", "three")

    def test_matmul_reshape_2of3_static_3_keep(self):
        optimized, _ = self._check_2of3("MatMul", "three", keep_intermediate=True)
        self.assertEqual(["Z", "inner"], [output.name for output in optimized.graph.output])

    def test_matmul_reshape_2of3_static_2_left(self):
        self._check_2of3("MatMul", "left")

    def test_matmul_reshape_2of3_static_2_right(self):
        self._check_2of3("MatMul", "right")

    def test_matmul_reshape_2of3_static_2_left_right(self):
        self._check_2of3("MatMul", "both")

    def test_reshape_reshape_binary(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Reshape", ["X", "shape1"], ["xc"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["yc"]),
                    oh.make_node("Add", ["xc", "yc"], ["Z"]),
                ],
                "reshape_reshape_binary",
                [self._value_info("X", ["a", 4]), self._value_info("Y", ["a", 4])],
                [self._value_info("Z", ["b", 8])],
                [self._initializer("shape1", [-1, 8]), self._initializer("shape2", [-1, 8])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(6, 4), "Y": self._range(6, 4)},
            ["ReshapeReshapeBinary"],
            ["Add", "Reshape"],
            required_patterns={"ReshapeReshapeBinary"},
        )

    def _make_reduce_reshape(
        self,
        input_shape: list[int],
        axes: list[int] | None,
        output_shape: list[int],
        *,
        opset: int = 18,
        add_cos: bool = False,
    ):
        reduce_inputs = ["X"]
        reduce_attributes = {"keepdims": 1}
        initializers = []
        if opset >= 13 and axes is not None:
            reduce_inputs.append("axes")
            initializers.append(self._initializer("axes", axes))
        elif axes is not None:
            reduce_attributes["axes"] = axes
        nodes = [
            oh.make_node("ReduceSum", reduce_inputs, ["reduced"], **reduce_attributes),
            oh.make_node("Reshape", ["reduced", "shape"], ["reshaped" if add_cos else "Y"]),
        ]
        if add_cos:
            nodes.append(oh.make_node("Cos", ["reshaped"], ["Y"]))
        initializers.append(self._initializer("shape", output_shape))
        graph = oh.make_graph(
            nodes,
            "reduce_reshape",
            [self._value_info("X", input_shape)],
            [self._value_info("Y", output_shape)],
            initializers,
        )
        return oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)], ir_version=10)

    def test_reduce_reshape(self):
        self._optimize_and_check(
            self._make_reduce_reshape([3, 2], [1], [3]),
            {"X": self._range(3, 2)},
            ["ReduceReshape"],
            ["ReduceSum"],
            required_patterns={"ReduceReshape"},
        )

    def test_reduce_reshape_2d(self):
        self._optimize_and_check(
            self._make_reduce_reshape([4, 3, 2], [0, 2], [3]),
            {"X": self._range(4, 3, 2)},
            ["ReduceReshape"],
            ["ReduceSum"],
            required_patterns={"ReduceReshape"},
        )

    def test_reduce_reshape_all(self):
        self._optimize_and_check(
            self._make_reduce_reshape([3, 2], None, [], add_cos=True),
            {"X": self._range(3, 2)},
            ["ReduceReshape"],
            ["ReduceSum", "Cos"],
            required_patterns={"ReduceReshape"},
        )

    def test_reduce_reshape_opset10(self):
        self._optimize_and_check(
            self._make_reduce_reshape([3, 2], [1], [3], opset=10),
            {"X": self._range(3, 2)},
            ["ReduceReshape"],
            ["ReduceSum"],
            required_patterns={"ReduceReshape"},
        )

    def _make_concat_reshape(self, *, with_abs: bool):
        nodes = [
            oh.make_node("Shape", ["X"], ["D2"], start=2, end=3),
            oh.make_node("Shape", ["X"], ["D1"], start=3, end=4),
        ]
        dimension = "D1"
        if with_abs:
            nodes.append(oh.make_node("Abs", ["D1"], ["abs_D1"]))
            dimension = "abs_D1"
        nodes.extend(
            [
                oh.make_node("Concat", ["I1", "I2", dimension, "D2"], ["shape"], axis=0),
                oh.make_node("Reshape", ["X", "shape"], ["Y"]),
            ]
        )
        graph = oh.make_graph(
            nodes,
            "concat_reshape",
            [self._value_info("X", ["a", "b", "c", "d"])],
            [self._value_info("Y", ["a", "b", "d", "c"])],
            [self._initializer("I1", [2]), self._initializer("I2", [1])],
        )
        return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10)

    def test_concat_reshape_shape_inputs(self):
        self._optimize_and_check(
            self._make_concat_reshape(with_abs=False),
            {"X": self._range(2, 1, 3, 5)},
            ["ConcatReshape"],
            ["Shape", "Concat", "Reshape"],
            required_patterns={"ConcatReshape"},
        )

    def test_concat_reshape_any(self):
        self._optimize_and_check(
            self._make_concat_reshape(with_abs=True),
            {"X": self._range(2, 1, 3, 5)},
            ["ConcatReshape"],
            ["Shape", "Concat", "Reshape"],
            required_patterns={"ConcatReshape"},
        )

    def test_static_concat_reshape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["X"], ["D2"], start=2, end=3),
                    oh.make_node("Concat", ["I1", "D2"], ["shape"], axis=0),
                    oh.make_node("Reshape", ["X", "shape"], ["Y"]),
                ],
                "static_concat_reshape",
                [self._value_info("X", [2, 3, "d"])],
                [self._value_info("Y", [6, "d"])],
                [self._initializer("I1", [6])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(2, 3, 12)},
            ["StaticConcatReshape"],
            ["Reshape"],
            required_patterns={"StaticConcatReshape"},
        )

    def test_shape_based_edit_distance_reshape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["X"], ["D2"], start=2, end=3),
                    oh.make_node("Concat", ["minus_one", "D2"], ["shape"], axis=0),
                    oh.make_node("Reshape", ["X", "shape"], ["Y"]),
                ],
                "shape_based_edit_distance_reshape",
                [self._value_info("X", [2, 3, "d"])],
                [self._value_info("Y", [6, "d"])],
                [self._initializer("minus_one", [-1])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(2, 3, 12)},
            ["ShapeBasedEditDistanceReshape"],
            ["Reshape"],
            required_patterns={"ShapeBasedEditDistanceReshape"},
        )

    def test_shape_based_reshape_is_squeeze_reshape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["X"], ["shape_x"]),
                    oh.make_node("Concat", ["one", "shape_x", "one"], ["shape"], axis=0),
                    oh.make_node("Reshape", ["X", "shape"], ["Y"]),
                ],
                "shape_based_reshape_is_squeeze",
                [self._value_info("X", [2, 3, "d"])],
                [self._value_info("Y", [1, 2, 3, "d", 1])],
                [self._initializer("one", [1])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._optimize_and_check(
            model,
            {"X": self._range(2, 3, 12)},
            ["ShapeBasedReshapeIsSqueeze"],
            ["Unsqueeze"],
            required_patterns={"ShapeBasedReshapeIsSqueeze"},
        )

    def _make_unsqueeze_reshape(
        self,
        input_shape: list[int | str],
        axis: int,
        reshape_shape: list[int],
        output_shape: list[int | str],
    ):
        graph = oh.make_graph(
            [
                oh.make_node("Unsqueeze", ["X", "axis"], ["xu"]),
                oh.make_node("Reshape", ["xu", "shape"], ["Z"]),
            ],
            "unsqueeze_reshape",
            [self._value_info("X", input_shape)],
            [self._value_info("Z", output_shape)],
            [self._initializer("axis", [axis]), self._initializer("shape", reshape_shape)],
        )
        return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10)

    def test_unsqueeze_or_squeeze_reshape(self):
        self._optimize_and_check(
            self._make_unsqueeze_reshape(["a", 8, 16], 0, [-1, 128], ["a*2", 64]),
            {"X": self._range(32, 8, 16)},
            ["UnsqueezeOrSqueezeReshape"],
            ["Reshape"],
            required_patterns={"UnsqueezeOrSqueezeReshape"},
        )

    def test_unsqueeze_reshape_not(self):
        self._optimize_and_check(
            self._make_unsqueeze_reshape(
                ["a", "b", "c", "d"], 2, [0, 1, -1, 0], ["e", "f", "g", "h"]
            ),
            {"X": self._range(2, 8, 16, 4)},
            ["UnsqueezeReshape"],
            ["Unsqueeze", "Reshape"],
            forbidden_patterns={"UnsqueezeReshape"},
        )

    def test_unsqueeze_reshape(self):
        self._optimize_and_check(
            self._make_unsqueeze_reshape(["a", "b", "c"], 2, [0, 1, -1, 0], ["e", "f", "g", "h"]),
            {"X": self._range(32, 8, 16)},
            ["UnsqueezeReshape"],
            ["Unsqueeze"],
            required_patterns={"UnsqueezeReshape"},
        )

    def test_unsqueeze_or_squeeze_reshape_zeros_axis_beyond(self):
        self._optimize_and_check(
            self._make_unsqueeze_reshape(["a", 8, 16], 3, [0, 0, 0, -1], ["a", 8, 16, 1]),
            {"X": self._range(2, 8, 16)},
            ["UnsqueezeOrSqueezeReshape"],
            ["Reshape"],
            required_patterns={"UnsqueezeOrSqueezeReshape"},
        )

    def test_unsqueeze_or_squeeze_reshape_zeros_axis_overlaps(self):
        self._optimize_and_check(
            self._make_unsqueeze_reshape(["a", 8, 16, 4], 2, [0, 0, 0, -1], ["a", 8, 1, 64]),
            {"X": self._range(2, 8, 16, 4)},
            ["UnsqueezeOrSqueezeReshape"],
            ["Unsqueeze", "Reshape"],
            forbidden_patterns={"UnsqueezeOrSqueezeReshape"},
        )

    def _make_reshape_squeeze(
        self,
        input_shape: list[int | str],
        reshape_shape: list[int],
        output_shape: list[int | str],
    ):
        graph = oh.make_graph(
            [
                oh.make_node("Reshape", ["X", "shape"], ["xr"]),
                oh.make_node("Squeeze", ["xr", "axis"], ["Z"]),
            ],
            "reshape_squeeze",
            [self._value_info("X", input_shape)],
            [self._value_info("Z", output_shape)],
            [self._initializer("shape", reshape_shape), self._initializer("axis", [3])],
        )
        return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=10)

    def test_reshape_squeeze_basic(self):
        self._optimize_and_check(
            self._make_reshape_squeeze(["a", "b", "c", "d"], [0, 0, 0, 1, 8], ["a", "b", "c", 8]),
            {"X": self._range(2, 3, 4, 8)},
            ["ReshapeSqueeze"],
            ["Reshape"],
            required_patterns={"ReshapeSqueeze"},
        )

    def test_reshape_squeeze_static_shape(self):
        self._optimize_and_check(
            self._make_reshape_squeeze([2, 3, 4, 8], [2, 3, 4, 1, 8], [2, 3, 4, 8]),
            {"X": self._range(2, 3, 4, 8)},
            ["ReshapeSqueeze"],
            ["Reshape"],
            required_patterns={"ReshapeSqueeze"},
        )

    def test_reshape_squeeze_zero_after_axis_not_applied(self):
        self._optimize_and_check(
            self._make_reshape_squeeze(
                ["a", "b", "c", "d", "e"], [0, 0, 0, 1, 0], ["a", "b", "c", "e"]
            ),
            {"X": self._range(2, 3, 4, 1, 8)},
            ["ReshapeSqueeze"],
            ["Reshape", "Squeeze"],
            forbidden_patterns={"ReshapeSqueeze"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
