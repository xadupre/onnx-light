# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests migrated matmul, normalization, and activation graph patterns."""

from __future__ import annotations

import unittest

import numpy as np

from onnx_light.onnx import TensorProto, checker, helper, numpy_helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core import optimization


class TestPatternsMatmulNormalization(unittest.TestCase):
    @staticmethod
    def _range(*shape: int, bias: float | None = None, dtype=np.float32) -> np.ndarray:
        """Returns deterministic data with the requested shape and type."""
        size = int(np.prod(shape))
        value = np.arange(size, dtype=np.float32) / max(size, 1)
        if bias is not None:
            value += bias
        return value.reshape(shape).astype(dtype)

    @staticmethod
    def _value_info(name: str, data_type: int, shape):
        """Creates a tensor value-info entry."""
        return helper.make_tensor_value_info(name, data_type, shape)

    @staticmethod
    def _initializer(name: str, values, dtype) -> TensorProto:
        """Creates an initializer from NumPy-compatible values."""
        return numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)

    def _make_model(
        self, nodes, inputs, outputs, initializers=(), *, opset: int = 18, name: str = "pattern"
    ):
        """Creates a small model for one pattern."""
        graph = helper.make_graph(
            list(nodes),
            name,
            [self._value_info(*value) for value in inputs],
            [self._value_info(*value) for value in outputs],
            list(initializers),
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
        )

    @staticmethod
    def _attribute(node, name: str, default):
        """Returns an integer or floating-point node attribute."""
        for attribute in node.attribute:
            if attribute.name == name:
                if attribute.type == attribute.INT:
                    return attribute.i
                if attribute.type == attribute.FLOAT:
                    return attribute.f
                if attribute.type == attribute.STRING:
                    return attribute.s.decode()
        return default

    @classmethod
    def _layer_normalization_kernel(cls, node, x, scale, bias=None):
        """Computes LayerNormalization for the test reference evaluator."""
        axis = int(cls._attribute(node, "axis", -1))
        if axis < 0:
            axis += x.ndim
        epsilon = float(cls._attribute(node, "epsilon", 1e-5))
        compute = x.astype(np.float32)
        axes = tuple(range(axis, x.ndim))
        mean = np.mean(compute, axis=axes, keepdims=True)
        variance = np.mean((compute - mean) ** 2, axis=axes, keepdims=True)
        result = (compute - mean) / np.sqrt(variance + epsilon)
        result *= scale.astype(np.float32)
        if bias is not None:
            result += bias.astype(np.float32)
        result = result.astype(x.dtype)
        if len(node.output) == 1:
            return result
        inverse_standard_deviation = 1.0 / np.sqrt(variance + epsilon)
        return result, mean.astype(np.float32), inverse_standard_deviation.astype(np.float32)

    @classmethod
    def _batch_normalization_kernel(cls, node, x, scale, bias, input_mean, input_variance):
        """Computes inference or training BatchNormalization."""
        epsilon = float(cls._attribute(node, "epsilon", 1e-5))
        training_mode = int(cls._attribute(node, "training_mode", 0))
        momentum = float(cls._attribute(node, "momentum", 0.9))
        parameter_shape = [1] * x.ndim
        parameter_shape[1] = x.shape[1]
        reshaped_scale = scale.reshape(parameter_shape)
        reshaped_bias = bias.reshape(parameter_shape)
        if training_mode:
            axes = (0, *range(2, x.ndim))
            mean = np.mean(x, axis=axes, keepdims=True)
            variance = np.mean((x - mean) ** 2, axis=axes, keepdims=True)
            running_mean = input_mean * momentum + mean.reshape(-1) * (1.0 - momentum)
            running_variance = input_variance * momentum + variance.reshape(-1) * (1.0 - momentum)
        else:
            mean = input_mean.reshape(parameter_shape)
            variance = input_variance.reshape(parameter_shape)
            running_mean = input_mean
            running_variance = input_variance
        result = (
            (x - mean) / np.sqrt(variance + epsilon) * reshaped_scale + reshaped_bias
        ).astype(x.dtype)
        if len(node.output) == 1:
            return result
        return result, running_mean.astype(x.dtype), running_variance.astype(x.dtype)

    @classmethod
    def _softmax_cross_entropy_kernel(cls, node, scores, labels):
        """Computes the migrated masked mean cross-entropy loss."""
        axis = int(cls._attribute(node, "axis", 1))
        ignore_index = int(cls._attribute(node, "ignore_index", -100))
        reduction = cls._attribute(node, "reduction", "mean")
        compute = scores.astype(np.float32)
        maximum = np.max(compute, axis=axis, keepdims=True)
        log_probabilities = compute - maximum
        log_probabilities -= np.log(np.sum(np.exp(log_probabilities), axis=axis, keepdims=True))
        valid = labels != ignore_index
        safe_labels = np.where(valid, labels, 0)
        gathered = np.take_along_axis(
            log_probabilities, np.expand_dims(safe_labels, axis=axis), axis=axis
        ).squeeze(axis=axis)
        losses = np.where(valid, -gathered, 0.0)
        if reduction == "none":
            result = losses
        elif reduction == "sum":
            result = np.sum(losses)
        else:
            result = np.sum(losses) / np.sum(valid)
        return np.asarray(result, dtype=scores.dtype)

    @classmethod
    def _log_softmax_kernel(cls, node, x):
        """Computes LogSoftmax for half-precision source graphs."""
        axis = int(cls._attribute(node, "axis", -1))
        compute = x.astype(np.float32)
        maximum = np.max(compute, axis=axis, keepdims=True)
        shifted = compute - maximum
        result = shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        return result.astype(x.dtype)

    @staticmethod
    def _where_kernel(node, condition, x, y):
        """Computes Where for half-precision source graphs."""
        del node
        return np.where(condition, x, y)

    @classmethod
    def _reduce_mean_kernel(cls, node, data, axes=None):
        """Computes ReduceMean for half-precision source graphs."""
        keepdims = bool(cls._attribute(node, "keepdims", 1))
        if axes is None:
            reduced_axes = None
        else:
            reduced_axes = tuple(int(axis) for axis in np.asarray(axes).reshape(-1))
        result = np.mean(data.astype(np.float32), axis=reduced_axes, keepdims=keepdims)
        return result.astype(data.dtype)

    def _evaluate(self, model, feeds):
        """Evaluates a model using only onnx-light kernels and NumPy callbacks."""
        evaluator = ReferenceEvaluator(model)
        evaluator.register_custom_kernel(
            "", "LayerNormalization", self._layer_normalization_kernel
        )
        evaluator.register_custom_kernel(
            "", "BatchNormalization", self._batch_normalization_kernel
        )
        evaluator.register_custom_kernel(
            "", "SoftmaxCrossEntropyLoss", self._softmax_cross_entropy_kernel
        )
        evaluator.register_custom_kernel("", "LogSoftmax", self._log_softmax_kernel)
        evaluator.register_custom_kernel("", "Where", self._where_kernel)
        evaluator.register_custom_kernel("", "ReduceMean", self._reduce_mean_kernel)
        return evaluator.run(None, feeds)

    def _optimize(self, model, patterns, *, check_model: bool = True, schema_lookup: bool = True):
        """Optimizes a model with an isolated GraphGraph."""
        if check_model:
            checker.check_model(model)
        builder = (
            optimization.GraphBuilder(model)
            if schema_lookup
            else optimization.GraphBuilder(model, schema_lookup=None)
        )
        graph = optimization.GraphGraph(builder, patterns, use_global_patterns=False)
        rewrites, report = graph.optimize(report=True)
        optimized = builder.to_onnx("model")
        if check_model:
            checker.check_model(optimized)
        return optimized, rewrites, report

    def _check_equivalent(
        self, model, optimized, feeds, *, expected=None, atol: float = 1e-6, rtol: float = 1e-6
    ):
        """Checks numerical equivalence between source and optimized models."""
        expected_values = self._evaluate(model, feeds) if expected is None else expected
        got_values = self._evaluate(optimized, feeds)
        self.assertEqual(len(expected_values), len(got_values))
        for expected_value, got_value in zip(expected_values, got_values):
            self.assertEqual(expected_value.dtype, got_value.dtype)
            np.testing.assert_allclose(expected_value, got_value, atol=atol, rtol=rtol)

    def _optimize_and_check(
        self,
        model,
        feeds,
        patterns,
        expected_ops,
        *,
        required_pattern: str,
        initializer_count: int | None = None,
        expected=None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        check_model: bool = True,
        schema_lookup: bool = True,
    ):
        """Checks a successful rewrite, topology, initializers, and values."""
        optimized, rewrites, report = self._optimize(
            model, patterns, check_model=check_model, schema_lookup=schema_lookup
        )
        self.assertEqual(expected_ops, [node.op_type for node in optimized.graph.node])
        self.assertIn(required_pattern, {rewrite.pattern_name for rewrite in rewrites})
        self.assertGreaterEqual(report.rewrites, 1)
        if initializer_count is not None:
            self.assertEqual(initializer_count, len(optimized.graph.initializer))
        self._check_equivalent(model, optimized, feeds, expected=expected, atol=atol, rtol=rtol)
        return optimized

    def _assert_no_match(
        self,
        model,
        feeds,
        pattern,
        *,
        check_model: bool = True,
        evaluate: bool = True,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        schema_lookup: bool = True,
    ):
        """Checks that an isolated pattern leaves a model unchanged."""
        expected_ops = [node.op_type for node in model.graph.node]
        optimized, rewrites, report = self._optimize(
            model, [pattern], check_model=check_model, schema_lookup=schema_lookup
        )
        del report
        pattern_name = pattern if isinstance(pattern, str) else pattern.name
        self.assertNotIn(pattern_name, {rewrite.pattern_name for rewrite in rewrites})
        self.assertEqual(expected_ops, [node.op_type for node in optimized.graph.node])
        if evaluate:
            self._check_equivalent(model, optimized, feeds, atol=atol, rtol=rtol)
        return optimized

    def test_gemm_transpose_variants(self):
        for with_bias in (False, True):
            with self.subTest(with_bias=with_bias):
                inputs = [("X", TensorProto.FLOAT, [2, 3])]
                initializers = [self._initializer("B", self._range(3, 2), np.float32)]
                gemm_inputs = ["X", "B"]
                feeds = {"X": self._range(2, 3)}
                if with_bias:
                    inputs.append(("C", TensorProto.FLOAT, [2]))
                    gemm_inputs.append("C")
                    feeds["C"] = self._range(2, bias=0.25)
                model = self._make_model(
                    [helper.make_node("Gemm", gemm_inputs, ["Y"], alpha=0.75)],
                    inputs,
                    [("Y", TensorProto.FLOAT, [2, 2])],
                    initializers,
                )
                optimized = self._optimize_and_check(
                    model,
                    feeds,
                    ["GemmTranspose"],
                    ["Gemm"],
                    required_pattern="GemmTranspose",
                    initializer_count=2,
                )
                attributes = {
                    attribute.name: attribute for attribute in optimized.graph.node[-1].attribute
                }
                self.assertEqual(1, attributes["transB"].i)
                self.assertAlmostEqual(0.75, attributes["alpha"].f)

    def test_gemm_transpose_no_match(self):
        variants = [{"transB": 1}, {"beta": 0.5}]
        for attributes in variants:
            with self.subTest(attributes=attributes):
                model = self._make_model(
                    [helper.make_node("Gemm", ["X", "B"], ["Y"], **attributes)],
                    [("X", TensorProto.FLOAT, [2, 3])],
                    [("Y", TensorProto.FLOAT, [2, 2])],
                    [self._initializer("B", self._range(3, 2), np.float32)],
                )
                self._assert_no_match(model, {"X": self._range(2, 3)}, "GemmTranspose")

        dynamic_weight = self._make_model(
            [helper.make_node("Gemm", ["X", "B"], ["Y"])],
            [("X", TensorProto.FLOAT, [2, 3]), ("B", TensorProto.FLOAT, [3, 2])],
            [("Y", TensorProto.FLOAT, [2, 2])],
        )
        self._assert_no_match(
            dynamic_weight, {"X": self._range(2, 3), "B": self._range(3, 2)}, "GemmTranspose"
        )

    def test_matmul_add_variants(self):
        variants = [
            (
                "MatMul",
                [helper.make_node("MatMul", ["X1", "X2"], ["M"])],
                [
                    ("X1", TensorProto.FLOAT, [2, 3]),
                    ("X2", TensorProto.FLOAT, [3, 2]),
                    ("B", TensorProto.FLOAT, [2]),
                ],
                {"X1": self._range(2, 3), "X2": self._range(3, 2), "B": self._range(2)},
                ["Gemm"],
            ),
            (
                "Gemm",
                [helper.make_node("Gemm", ["X1", "X2"], ["M"], transB=1)],
                [
                    ("X1", TensorProto.FLOAT, [2, 3]),
                    ("X2", TensorProto.FLOAT, [2, 3]),
                    ("B", TensorProto.FLOAT, [2]),
                ],
                {"X1": self._range(2, 3), "X2": self._range(2, 3), "B": self._range(2)},
                ["Gemm"],
            ),
            (
                "GemmBias",
                [helper.make_node("Gemm", ["X1", "X2", "B1"], ["M"], transB=1)],
                [
                    ("X1", TensorProto.FLOAT, [2, 3]),
                    ("X2", TensorProto.FLOAT, [2, 3]),
                    ("B1", TensorProto.FLOAT, [2]),
                    ("B2", TensorProto.FLOAT, [2]),
                ],
                {
                    "X1": self._range(2, 3),
                    "X2": self._range(2, 3),
                    "B1": self._range(2),
                    "B2": self._range(2, bias=10),
                },
                ["Add", "Gemm"],
            ),
        ]
        for name, prefix, inputs, feeds, expected_ops in variants:
            with self.subTest(name=name):
                bias_name = "B2" if name == "GemmBias" else "B"
                model = self._make_model(
                    [*prefix, helper.make_node("Add", ["M", bias_name], ["Y"])],
                    inputs,
                    [("Y", TensorProto.FLOAT, [2, 2])],
                )
                self._optimize_and_check(
                    model,
                    feeds,
                    ["MatMulAdd"],
                    expected_ops,
                    required_pattern="MatMulAdd",
                    initializer_count=0,
                    atol=1e-5,
                )

    def test_matmul_add_no_match(self):
        rank_three = self._make_model(
            [
                helper.make_node("MatMul", ["X1", "X2"], ["M"]),
                helper.make_node("Add", ["M", "B"], ["Y"]),
            ],
            [
                ("X1", TensorProto.FLOAT, [4, 2, 3]),
                ("X2", TensorProto.FLOAT, [3, 2]),
                ("B", TensorProto.FLOAT, [2]),
            ],
            [("Y", TensorProto.FLOAT, [4, 2, 2])],
        )
        self._assert_no_match(
            rank_three,
            {"X1": self._range(4, 2, 3), "X2": self._range(3, 2), "B": self._range(2)},
            "MatMulAdd",
        )

        beta = self._make_model(
            [
                helper.make_node("Gemm", ["X1", "X2"], ["M"], beta=0.5),
                helper.make_node("Add", ["M", "B"], ["Y"]),
            ],
            [
                ("X1", TensorProto.FLOAT, [2, 3]),
                ("X2", TensorProto.FLOAT, [3, 2]),
                ("B", TensorProto.FLOAT, [2]),
            ],
            [("Y", TensorProto.FLOAT, [2, 2])],
        )
        self._assert_no_match(
            beta,
            {"X1": self._range(2, 3), "X2": self._range(3, 2), "B": self._range(2)},
            "MatMulAdd",
        )

        shared = self._make_model(
            [
                helper.make_node("MatMul", ["X1", "X2"], ["M"]),
                helper.make_node("Add", ["M", "B"], ["Y"]),
            ],
            [
                ("X1", TensorProto.FLOAT, [2, 3]),
                ("X2", TensorProto.FLOAT, [3, 2]),
                ("B", TensorProto.FLOAT, [2]),
            ],
            [("Y", TensorProto.FLOAT, [2, 2]), ("M", TensorProto.FLOAT, [2, 2])],
        )
        self._assert_no_match(
            shared,
            {"X1": self._range(2, 3), "X2": self._range(3, 2), "B": self._range(2)},
            "MatMulAdd",
        )

    def test_matmul_add_reshape_1(self):
        model = self._make_model(
            [
                helper.make_node("MatMul", ["X1", "X2"], ["M"]),
                helper.make_node("Add", ["M", "B"], ["Y"]),
            ],
            [
                ("X1", TensorProto.FLOAT, [4, 2, 3]),
                ("X2", TensorProto.FLOAT, [3, 2]),
                ("B", TensorProto.FLOAT, [2]),
            ],
            [("Y", TensorProto.FLOAT, [4, 2, 2])],
        )
        self._optimize_and_check(
            model,
            {"X1": self._range(4, 2, 3), "X2": self._range(3, 2), "B": self._range(2)},
            [optimization.MatMulAddPattern(priority=3, allow_reshape=True)],
            ["Reshape", "Gemm", "Reshape"],
            required_pattern="MatMulAdd",
            atol=1e-5,
        )

    def test_matmul_add_reshape_2(self):
        model = self._make_model(
            [
                helper.make_node("MatMul", ["X1", "X2"], ["M"]),
                helper.make_node("Add", ["M", "B"], ["Y"]),
            ],
            [
                ("X1", TensorProto.FLOAT, [4, 2, 3]),
                ("X2", TensorProto.FLOAT, [3, 2]),
                ("B", TensorProto.FLOAT, [4, 2, 2]),
            ],
            [("Y", TensorProto.FLOAT, [4, 2, 2])],
        )
        self._optimize_and_check(
            model,
            {"X1": self._range(4, 2, 3), "X2": self._range(3, 2), "B": self._range(4, 2, 2)},
            [optimization.MatMulAddPattern(priority=3, allow_reshape=True)],
            ["Reshape", "Reshape", "Gemm", "Reshape"],
            required_pattern="MatMulAdd",
            atol=1e-5,
        )

    def test_matmul_add_reshape_2_dyn(self):
        model = self._make_model(
            [
                helper.make_node("MatMul", ["X1", "X2"], ["M"]),
                helper.make_node("Add", ["M", "B"], ["Y"]),
            ],
            [
                ("X1", TensorProto.FLOAT, ["a", "b", 3]),
                ("X2", TensorProto.FLOAT, [3, "d"]),
                ("B", TensorProto.FLOAT, ["a", "b", "d"]),
            ],
            [("Y", TensorProto.FLOAT, ["a", "b", "d"])],
        )
        self._optimize_and_check(
            model,
            {"X1": self._range(4, 2, 3), "X2": self._range(3, 2), "B": self._range(4, 2, 2)},
            [optimization.MatMulAddPattern(priority=3, allow_reshape=True)],
            ["Reshape", "Shape", "Concat", "Reshape", "Shape", "Concat", "Gemm", "Reshape"],
            required_pattern="MatMulAdd",
            atol=1e-5,
        )

    def test_mul_mul_matmul_variants(self):
        variants = [
            (["X", "left_scale"], ["right_scale", "Y"]),
            (["left_scale", "X"], ["Y", "right_scale"]),
        ]
        for left_inputs, right_inputs in variants:
            with self.subTest(left=left_inputs, right=right_inputs):
                model = self._make_model(
                    [
                        helper.make_node("Mul", left_inputs, ["A"]),
                        helper.make_node("Mul", right_inputs, ["B"]),
                        helper.make_node("MatMul", ["A", "B"], ["M"]),
                        helper.make_node("Add", ["M", "M"], ["Yout"]),
                    ],
                    [("X", TensorProto.FLOAT, [4, 3]), ("Y", TensorProto.FLOAT, [3, 5])],
                    [("Yout", TensorProto.FLOAT, [4, 5])],
                    [
                        self._initializer("left_scale", [0.4], np.float32),
                        self._initializer("right_scale", [0.6], np.float32),
                    ],
                )
                self._optimize_and_check(
                    model,
                    {"X": self._range(4, 3), "Y": self._range(3, 5)},
                    ["MulMulMatMul"],
                    ["MatMul", "Mul", "Add"],
                    required_pattern="MulMulMatMul",
                    initializer_count=3,
                    atol=1e-5,
                )

    def test_mul_mul_matmul_no_match(self):
        vector_scale = self._make_model(
            [
                helper.make_node("Mul", ["X", "left_scale"], ["A"]),
                helper.make_node("Mul", ["Y", "right_scale"], ["B"]),
                helper.make_node("MatMul", ["A", "B"], ["Z"]),
            ],
            [("X", TensorProto.FLOAT, [4, 3]), ("Y", TensorProto.FLOAT, [3, 5])],
            [("Z", TensorProto.FLOAT, [4, 5])],
            [
                self._initializer("left_scale", [0.4, 0.5, 0.6], np.float32),
                self._initializer("right_scale", [0.6], np.float32),
            ],
        )
        self._assert_no_match(
            vector_scale, {"X": self._range(4, 3), "Y": self._range(3, 5)}, "MulMulMatMul"
        )

        shared = self._make_model(
            [
                helper.make_node("Mul", ["X", "left_scale"], ["A"]),
                helper.make_node("Mul", ["Y", "right_scale"], ["B"]),
                helper.make_node("MatMul", ["A", "B"], ["Z"]),
            ],
            [("X", TensorProto.FLOAT, [4, 3]), ("Y", TensorProto.FLOAT, [3, 5])],
            [("Z", TensorProto.FLOAT, [4, 5]), ("A", TensorProto.FLOAT, [4, 3])],
            [
                self._initializer("left_scale", [0.4], np.float32),
                self._initializer("right_scale", [0.6], np.float32),
            ],
        )
        self._assert_no_match(
            shared, {"X": self._range(4, 3), "Y": self._range(3, 5)}, "MulMulMatMul"
        )

    def _make_transpose_matmul(self, topology: str, *, opset: int = 18, bad_perm=False):
        nodes = []
        inputs = []
        feeds = {}
        left = "X"
        right = "Y"
        if topology in {"left", "both", "shared"}:
            perm = [0, 1] if bad_perm else [1, 0]
            nodes.append(helper.make_node("Transpose", ["X"], ["Xt"], perm=perm))
            left = "Xt"
            input_shape = [2, 3] if bad_perm else [3, 2]
            inputs.append(("X", TensorProto.FLOAT, input_shape))
            feeds["X"] = self._range(*input_shape)
        else:
            inputs.append(("X", TensorProto.FLOAT, [2, 3]))
            feeds["X"] = self._range(2, 3)
        if topology in {"right", "both"}:
            nodes.append(helper.make_node("Transpose", ["Y"], ["Yt"], perm=[1, 0]))
            right = "Yt"
            inputs.append(("Y", TensorProto.FLOAT, [4, 3]))
            feeds["Y"] = self._range(4, 3)
        else:
            inputs.append(("Y", TensorProto.FLOAT, [3, 4]))
            feeds["Y"] = self._range(3, 4)
        nodes.append(helper.make_node("MatMul", [left, right], ["Z"]))
        outputs = [("Z", TensorProto.FLOAT, [2, 4])]
        if topology == "shared":
            nodes.append(helper.make_node("Transpose", ["Xt"], ["W"], perm=[1, 0]))
            outputs.append(("W", TensorProto.FLOAT, [3, 2]))
        return self._make_model(nodes, inputs, outputs, opset=opset), feeds

    def test_transpose_matmul_variants(self):
        expected = {
            "left": ["Gemm"],
            "right": ["Gemm"],
            "both": ["Gemm"],
            "shared": ["Gemm", "Transpose", "Transpose"],
        }
        for topology in expected:
            with self.subTest(topology=topology):
                model, feeds = self._make_transpose_matmul(topology)
                self._optimize_and_check(
                    model,
                    feeds,
                    ["TransposeMatMul"],
                    expected[topology],
                    required_pattern="TransposeMatMul",
                    initializer_count=0,
                )

        model = self._make_model(
            [
                helper.make_node("Transpose", ["X"], ["Xt"], perm=[1, 0]),
                helper.make_node(
                    "Gemm", ["Xt", "Y"], ["Z"], alpha=2.0, beta=0.0, transA=1, transB=1
                ),
            ],
            [("X", TensorProto.FLOAT, [2, 3]), ("Y", TensorProto.FLOAT, [4, 3])],
            [("Z", TensorProto.FLOAT, [2, 4])],
        )
        optimized = self._optimize_and_check(
            model,
            {"X": self._range(2, 3), "Y": self._range(4, 3)},
            ["TransposeMatMul"],
            ["Gemm"],
            required_pattern="TransposeMatMul",
        )
        attributes = {
            attribute.name: attribute for attribute in optimized.graph.node[0].attribute
        }
        self.assertEqual(0, attributes["transA"].i)
        self.assertEqual(1, attributes["transB"].i)
        self.assertAlmostEqual(2.0, attributes["alpha"].f)
        self.assertAlmostEqual(0.0, attributes["beta"].f)

    def test_transpose_matmul_no_match(self):
        bad_perm, feeds = self._make_transpose_matmul("left", bad_perm=True)
        self._assert_no_match(bad_perm, feeds, "TransposeMatMul")

        old_opset, feeds = self._make_transpose_matmul("left", opset=10)
        self._assert_no_match(old_opset, feeds, "TransposeMatMul")

        rank_three = self._make_model(
            [
                helper.make_node("Transpose", ["X"], ["Xt"], perm=[0, 2, 1]),
                helper.make_node("MatMul", ["Xt", "Y"], ["Z"]),
            ],
            [("X", TensorProto.FLOAT, [2, 4, 3]), ("Y", TensorProto.FLOAT, [2, 4, 5])],
            [("Z", TensorProto.FLOAT, [2, 3, 5])],
        )
        self._assert_no_match(
            rank_three, {"X": self._range(2, 4, 3), "Y": self._range(2, 4, 5)}, "TransposeMatMul"
        )

    def _make_transpose_reshape_matmul(self, side: str, *, shared=False):
        if side == "left":
            nodes = [
                helper.make_node("Transpose", ["X"], ["Xt"], perm=[0, 2, 1]),
                helper.make_node("Reshape", ["Xt", "shape"], ["Xtr"]),
                helper.make_node("MatMul", ["Xtr", "Y"], ["Z"]),
            ]
            inputs = [("X", TensorProto.FLOAT, [4, 5, 7]), ("Y", TensorProto.FLOAT, [2, 2, 5, 3])]
            outputs = [("Z", TensorProto.FLOAT, [2, 2, 7, 3])]
            feeds = {"X": self._range(4, 5, 7), "Y": self._range(2, 2, 5, 3)}
            shape = [2, 2, 7, 5]
            shared_name = "Xt"
            shared_shape = [4, 7, 5]
        else:
            nodes = [
                helper.make_node("Transpose", ["Y"], ["Yt"], perm=[0, 2, 1]),
                helper.make_node("Reshape", ["Yt", "shape"], ["Ytr"]),
                helper.make_node("MatMul", ["X", "Ytr"], ["Z"]),
            ]
            inputs = [("X", TensorProto.FLOAT, [2, 2, 5, 7]), ("Y", TensorProto.FLOAT, [4, 3, 7])]
            outputs = [("Z", TensorProto.FLOAT, [2, 2, 5, 3])]
            feeds = {"X": self._range(2, 2, 5, 7), "Y": self._range(4, 3, 7)}
            shape = [2, 2, 7, 3]
            shared_name = "Yt"
            shared_shape = [4, 7, 3]
        if shared:
            outputs.append((shared_name, TensorProto.FLOAT, shared_shape))
        return (
            self._make_model(
                nodes, inputs, outputs, [self._initializer("shape", shape, np.int64)]
            ),
            feeds,
        )

    def test_transpose_reshape_matmul_variants(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                model, feeds = self._make_transpose_reshape_matmul(side)
                self._optimize_and_check(
                    model,
                    feeds,
                    ["TransposeReshapeMatMul"],
                    ["Reshape", "Transpose", "MatMul"],
                    required_pattern="TransposeReshapeMatMul",
                    initializer_count=2,
                )

    def test_transpose_reshape_matmul_no_match(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                model, feeds = self._make_transpose_reshape_matmul(side, shared=True)
                self._assert_no_match(model, feeds, "TransposeReshapeMatMul")

    def _make_switch_reshape_activation(self, layout: str, activation: str):
        if layout == "Transpose":
            nodes = [
                helper.make_node("MatMul", ["X", "Y"], ["M"]),
                helper.make_node("Transpose", ["M"], ["L"], perm=[0, 2, 1, 3]),
                helper.make_node(activation, ["L"], ["Z"]),
            ]
            inputs = [
                ("X", TensorProto.FLOAT, [3, 2, 6, 5]),
                ("Y", TensorProto.FLOAT, [3, 2, 5, 6]),
            ]
            outputs = [("Z", TensorProto.FLOAT, [3, 6, 2, 6])]
            initializers = []
            feeds = {"X": self._range(3, 2, 6, 5), "Y": self._range(3, 2, 5, 6)}
        else:
            nodes = [
                helper.make_node("MatMul", ["X", "Y"], ["M"]),
                helper.make_node("Reshape", ["M", "shape"], ["L"]),
                helper.make_node(activation, ["L"], ["Z"]),
            ]
            inputs = [("X", TensorProto.FLOAT, [2, 3]), ("Y", TensorProto.FLOAT, [3, 4])]
            outputs = [("Z", TensorProto.FLOAT, [4, 2])]
            initializers = [self._initializer("shape", [4, 2], np.int64)]
            feeds = {"X": self._range(2, 3, bias=-0.5), "Y": self._range(3, 4)}
        return self._make_model(nodes, inputs, outputs, initializers), feeds

    def test_switch_reshape_activation_variants(self):
        for layout, activation in (("Transpose", "Relu"), ("Reshape", "Tanh")):
            with self.subTest(layout=layout, activation=activation):
                model, feeds = self._make_switch_reshape_activation(layout, activation)
                self._optimize_and_check(
                    model,
                    feeds,
                    ["SwitchReshapeActivation"],
                    ["MatMul", activation, layout],
                    required_pattern="SwitchReshapeActivation",
                    initializer_count=0 if layout == "Transpose" else 1,
                )

    def test_switch_reshape_activation_no_match(self):
        for activation in ("Softmax", "PRelu"):
            with self.subTest(activation=activation):
                model, feeds = self._make_switch_reshape_activation("Transpose", "Relu")
                model.graph.node[-1].op_type = activation
                if activation == "PRelu":
                    model.graph.node[-1].input.append("slope")
                    model.graph.initializer.append(self._initializer("slope", [0.1], np.float32))
                self._assert_no_match(model, feeds, "SwitchReshapeActivation")

        model, feeds = self._make_switch_reshape_activation("Transpose", "Relu")
        model.graph.output.append(self._value_info("L", TensorProto.FLOAT, [3, 6, 2, 6]))
        self._assert_no_match(model, feeds, "SwitchReshapeActivation")

    def test_batch_normalization_identity_and_no_match(self):
        neutral = self._make_model(
            [
                helper.make_node(
                    "BatchNormalization",
                    ["X", "scale", "bias", "mean", "variance"],
                    ["Y"],
                    epsilon=0.0,
                )
            ],
            [("X", TensorProto.FLOAT, [2, 3, 4, 5])],
            [("Y", TensorProto.FLOAT, [2, 3, 4, 5])],
            [
                self._initializer("scale", np.ones(3), np.float32),
                self._initializer("bias", np.zeros(3), np.float32),
                self._initializer("mean", np.zeros(3), np.float32),
                self._initializer("variance", np.ones(3), np.float32),
            ],
        )
        feeds = {"X": self._range(2, 3, 4, 5)}
        self._optimize_and_check(
            neutral,
            feeds,
            ["BatchNormalization"],
            ["Identity"],
            required_pattern="BatchNormalization",
            initializer_count=2,
            schema_lookup=False,
        )

        for changed in ("epsilon", "scale", "training"):
            with self.subTest(changed=changed):
                model = self._make_model(
                    [
                        helper.make_node(
                            "BatchNormalization",
                            ["X", "scale", "bias", "mean", "variance"],
                            ["Y"],
                            epsilon=0.1 if changed == "epsilon" else 0.0,
                            training_mode=1 if changed == "training" else 0,
                        )
                    ],
                    [("X", TensorProto.FLOAT, [2, 3, 4, 5])],
                    [("Y", TensorProto.FLOAT, [2, 3, 4, 5])],
                    [
                        self._initializer(
                            "scale",
                            np.full(3, 2.0) if changed == "scale" else np.ones(3),
                            np.float32,
                        ),
                        self._initializer("bias", np.zeros(3), np.float32),
                        self._initializer("mean", np.zeros(3), np.float32),
                        self._initializer("variance", np.ones(3), np.float32),
                    ],
                )
                self._assert_no_match(model, feeds, "BatchNormalization", schema_lookup=False)

        identity_pattern_topology = self._make_model(
            [
                helper.make_node(
                    "BatchNormalization",
                    ["X", "scale", "bias", "mean", "variance"],
                    ["normalized"],
                ),
                helper.make_node("Neg", ["normalized"], ["Y"]),
            ],
            [("X", TensorProto.FLOAT16, [6, 2])],
            [("Y", TensorProto.FLOAT16, [6, 2])],
            [
                self._initializer("scale", np.ones(2), np.float16),
                self._initializer("bias", np.zeros(2), np.float16),
                self._initializer("mean", np.zeros(2), np.float16),
                self._initializer("variance", np.ones(2), np.float16),
            ],
            opset=20,
        )
        self._assert_no_match(
            identity_pattern_topology,
            {"X": self._range(6, 2, dtype=np.float16)},
            "BatchNormalization",
            schema_lookup=False,
            atol=1e-3,
            rtol=1e-3,
        )

    @staticmethod
    def _batch_normalization_training_expected(x, scale, bias, epsilon):
        """Computes the training-mode BatchNormalization data output."""
        axes = (0, *range(2, x.ndim))
        mean = np.mean(x, axis=axes, keepdims=True)
        variance = np.mean((x - mean) ** 2, axis=axes, keepdims=True)
        return ((x - mean) / np.sqrt(variance + epsilon) * scale + bias).astype(x.dtype)

    def test_batch_normalization_training_variants(self):
        x = self._range(2, 3, 4, 5)
        variants = [
            (
                "rank_one_parameters",
                np.arange(3, dtype=np.float32) + 1,
                np.arange(3, dtype=np.float32) + 100,
                ["ReduceMean", "Sub", "Mul", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add"],
                True,
            ),
            (
                "broadcast_parameters",
                self._range(1, 3, 4, 5, bias=1),
                self._range(1, 3, 4, 1, bias=100),
                ["ReduceMean", "Sub", "Mul", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add"],
                False,
            ),
        ]
        for name, scale, bias, expected_ops, schema_valid in variants:
            with self.subTest(name=name):
                model = self._make_model(
                    [
                        helper.make_node(
                            "BatchNormalization",
                            ["X", "scale", "bias", "mean", "variance"],
                            ["Y", "unused_mean", "unused_variance"],
                            epsilon=0.5,
                            momentum=1.0,
                            training_mode=1,
                        )
                    ],
                    [("X", TensorProto.FLOAT, [2, 3, 4, 5])],
                    [("Y", TensorProto.FLOAT, [2, 3, 4, 5])],
                    [
                        self._initializer("scale", scale, np.float32),
                        self._initializer("bias", bias, np.float32),
                        self._initializer("mean", np.arange(3) + 20, np.float32),
                        self._initializer("variance", np.arange(3) + 2, np.float32),
                    ],
                )
                broadcast_scale = scale.reshape(1, 3, 1, 1) if scale.ndim == 1 else scale
                broadcast_bias = bias.reshape(1, 3, 1, 1) if bias.ndim == 1 else bias
                expected = [
                    self._batch_normalization_training_expected(
                        x, broadcast_scale, broadcast_bias, 0.5
                    )
                ]
                self._optimize_and_check(
                    model,
                    {"X": x},
                    ["BatchNormalizationTraining"],
                    expected_ops,
                    required_pattern="BatchNormalizationTraining",
                    expected=expected,
                    atol=1e-5,
                    check_model=schema_valid,
                )

    def test_batch_normalization_training_no_match(self):
        variants = [(17, 1, False), (18, 0, False), (18, 1, True)]
        for opset, training_mode, use_running_mean in variants:
            with self.subTest(
                opset=opset, training_mode=training_mode, use_running_mean=use_running_mean
            ):
                outputs = [("Y", TensorProto.FLOAT, [2, 3, 4, 5])]
                node_outputs = ["Y", "running_mean", "running_variance"]
                if use_running_mean:
                    outputs.append(("running_mean", TensorProto.FLOAT, [3]))
                model = self._make_model(
                    [
                        helper.make_node(
                            "BatchNormalization",
                            ["X", "scale", "bias", "mean", "variance"],
                            node_outputs,
                            training_mode=training_mode,
                        )
                    ],
                    [("X", TensorProto.FLOAT, [2, 3, 4, 5])],
                    outputs,
                    [
                        self._initializer("scale", np.ones(3), np.float32),
                        self._initializer("bias", np.zeros(3), np.float32),
                        self._initializer("mean", np.zeros(3), np.float32),
                        self._initializer("variance", np.ones(3), np.float32),
                    ],
                    opset=opset,
                )
                self._assert_no_match(
                    model, {"X": self._range(2, 3, 4, 5)}, "BatchNormalizationTraining"
                )

    def _make_layer_normalization_decomposition(
        self,
        *,
        dtype,
        axis: int = -1,
        dynamic: bool = False,
        reciprocal: bool = False,
        epsilon: float | None = None,
        first_keepdims: bool = True,
        expose_mean: bool = False,
    ):
        data_type = (
            TensorProto.FLOAT16 if np.dtype(dtype) == np.dtype(np.float16) else TensorProto.FLOAT
        )
        axes = [-1] if axis == -1 else [0, 1]
        nodes = [
            helper.make_node(
                "ReduceMean",
                ["X", "axes"],
                ["mean"],
                **({"keepdims": 1} if first_keepdims else {}),
            ),
            helper.make_node("Sub", ["X", "mean"], ["centered"]),
            helper.make_node("Pow", ["centered", "two"], ["squared"]),
            helper.make_node("ReduceMean", ["squared", "axes"], ["variance"], keepdims=1),
        ]
        sqrt_input = "variance"
        initializers = [
            self._initializer("axes", axes, np.int64),
            self._initializer("two", [2], np.float32),
        ]
        if epsilon is not None:
            nodes.append(helper.make_node("Add", ["variance", "epsilon"], ["variance_epsilon"]))
            sqrt_input = "variance_epsilon"
            initializers.append(self._initializer("epsilon", [epsilon], dtype))
        nodes.append(helper.make_node("Sqrt", [sqrt_input], ["standard_deviation"]))
        if reciprocal:
            nodes.extend(
                [
                    helper.make_node(
                        "Reciprocal", ["standard_deviation"], ["inverse_standard_deviation"]
                    ),
                    helper.make_node("Mul", ["centered", "inverse_standard_deviation"], ["Y"]),
                ]
            )
        else:
            nodes.append(helper.make_node("Div", ["centered", "standard_deviation"], ["Y"]))
        shape = ["rows", "columns"] if dynamic else [2, 3]
        outputs = [("Y", data_type, shape)]
        if expose_mean:
            outputs.append(("mean", data_type, [1, 1] if axis == 0 else [2, 1]))
        return self._make_model(nodes, [("X", data_type, shape)], outputs, initializers, opset=20)

    def test_layer_normalization_reconstructed_minimal_variants(self):
        for reciprocal, epsilon in ((False, 1e-4), (True, None)):
            with self.subTest(reciprocal=reciprocal, epsilon=epsilon):
                model = self._make_layer_normalization_decomposition(
                    dtype=np.float32, reciprocal=reciprocal, epsilon=epsilon
                )
                self._optimize_and_check(
                    model,
                    {"X": self._range(2, 3, bias=1)},
                    ["LayerNormalization"],
                    ["LayerNormalization"],
                    required_pattern="LayerNormalization",
                    initializer_count=5 if epsilon is not None else 4,
                    atol=1e-5,
                )

    def test_layer_normalization_upstream_type_shape_matrix(self):
        for dtype in (np.float32, np.float16):
            for axis in (-1, 0):
                for dynamic in (False, True):
                    with self.subTest(dtype=dtype, axis=axis, dynamic=dynamic):
                        model = self._make_layer_normalization_decomposition(
                            dtype=dtype, axis=axis, dynamic=dynamic
                        )
                        expected_ops = (
                            ["LayerNormalization"]
                            if axis == -1 and not dynamic
                            else [
                                "Shape",
                                "ConstantOfShape",
                                "ConstantOfShape",
                                "LayerNormalization",
                            ]
                        )
                        self._optimize_and_check(
                            model,
                            {
                                "X": (self._range(2, 3, bias=1) + np.array([1, 2, 3])).astype(
                                    dtype
                                )
                            },
                            ["LayerNormalization"],
                            expected_ops,
                            required_pattern="LayerNormalization",
                            initializer_count=4 if expected_ops == ["LayerNormalization"] else 2,
                            atol=1e-3 if dtype == np.float16 else 1e-5,
                            rtol=1e-3 if dtype == np.float16 else 1e-5,
                        )

    def test_layer_normalization_no_match(self):
        implicit_keepdims = self._make_layer_normalization_decomposition(
            dtype=np.float32, first_keepdims=False
        )
        self._assert_no_match(
            implicit_keepdims, {"X": self._range(2, 3, bias=1)}, "LayerNormalization"
        )

        non_suffix = self._make_layer_normalization_decomposition(dtype=np.float32, axis=-1)
        non_suffix.graph.initializer[0].CopyFrom(self._initializer("axes", [0], np.int64))
        self._assert_no_match(non_suffix, {"X": self._range(2, 3, bias=1)}, "LayerNormalization")

        exposed = self._make_layer_normalization_decomposition(dtype=np.float32, expose_mean=True)
        self._assert_no_match(exposed, {"X": self._range(2, 3, bias=1)}, "LayerNormalization")

    def _make_layer_normalization_scale(
        self,
        *,
        with_bias: bool,
        axis: int = -1,
        epsilon: float = 1e-5,
        stash_type: int = TensorProto.FLOAT,
        mismatch: bool = False,
    ):
        nodes = [
            helper.make_node(
                "LayerNormalization",
                ["X", "scale0"] + (["bias0"] if with_bias else []),
                ["normalized"],
                axis=axis,
                epsilon=epsilon,
                stash_type=stash_type,
            ),
            helper.make_node("Mul", ["normalized", "scale1"], ["scaled"]),
        ]
        initializers = [
            self._initializer("scale0", [-0.1, -0.01, -0.05], np.float32),
            self._initializer("scale1", [2.0] if mismatch else [2.0, 3.0, 4.0], np.float32),
        ]
        output_name = "scaled"
        if with_bias:
            nodes.append(helper.make_node("Add", ["scaled", "bias1"], ["Y"]))
            initializers.extend(
                [
                    self._initializer("bias0", [0.5, 0.6, 0.7], np.float32),
                    self._initializer("bias1", [2.0, 3.0, 4.0], np.float32),
                ]
            )
            output_name = "Y"
        return self._make_model(
            nodes,
            [("X", TensorProto.FLOAT, [2, 3])],
            [(output_name, TensorProto.FLOAT, [2, 3])],
            initializers,
            opset=20,
        )

    def test_layer_normalization_scale_variants(self):
        variants = [
            (False, -1, 1e-5, TensorProto.FLOAT),
            (False, 1, 0.1, TensorProto.FLOAT),
            (True, -1, 1e-5, TensorProto.FLOAT),
            (True, 1, 0.1, TensorProto.FLOAT),
        ]
        for with_bias, axis, epsilon, stash_type in variants:
            with self.subTest(
                with_bias=with_bias, axis=axis, epsilon=epsilon, stash_type=stash_type
            ):
                model = self._make_layer_normalization_scale(
                    with_bias=with_bias, axis=axis, epsilon=epsilon, stash_type=stash_type
                )
                optimized = self._optimize_and_check(
                    model,
                    {"X": self._range(2, 3, bias=1)},
                    ["LayerNormalizationScale"],
                    ["LayerNormalization"],
                    required_pattern="LayerNormalizationScale",
                    atol=1e-5,
                    schema_lookup=False,
                )
                layer = optimized.graph.node[-1]
                attributes = {attribute.name: attribute for attribute in layer.attribute}
                self.assertEqual(axis, attributes["axis"].i)
                self.assertAlmostEqual(epsilon, attributes["epsilon"].f, places=6)
                self.assertEqual(stash_type, attributes["stash_type"].i)

    def test_layer_normalization_scale_no_match(self):
        model = self._make_layer_normalization_scale(with_bias=False, mismatch=True)
        self._assert_no_match(
            model,
            {"X": self._range(2, 3, bias=1)},
            "LayerNormalizationScale",
            schema_lookup=False,
        )

        shared = self._make_layer_normalization_scale(with_bias=False)
        shared.graph.output.append(self._value_info("normalized", TensorProto.FLOAT, [2, 3]))
        self._assert_no_match(
            shared,
            {"X": self._range(2, 3, bias=1)},
            "LayerNormalizationScale",
            schema_lookup=False,
        )

    def test_cast_layer_normalization_cast_variants(self):
        layer = self._make_model(
            [
                helper.make_node("Cast", ["X"], ["X32"], to=TensorProto.FLOAT),
                helper.make_node(
                    "LayerNormalization",
                    ["X32", "scale", "bias"],
                    ["normalized"],
                    stash_type=TensorProto.FLOAT,
                ),
                helper.make_node("Cast", ["normalized"], ["Y"], to=TensorProto.FLOAT16),
            ],
            [("X", TensorProto.FLOAT16, [3, 3])],
            [("Y", TensorProto.FLOAT16, [3, 3])],
            [
                self._initializer("scale", [0.5, 0.6, 0.7], np.float32),
                self._initializer("bias", [-0.5, -0.6, -0.7], np.float32),
            ],
            opset=18,
        )
        self._optimize_and_check(
            layer,
            {"X": self._range(3, 3, dtype=np.float16)},
            ["CastLayerNormalizationCast"],
            ["LayerNormalization"],
            required_pattern="CastLayerNormalizationCast",
            initializer_count=4,
            atol=1e-2,
            rtol=1e-2,
            schema_lookup=False,
        )

        rms = self._make_model(
            [
                helper.make_node("Cast", ["X"], ["X32"], to=TensorProto.FLOAT),
                helper.make_node(
                    "RMSNormalization",
                    ["X32", "scale"],
                    ["normalized"],
                    stash_type=TensorProto.FLOAT,
                ),
                helper.make_node("Cast", ["normalized"], ["Y"], to=TensorProto.FLOAT16),
            ],
            [("X", TensorProto.FLOAT16, [3, 3])],
            [("Y", TensorProto.FLOAT16, [3, 3])],
            [self._initializer("scale", [0.5, 0.6, 0.7], np.float32)],
            opset=24,
        )
        self._optimize_and_check(
            rms,
            {"X": self._range(3, 3, dtype=np.float16)},
            ["CastLayerNormalizationCast"],
            ["RMSNormalization"],
            required_pattern="CastLayerNormalizationCast",
            initializer_count=2,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_cast_layer_normalization_cast_no_match(self):
        model = self._make_model(
            [
                helper.make_node("Cast", ["X"], ["X32"], to=TensorProto.FLOAT),
                helper.make_node(
                    "LayerNormalization",
                    ["X32", "scale"],
                    ["normalized"],
                    stash_type=TensorProto.FLOAT,
                ),
                helper.make_node("Cast", ["normalized"], ["Y"], to=TensorProto.FLOAT),
            ],
            [("X", TensorProto.FLOAT16, [3, 3])],
            [("Y", TensorProto.FLOAT, [3, 3])],
            [self._initializer("scale", [0.5, 0.6, 0.7], np.float32)],
            opset=18,
        )
        self._assert_no_match(
            model,
            {"X": self._range(3, 3, dtype=np.float16)},
            "CastLayerNormalizationCast",
            atol=1e-3,
            rtol=1e-3,
            schema_lookup=False,
        )

    def _make_rms_normalization_decomposition(
        self,
        *,
        divide: bool,
        dynamic: bool,
        cast: bool,
        opset: int = 23,
        axes=(-1,),
        keepdims: int = 1,
        expose_add: bool = False,
        static_shape=(5, 4),
    ):
        input_type = TensorProto.FLOAT16 if cast else TensorProto.FLOAT
        data = "X32" if cast else "X"
        nodes = []
        if cast:
            nodes.append(helper.make_node("Cast", ["X"], ["X32"], to=TensorProto.FLOAT))
        nodes.extend(
            [
                helper.make_node("Pow", [data, "two"], ["squared"]),
                helper.make_node("ReduceMean", ["squared", "axes"], ["mean"], keepdims=keepdims),
                helper.make_node("Add", ["mean", "epsilon"], ["mean_epsilon"]),
                helper.make_node("Sqrt", ["mean_epsilon"], ["root"]),
            ]
        )
        if divide:
            nodes.append(helper.make_node("Div", ["one", "root"], ["inverse"]))
        else:
            nodes.append(helper.make_node("Reciprocal", ["root"], ["inverse"]))
        nodes.append(helper.make_node("Mul", ["inverse", data], ["normalized"]))
        output_name = "normalized"
        if cast:
            nodes.append(helper.make_node("Cast", ["normalized"], ["Y"], to=TensorProto.FLOAT16))
            output_name = "Y"
        shape = ["rows", "columns"] if dynamic else list(static_shape)
        outputs = [(output_name, input_type, shape)]
        if expose_add:
            outputs.append(("mean_epsilon", TensorProto.FLOAT, [5, 1]))
        return self._make_model(
            nodes,
            [("X", input_type, shape)],
            outputs,
            [
                self._initializer("two", [2], np.float32),
                self._initializer("epsilon", [1e-6], np.float32),
                self._initializer("axes", axes, np.int64),
                self._initializer("one", [1], np.float32),
            ],
            opset=opset,
        )

    def test_rms_normalization_variants(self):
        for divide in (False, True):
            for dynamic in (False, True):
                for cast in (False, True):
                    with self.subTest(divide=divide, dynamic=dynamic, cast=cast):
                        model = self._make_rms_normalization_decomposition(
                            divide=divide, dynamic=dynamic, cast=cast
                        )
                        expected_ops = (
                            ["Shape", "ConstantOfShape", "RMSNormalization"]
                            if dynamic
                            else ["RMSNormalization"]
                        )
                        dtype = np.float16 if cast else np.float32
                        self._optimize_and_check(
                            model,
                            {"X": self._range(5, 4, dtype=dtype)},
                            ["RMSNormalization"],
                            expected_ops,
                            required_pattern="RMSNormalization",
                            initializer_count=4 if dynamic else 5,
                            atol=1e-3 if cast else 1e-5,
                            rtol=1e-3 if cast else 1e-5,
                        )

    def test_rms_normalization_no_match(self):
        variants = [
            self._make_rms_normalization_decomposition(
                divide=False, dynamic=False, cast=False, opset=22
            ),
            self._make_rms_normalization_decomposition(
                divide=False, dynamic=False, cast=False, axes=(0,)
            ),
            self._make_rms_normalization_decomposition(
                divide=False, dynamic=False, cast=False, keepdims=0, static_shape=(5, 5)
            ),
            self._make_rms_normalization_decomposition(
                divide=False, dynamic=False, cast=False, expose_add=True
            ),
        ]
        for index, model in enumerate(variants):
            with self.subTest(index=index):
                shape = (5, 5) if index == 2 else (5, 4)
                self._assert_no_match(model, {"X": self._range(*shape)}, "RMSNormalization")

    def test_rms_normalization_mul_variants(self):
        for dtype, data_type, tolerance in (
            (np.float32, TensorProto.FLOAT, 1e-6),
            (np.float16, TensorProto.FLOAT16, 1e-3),
        ):
            with self.subTest(dtype=dtype):
                model = self._make_model(
                    [
                        helper.make_node("RMSNormalization", ["X", "scale0"], ["normalized"]),
                        helper.make_node("Mul", ["normalized", "scale1"], ["Y"]),
                    ],
                    [("X", data_type, [2, 2])],
                    [("Y", data_type, [2, 2])],
                    [
                        self._initializer("scale0", [3, 4], dtype),
                        self._initializer("scale1", [3, 4], dtype),
                    ],
                    opset=23,
                )
                optimized = self._optimize_and_check(
                    model,
                    {"X": np.array([[0, 1], [2, 3]], dtype=dtype)},
                    ["RMSNormalizationMul"],
                    ["RMSNormalization"],
                    required_pattern="RMSNormalizationMul",
                    initializer_count=2,
                    atol=tolerance,
                    rtol=tolerance,
                )
                scales = [
                    numpy_helper.to_array(initializer)
                    for initializer in optimized.graph.initializer
                ]
                self.assertTrue(
                    any(
                        np.array_equal(np.array([9, 16], dtype=dtype), scale) for scale in scales
                    ),
                    scales,
                )

    def test_rms_normalization_mul_no_match(self):
        add = self._make_model(
            [
                helper.make_node("RMSNormalization", ["X", "scale"], ["normalized"]),
                helper.make_node("Add", ["normalized", "bias"], ["Y"]),
            ],
            [("X", TensorProto.FLOAT, [2, 2])],
            [("Y", TensorProto.FLOAT, [2, 2])],
            [
                self._initializer("scale", [3, 4], np.float32),
                self._initializer("bias", [1, 2], np.float32),
            ],
            opset=23,
        )
        self._assert_no_match(add, {"X": self._range(2, 2)}, "RMSNormalizationMul")

        mismatch = self._make_model(
            [
                helper.make_node("RMSNormalization", ["X", "scale0"], ["normalized"]),
                helper.make_node("Mul", ["normalized", "scale1"], ["Y"]),
            ],
            [("X", TensorProto.FLOAT, [2, 2])],
            [("Y", TensorProto.FLOAT, [2, 2])],
            [
                self._initializer("scale0", [3, 4], np.float32),
                self._initializer("scale1", [2], np.float32),
            ],
            opset=23,
        )
        self._assert_no_match(mismatch, {"X": self._range(2, 2)}, "RMSNormalizationMul")

    def _make_gelu_decomposition(
        self,
        *,
        dtype,
        opset: int = 20,
        cubic_scale: float = 0.044708251953125,
        expose_power: bool = False,
    ):
        data_type = (
            TensorProto.FLOAT16 if np.dtype(dtype) == np.dtype(np.float16) else TensorProto.FLOAT
        )
        nodes = [
            helper.make_node("Pow", ["X", "three"], ["power"]),
            helper.make_node("Mul", ["power", "cubic_scale"], ["cubic"]),
            helper.make_node("Add", ["X", "cubic"], ["inner"]),
            helper.make_node("Mul", ["inner", "tanh_scale"], ["scaled"]),
            helper.make_node("Tanh", ["scaled"], ["tanh"]),
            helper.make_node("Add", ["tanh", "one"], ["tanh_one"]),
            helper.make_node("Mul", ["X", "half"], ["x_half"]),
            helper.make_node("Mul", ["x_half", "tanh_one"], ["Y"]),
        ]
        outputs = [("Y", data_type, [2, 3])]
        if expose_power:
            outputs.append(("power", data_type, [2, 3]))
        return self._make_model(
            nodes,
            [("X", data_type, [2, 3])],
            outputs,
            [
                self._initializer("three", [3], dtype),
                self._initializer("cubic_scale", [cubic_scale], dtype),
                self._initializer("tanh_scale", [0.7978515625], dtype),
                self._initializer("one", [1], dtype),
                self._initializer("half", [0.5], dtype),
            ],
            opset=opset,
        )

    def test_gelu_variants(self):
        for dtype, tolerance in ((np.float32, 1e-5), (np.float16, 2e-3)):
            with self.subTest(dtype=dtype):
                model = self._make_gelu_decomposition(dtype=dtype)
                optimized = self._optimize_and_check(
                    model,
                    {"X": self._range(2, 3, bias=-0.5, dtype=dtype)},
                    ["Gelu"],
                    ["Gelu"],
                    required_pattern="Gelu",
                    initializer_count=5,
                    atol=tolerance,
                    rtol=tolerance,
                )
                approximate = next(
                    attribute
                    for attribute in optimized.graph.node[0].attribute
                    if attribute.name == "approximate"
                )
                self.assertEqual(b"tanh", approximate.s)

    def test_gelu_no_match_and_opset(self):
        variants = [
            self._make_gelu_decomposition(dtype=np.float32, cubic_scale=0.05),
            self._make_gelu_decomposition(dtype=np.float32, opset=19),
            self._make_gelu_decomposition(dtype=np.float32, expose_power=True),
        ]
        for index, model in enumerate(variants):
            with self.subTest(index=index):
                self._assert_no_match(
                    model, {"X": self._range(2, 3, bias=-0.5)}, "Gelu", atol=1e-5
                )

    def _make_leaky_relu_decomposition(
        self, *, dtype, slope: float, threshold: float = 0.0, opset: int = 18
    ):
        data_type = (
            TensorProto.FLOAT16 if np.dtype(dtype) == np.dtype(np.float16) else TensorProto.FLOAT
        )
        return self._make_model(
            [
                helper.make_node("Greater", ["X", "threshold"], ["positive"]),
                helper.make_node("Mul", ["X", "slope"], ["negative"]),
                helper.make_node("Where", ["positive", "X", "negative"], ["Y"]),
            ],
            [("X", data_type, [3, 3])],
            [("Y", data_type, [3, 3])],
            [
                self._initializer("threshold", [threshold], dtype),
                self._initializer("slope", [slope], dtype),
            ],
            opset=opset,
        )

    def test_leaky_relu_variants(self):
        for dtype, slope, tolerance in ((np.float32, 0.76, 1e-6), (np.float16, 0.75, 1e-3)):
            with self.subTest(dtype=dtype, slope=slope):
                model = self._make_leaky_relu_decomposition(dtype=dtype, slope=slope)
                optimized = self._optimize_and_check(
                    model,
                    {"X": self._range(3, 3, bias=-0.5, dtype=dtype)},
                    ["LeakyRelu"],
                    ["LeakyRelu"],
                    required_pattern="LeakyRelu",
                    initializer_count=2,
                    atol=tolerance,
                    rtol=tolerance,
                )
                alpha = next(
                    attribute
                    for attribute in optimized.graph.node[0].attribute
                    if attribute.name == "alpha"
                )
                self.assertAlmostEqual(slope, alpha.f, places=3)

        model = self._make_model(
            [
                helper.make_node("Greater", ["X", "threshold"], ["positive"]),
                helper.make_node("Mul", ["X", "slope"], ["negative"]),
                helper.make_node("Where", ["positive", "X", "negative"], ["X1"]),
                helper.make_node("Greater", ["X1", "threshold2"], ["positive2"]),
                helper.make_node("Mul", ["X1", "slope2"], ["negative2"]),
                helper.make_node("Where", ["positive2", "X1", "negative2"], ["Y"]),
            ],
            [("X", TensorProto.FLOAT, [3, 3])],
            [("Y", TensorProto.FLOAT, [3, 3])],
            [
                self._initializer("threshold", [0], np.float32),
                self._initializer("slope", [0.76], np.float32),
                self._initializer("threshold2", [0], np.float32),
                self._initializer("slope2", [-0.33], np.float32),
            ],
        )
        self._optimize_and_check(
            model,
            {"X": self._range(3, 3, bias=-0.5)},
            ["LeakyRelu"],
            ["LeakyRelu", "LeakyRelu"],
            required_pattern="LeakyRelu",
            initializer_count=3,
        )

    def test_leaky_relu_no_match_and_opset(self):
        nonzero = self._make_leaky_relu_decomposition(dtype=np.float32, slope=0.75, threshold=1.0)
        self._assert_no_match(nonzero, {"X": self._range(3, 3, bias=-0.5)}, "LeakyRelu")

        wrong_branch = self._make_leaky_relu_decomposition(dtype=np.float32, slope=0.75)
        wrong_branch.graph.node[-1].input.clear()
        wrong_branch.graph.node[-1].input.extend(["positive", "negative", "negative"])
        self._assert_no_match(wrong_branch, {"X": self._range(3, 3, bias=-0.5)}, "LeakyRelu")

        old_opset = self._make_leaky_relu_decomposition(dtype=np.float32, slope=0.75, opset=5)
        self._assert_no_match(
            old_opset,
            {"X": self._range(3, 3, bias=-0.5)},
            "LeakyRelu",
            check_model=False,
            evaluate=False,
        )

    def test_max_relu_types_and_commutation(self):
        variants = [
            (np.float32, TensorProto.FLOAT, 13),
            (np.float16, TensorProto.FLOAT16, 18),
            (np.int16, TensorProto.INT16, 18),
            (np.int32, TensorProto.INT32, 18),
        ]
        for dtype, data_type, opset in variants:
            for commuted in (False, True):
                with self.subTest(dtype=dtype, opset=opset, commuted=commuted):
                    inputs = ["zero", "X"] if commuted else ["X", "zero"]
                    model = self._make_model(
                        [helper.make_node("Max", inputs, ["Y"])],
                        [("X", data_type, [3, 3])],
                        [("Y", data_type, [3, 3])],
                        [self._initializer("zero", [0], dtype)],
                        opset=opset,
                    )
                    self._optimize_and_check(
                        model,
                        {"X": self._range(3, 3, bias=-0.5, dtype=dtype)},
                        ["MaxRelu"],
                        ["Relu"],
                        required_pattern="MaxRelu",
                        initializer_count=1,
                    )

    def test_max_relu_no_match_and_opset(self):
        one = self._make_model(
            [helper.make_node("Max", ["X", "one"], ["Y"])],
            [("X", TensorProto.FLOAT, [3, 3])],
            [("Y", TensorProto.FLOAT, [3, 3])],
            [self._initializer("one", [1], np.float32)],
        )
        self._assert_no_match(one, {"X": self._range(3, 3, bias=-0.5)}, "MaxRelu")

        both_zero = self._make_model(
            [helper.make_node("Max", ["zero1", "zero2"], ["Y"])],
            [],
            [("Y", TensorProto.FLOAT, [1])],
            [
                self._initializer("zero1", [0], np.float32),
                self._initializer("zero2", [0], np.float32),
            ],
        )
        self._assert_no_match(both_zero, {}, "MaxRelu")

        unsupported_type = self._make_model(
            [helper.make_node("Max", ["X", "zero"], ["Y"])],
            [("X", TensorProto.INT64, [3, 3])],
            [("Y", TensorProto.INT64, [3, 3])],
            [self._initializer("zero", [0], np.int64)],
        )
        self._assert_no_match(
            unsupported_type, {"X": self._range(3, 3, bias=-2, dtype=np.int64)}, "MaxRelu"
        )

        old_integer_opset = self._make_model(
            [helper.make_node("Max", ["X", "zero"], ["Y"])],
            [("X", TensorProto.INT32, [3, 3])],
            [("Y", TensorProto.INT32, [3, 3])],
            [self._initializer("zero", [0], np.int32)],
            opset=13,
        )
        self._assert_no_match(
            old_integer_opset, {"X": self._range(3, 3, bias=-2, dtype=np.int32)}, "MaxRelu"
        )

    def _make_softmax_cross_entropy_loss(
        self,
        *,
        opset: int = 18,
        score_type: int = TensorProto.FLOAT16,
        score_dtype=np.float16,
        output_type: int = TensorProto.FLOAT16,
        output_cast_type: int = TensorProto.FLOAT16,
        zero_index: int = 0,
        axis: int = 1,
        squeeze_axis: int = 1,
    ):
        return self._make_model(
            [
                helper.make_node("Equal", ["labels", "ignore"], ["equal"]),
                helper.make_node("Not", ["equal"], ["valid"]),
                helper.make_node("Where", ["valid", "labels", "zero_index"], ["safe_labels"]),
                helper.make_node("Unsqueeze", ["safe_labels", "axis"], ["expanded_labels"]),
                helper.make_node("LogSoftmax", ["scores"], ["log_probabilities"], axis=1),
                helper.make_node(
                    "GatherElements",
                    ["log_probabilities", "expanded_labels"],
                    ["gathered"],
                    axis=1,
                ),
                helper.make_node("Squeeze", ["gathered", "squeeze_axis"], ["flat_gathered"]),
                helper.make_node("Neg", ["flat_gathered"], ["losses"]),
                helper.make_node("Where", ["valid", "losses", "zero_loss"], ["masked_losses"]),
                helper.make_node("Cast", ["masked_losses"], ["losses32"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["valid"], ["valid32"], to=TensorProto.FLOAT),
                helper.make_node(
                    "ReduceSum", ["losses32"], ["loss_sum"], keepdims=0, noop_with_empty_axes=0
                ),
                helper.make_node(
                    "ReduceSum", ["valid32"], ["valid_sum"], keepdims=0, noop_with_empty_axes=0
                ),
                helper.make_node("Cast", ["loss_sum"], ["loss_out"], to=output_cast_type),
                helper.make_node("Cast", ["valid_sum"], ["valid_out"], to=output_cast_type),
                helper.make_node("Div", ["loss_out", "valid_out"], ["Y"]),
            ],
            [("scores", score_type, [3, 4]), ("labels", TensorProto.INT64, [3])],
            [("Y", output_type, [])],
            [
                self._initializer("ignore", [-100], np.int64),
                self._initializer("axis", [axis], np.int64),
                self._initializer("squeeze_axis", [squeeze_axis], np.int64),
                self._initializer("zero_index", [zero_index], np.int64),
                self._initializer("zero_loss", [0], score_dtype),
            ],
            opset=opset,
        )

    def test_softmax_cross_entropy_loss_cast(self):
        model = self._make_softmax_cross_entropy_loss()
        feeds = {
            "scores": np.array(
                [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.2, 0.7, 0.05, 0.05]],
                dtype=np.float16,
            ),
            "labels": np.array([3, -100, 1], dtype=np.int64),
        }
        optimized = self._optimize_and_check(
            model,
            feeds,
            ["SoftmaxCrossEntropyLossCast"],
            ["SoftmaxCrossEntropyLoss"],
            required_pattern="SoftmaxCrossEntropyLossCast",
            initializer_count=4,
            atol=2e-3,
            rtol=2e-3,
        )
        attributes = {
            attribute.name: attribute for attribute in optimized.graph.node[0].attribute
        }
        self.assertEqual(-100, attributes["ignore_index"].i)
        self.assertEqual(b"mean", attributes["reduction"].s)

    def test_softmax_cross_entropy_loss_cast_no_match_and_opset(self):
        variants = [
            self._make_softmax_cross_entropy_loss(opset=13),
            self._make_softmax_cross_entropy_loss(
                score_type=TensorProto.FLOAT, score_dtype=np.float32
            ),
            self._make_softmax_cross_entropy_loss(
                output_type=TensorProto.FLOAT, output_cast_type=TensorProto.FLOAT
            ),
            self._make_softmax_cross_entropy_loss(zero_index=1),
        ]
        feeds = {
            "scores": self._range(3, 4, dtype=np.float16),
            "labels": np.array([3, -100, 1], dtype=np.int64),
        }
        for index, model in enumerate(variants):
            with self.subTest(index=index):
                model_feeds = dict(feeds)
                if model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT:
                    model_feeds["scores"] = model_feeds["scores"].astype(np.float32)
                self._assert_no_match(
                    model, model_feeds, "SoftmaxCrossEntropyLossCast", atol=2e-3, rtol=2e-3
                )

        wrong_axis = self._make_softmax_cross_entropy_loss(axis=0, squeeze_axis=0)
        self._assert_no_match(wrong_axis, feeds, "SoftmaxCrossEntropyLossCast", evaluate=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
