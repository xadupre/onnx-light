# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Exercises native dtype and initializer boundaries without reference ONNX."""

import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.numpy_helper as onh
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
from onnx_light.onnx_core.graph_builder import GraphBuilder

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


class TestNativeDtypeRegressions(ExtTestCase):
    def run_node(self, op_type, feeds, opset=18, **attributes):
        """Runs a node using only compiled native kernels."""
        node = oh.make_node(op_type, list(feeds), ["result"], **attributes)
        function = oh.make_function(
            "",
            "native_regression",
            list(feeds),
            ["result"],
            [node],
            opset_imports=[oh.make_opsetid("", opset)],
        )
        return ReferenceEvaluator(function).run(None, feeds)[0]

    def test_cast_complex_shapes_and_precision(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for shape in ((), (3,), (0,), (2, 0, 3)):
                data = np.full(shape, 1.25, dtype=dtype)
                if np.issubdtype(dtype, np.complexfloating):
                    data += 2.5j
                for to, output_dtype in (
                    (TensorProto.COMPLEX64, np.complex64),
                    (TensorProto.COMPLEX128, np.complex128),
                    (TensorProto.FLOAT, np.float32),
                    (TensorProto.DOUBLE, np.float64),
                ):
                    with self.subTest(dtype=dtype, shape=shape, to=to):
                        actual = self.run_node("Cast", {"X": data}, opset=13, to=to)
                        expected = (
                            data.real.astype(output_dtype)
                            if np.issubdtype(output_dtype, np.floating)
                            else data.astype(output_dtype)
                        )
                        self.assertEqual(actual.dtype, np.dtype(output_dtype))
                        self.assertEqual(actual.shape, shape)
                        np.testing.assert_array_equal(actual, expected)

    def test_cast_complex_graph_builder(self):
        data = np.array([-1.5, 0, 2.25], dtype=np.float32)
        for opset in (6, 9, 13, 19, 21, 23, 24, 25, 28):
            with self.subTest(opset=opset):
                builder = GraphBuilder("complex_cast")
                builder.set_opset_version("", opset)
                x = builder.inp("x", TensorProto.FLOAT, [3])
                y = builder.op.Cast(x, to=TensorProto.COMPLEX64)
                z = builder.op.Cast(y, to=TensorProto.COMPLEX128)
                builder.out(z)
                model = builder.to_onnx()
                self.assertEqual(
                    model.graph.output[0].type.tensor_type.elem_type, TensorProto.COMPLEX128
                )
                actual = ReferenceEvaluator(model).run(None, {"x": data})[0]
                self.assertEqual(actual.dtype, np.dtype(np.complex128))
                np.testing.assert_array_equal(actual, data.astype(np.complex128))

    def test_cast_complex_rejects_unsupported_pairs(self):
        for to in (TensorProto.STRING, TensorProto.FLOAT8E5M2, TensorProto.INT4):
            with self.subTest(to=to), self.assertRaises(ValueError):
                self.run_node("Cast", {"X": np.array([1 + 2j], dtype=np.complex64)}, to=to)
        with self.assertRaises(ValueError):
            self.run_node("Cast", {"X": np.array(["1"], dtype=object)}, to=TensorProto.COMPLEX64)

    def test_string_initializer_cache_and_callback(self):
        values = np.array(["\u00e9t\u00e9", "", "\u6771\u4eac"], dtype=object)
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Equal", ["X", "labels"], ["Y"])],
                "strings",
                [oh.make_tensor_value_info("X", TensorProto.STRING, [3])],
                [oh.make_tensor_value_info("Y", TensorProto.BOOL, [3])],
                [onh.from_array(values, "labels")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        serialized = model.SerializeToString()
        for custom in (False, True):
            evaluator = ReferenceEvaluator(model)
            observed = []
            if custom:

                def equal(node, x, labels):
                    observed.append(labels.copy())
                    return np.equal(x, labels)

                evaluator.register_custom_kernel("", "Equal", equal)
            for inputs in (values, values[::-1].copy(), values):
                feeds = {"X": inputs}
                np.testing.assert_array_equal(
                    evaluator.run(None, feeds)[0], np.equal(inputs, values)
                )
                self.assertEqual(set(feeds), {"X"})
            if custom:
                for labels in observed:
                    np.testing.assert_array_equal(labels, values)
            np.testing.assert_array_equal(
                evaluator.run(None, {"X": values[::-1], "labels": values[::-1]})[0],
                np.ones(3, dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                evaluator.run(None, {"X": values})[0], np.ones(3, dtype=np.bool_)
            )
        self.assertEqual(model.SerializeToString(), serialized)

    def test_less_double_precision_and_broadcast(self):
        x = np.array([[1], [1 + 1e-12]], dtype=np.float64)
        y = np.array([1 + 1e-12, 1], dtype=np.float64)
        np.testing.assert_array_equal(self.run_node("Less", {"X": x, "Y": y}), x < y)

    def test_compress_strings(self):
        data = np.array([["\u00e9t\u00e9", ""], ["\u6771\u4eac", "D"]], dtype=object)
        for axis, mask in (
            (None, [True, False, True, False]),
            (None, []),
            (0, [False, False]),
            (0, [True]),
            (1, [False, True]),
            (-1, [True, False, True]),
        ):
            with self.subTest(axis=axis, mask=mask):
                condition = np.array(mask, dtype=np.bool_)
                attributes = {} if axis is None else {"axis": axis}
                actual = self.run_node("Compress", {"X": data, "C": condition}, **attributes)
                limit = data.size if axis is None else data.shape[axis]
                np.testing.assert_array_equal(
                    actual, np.compress(condition[:limit], data, axis=axis)
                )
                self.assertEqual(actual.dtype, np.dtype(object))

    def test_reduction_axes_versions_precision(self):
        for dtype in (np.float16, np.float32, np.float64):
            data = np.array([[1, 1 + 1e-12], [2, 2 + 1e-12]], dtype=dtype)
            for op_type, operation in (
                ("ReduceMin", np.min),
                ("ReduceMax", np.max),
                ("ReduceMean", np.mean),
            ):
                for opset in (13, 18):
                    for keepdims in (0, 1):
                        with self.subTest(
                            dtype=dtype, op_type=op_type, opset=opset, keepdims=keepdims
                        ):
                            feeds = {"X": data}
                            attributes = {"keepdims": keepdims}
                            if opset >= 18:
                                feeds["axes"] = np.array([-1], dtype=np.int64)
                            else:
                                attributes["axes"] = [-1]
                            actual = self.run_node(op_type, feeds, opset, **attributes)
                            expected = operation(data, axis=-1, keepdims=bool(keepdims))
                            self.assertEqual(actual.dtype, expected.dtype)
                            np.testing.assert_array_equal(actual, expected)

    def test_reduction_empty_axes_nan_and_invalid_axes(self):
        for op_type, operation in (
            ("ReduceMin", np.min),
            ("ReduceMax", np.max),
            ("ReduceMean", np.mean),
        ):
            for noop in (0, 1):
                data = np.arange(6, dtype=np.float64).reshape(2, 3)
                actual = self.run_node(
                    op_type,
                    {"X": data, "axes": np.array([], dtype=np.int64)},
                    keepdims=0,
                    noop_with_empty_axes=noop,
                )
                np.testing.assert_array_equal(actual, data if noop else operation(data))
            data = np.array([[np.nan, 1], [2, np.nan]], dtype=np.float64)
            actual = self.run_node(
                op_type, {"X": data, "axes": np.array([1], dtype=np.int64)}, keepdims=0
            )
            self.assertTrue(np.isnan(actual).all())
            with self.assertRaisesRegex(ValueError, "axis.*range"):
                self.run_node(op_type, {"X": data, "axes": np.array([2], dtype=np.int64)})

    def test_reduction_empty_data(self):
        for dtype in (np.float16, np.float32, np.float64, np.int64, np.bool_):
            data = np.empty((0, 3), dtype=dtype)
            for op_type in ("ReduceMin", "ReduceMax"):
                minimum = op_type == "ReduceMin"
                if np.issubdtype(dtype, np.floating):
                    neutral = np.inf if minimum else -np.inf
                elif dtype == np.bool_:
                    neutral = minimum
                else:
                    neutral = np.iinfo(dtype).max if minimum else np.iinfo(dtype).min
                with self.subTest(dtype=dtype, op_type=op_type):
                    actual = self.run_node(
                        op_type,
                        {"X": data, "axes": np.array([0], dtype=np.int64)},
                        opset=20 if dtype == np.bool_ else 18,
                        keepdims=0,
                    )
                    np.testing.assert_array_equal(actual, np.full(3, neutral, dtype))
            if np.issubdtype(dtype, np.floating):
                actual = self.run_node(
                    "ReduceMean", {"X": data, "axes": np.array([0], dtype=np.int64)}, keepdims=0
                )
                self.assertEqual(actual.shape, (3,))
                self.assertTrue(np.isnan(actual).all())

    def test_logsoftmax_versions_dtypes_and_stability(self):
        for dtype in (np.float16, np.float32, np.float64):
            data = (np.arange(24).reshape(2, 3, 4) * 10 + 10000).astype(dtype)
            for opset in (11, 13, 18):
                for axis in (None, 0, 1, -1):
                    with self.subTest(dtype=dtype, opset=opset, axis=axis):
                        attributes = {} if axis is None else {"axis": axis}
                        actual = self.run_node("LogSoftmax", {"X": data}, opset, **attributes)
                        selected_axis = (1 if opset < 13 else -1) if axis is None else axis
                        values = data.astype(np.float64)
                        if opset < 13:
                            selected_axis %= data.ndim
                            values = values.reshape(int(np.prod(data.shape[:selected_axis])), -1)
                            selected_axis = 1
                        values -= np.max(values, axis=selected_axis, keepdims=True)
                        expected = values - np.log(
                            np.sum(np.exp(values), axis=selected_axis, keepdims=True)
                        )
                        self.assertEqual(actual.dtype, data.dtype)
                        np.testing.assert_allclose(
                            actual,
                            expected.reshape(data.shape).astype(dtype),
                            atol=1e-6,
                            rtol=1e-6,
                        )
            for opset in (11, 18):
                for axis in (0, 1):
                    data = np.empty((2, 0), dtype=dtype)
                    actual = self.run_node("LogSoftmax", {"X": data}, opset, axis=axis)
                    self.assertEqual(actual.shape, data.shape)
                with self.assertRaisesRegex(ValueError, "axis.*range"):
                    self.run_node("LogSoftmax", {"X": data}, opset, axis=2)

    def test_where_broadcast_signed_zero_and_invalid_types(self):
        condition = np.array([[True, False, True], [False, True, False]])
        for dtype in (np.float16, np.float32, np.float64):
            x = np.array([-0.0, 1 + 1e-12, 10000], dtype=dtype)
            y = np.array([[0.0], [-2.0]], dtype=dtype)
            actual = self.run_node("Where", {"C": condition, "X": x, "Y": y})
            expected = np.where(condition, x, y)
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
            empty = np.empty((0, 3), dtype=dtype)
            actual = self.run_node(
                "Where", {"C": np.ones((0, 1), dtype=np.bool_), "X": empty, "Y": x}
            )
            self.assertEqual(actual.shape, empty.shape)
        with self.assertRaisesRegex(ValueError, "BOOL condition"):
            self.run_node("Where", {"C": np.ones(3), "X": x, "Y": x})
        with self.assertRaisesRegex(ValueError, "same dtype"):
            self.run_node("Where", {"C": condition, "X": x, "Y": x.astype(np.float32)})


if __name__ == "__main__":
    unittest.main()
