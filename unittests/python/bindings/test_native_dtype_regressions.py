# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Exercises native dtype and initializer boundaries without reference ONNX."""

import unittest

import numpy

from onnx_light.ext_test_case import import_or_skip
from onnx_light.onnx import TensorProto, helper, numpy_helper

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")


class TestNativeDtypeRegressions(unittest.TestCase):
    def run_node(self, op_type, feeds, opset=18, **attributes):
        """Runs a node using only compiled native kernels."""
        node = helper.make_node(op_type, list(feeds), ["result"], **attributes)
        function = helper.make_function(
            "",
            "native_regression",
            list(feeds),
            ["result"],
            [node],
            opset_imports=[helper.make_opsetid("", opset)],
        )
        return ReferenceEvaluator(function).run(None, feeds)[0]

    def test_string_initializer_cache_and_callback(self):
        values = numpy.array(["\u00e9t\u00e9", "", "\u6771\u4eac"], dtype=object)
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Equal", ["X", "labels"], ["Y"])],
                "strings",
                [helper.make_tensor_value_info("X", TensorProto.STRING, [3])],
                [helper.make_tensor_value_info("Y", TensorProto.BOOL, [3])],
                [numpy_helper.from_array(values, "labels")],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        serialized = model.SerializeToString()
        for custom in (False, True):
            evaluator = ReferenceEvaluator(model)
            observed = []
            if custom:

                def equal(node, x, labels):
                    observed.append(labels.copy())
                    return numpy.equal(x, labels)

                evaluator.register_custom_kernel("", "Equal", equal)
            for inputs in (values, values[::-1].copy(), values):
                feeds = {"X": inputs}
                numpy.testing.assert_array_equal(
                    evaluator.run(None, feeds)[0], numpy.equal(inputs, values)
                )
                self.assertEqual(set(feeds), {"X"})
            if custom:
                for labels in observed:
                    numpy.testing.assert_array_equal(labels, values)
            numpy.testing.assert_array_equal(
                evaluator.run(None, {"X": values[::-1], "labels": values[::-1]})[0],
                numpy.ones(3, dtype=numpy.bool_),
            )
            numpy.testing.assert_array_equal(
                evaluator.run(None, {"X": values})[0], numpy.ones(3, dtype=numpy.bool_)
            )
        self.assertEqual(model.SerializeToString(), serialized)

    def test_less_double_precision_and_broadcast(self):
        x = numpy.array([[1], [1 + 1e-12]], dtype=numpy.float64)
        y = numpy.array([1 + 1e-12, 1], dtype=numpy.float64)
        numpy.testing.assert_array_equal(self.run_node("Less", {"X": x, "Y": y}), x < y)

    def test_compress_strings(self):
        data = numpy.array([["\u00e9t\u00e9", ""], ["\u6771\u4eac", "D"]], dtype=object)
        for axis, mask in (
            (None, [True, False, True, False]),
            (None, []),
            (0, [False, False]),
            (0, [True]),
            (1, [False, True]),
            (-1, [True, False, True]),
        ):
            with self.subTest(axis=axis, mask=mask):
                condition = numpy.array(mask, dtype=numpy.bool_)
                attributes = {} if axis is None else {"axis": axis}
                actual = self.run_node("Compress", {"X": data, "C": condition}, **attributes)
                limit = data.size if axis is None else data.shape[axis]
                numpy.testing.assert_array_equal(
                    actual, numpy.compress(condition[:limit], data, axis=axis)
                )
                self.assertEqual(actual.dtype, numpy.dtype(object))

    def test_reduction_axes_versions_precision(self):
        for dtype in (numpy.float16, numpy.float32, numpy.float64):
            data = numpy.array([[1, 1 + 1e-12], [2, 2 + 1e-12]], dtype=dtype)
            for op_type, operation in (
                ("ReduceMin", numpy.min),
                ("ReduceMax", numpy.max),
                ("ReduceMean", numpy.mean),
            ):
                for opset in (13, 18):
                    for keepdims in (0, 1):
                        with self.subTest(
                            dtype=dtype, op_type=op_type, opset=opset, keepdims=keepdims
                        ):
                            feeds = {"X": data}
                            attributes = {"keepdims": keepdims}
                            if opset >= 18:
                                feeds["axes"] = numpy.array([-1], dtype=numpy.int64)
                            else:
                                attributes["axes"] = [-1]
                            actual = self.run_node(op_type, feeds, opset, **attributes)
                            expected = operation(data, axis=-1, keepdims=bool(keepdims))
                            self.assertEqual(actual.dtype, expected.dtype)
                            numpy.testing.assert_array_equal(actual, expected)

    def test_reduction_empty_axes_nan_and_invalid_axes(self):
        for op_type, operation in (
            ("ReduceMin", numpy.min),
            ("ReduceMax", numpy.max),
            ("ReduceMean", numpy.mean),
        ):
            for noop in (0, 1):
                data = numpy.arange(6, dtype=numpy.float64).reshape(2, 3)
                actual = self.run_node(
                    op_type,
                    {"X": data, "axes": numpy.array([], dtype=numpy.int64)},
                    keepdims=0,
                    noop_with_empty_axes=noop,
                )
                numpy.testing.assert_array_equal(actual, data if noop else operation(data))
            data = numpy.array([[numpy.nan, 1], [2, numpy.nan]], dtype=numpy.float64)
            actual = self.run_node(
                op_type, {"X": data, "axes": numpy.array([1], dtype=numpy.int64)}, keepdims=0
            )
            self.assertTrue(numpy.isnan(actual).all())
            with self.assertRaisesRegex(ValueError, "axis.*range"):
                self.run_node(op_type, {"X": data, "axes": numpy.array([2], dtype=numpy.int64)})

    def test_reduction_empty_data(self):
        for dtype in (numpy.float16, numpy.float32, numpy.float64, numpy.int64, numpy.bool_):
            data = numpy.empty((0, 3), dtype=dtype)
            for op_type in ("ReduceMin", "ReduceMax"):
                minimum = op_type == "ReduceMin"
                if numpy.issubdtype(dtype, numpy.floating):
                    neutral = numpy.inf if minimum else -numpy.inf
                elif dtype == numpy.bool_:
                    neutral = minimum
                else:
                    neutral = numpy.iinfo(dtype).max if minimum else numpy.iinfo(dtype).min
                with self.subTest(dtype=dtype, op_type=op_type):
                    actual = self.run_node(
                        op_type,
                        {"X": data, "axes": numpy.array([0], dtype=numpy.int64)},
                        opset=20 if dtype == numpy.bool_ else 18,
                        keepdims=0,
                    )
                    numpy.testing.assert_array_equal(actual, numpy.full(3, neutral, dtype))
            if numpy.issubdtype(dtype, numpy.floating):
                actual = self.run_node(
                    "ReduceMean",
                    {"X": data, "axes": numpy.array([0], dtype=numpy.int64)},
                    keepdims=0,
                )
                self.assertEqual(actual.shape, (3,))
                self.assertTrue(numpy.isnan(actual).all())

    def test_logsoftmax_versions_dtypes_and_stability(self):
        for dtype in (numpy.float16, numpy.float32, numpy.float64):
            data = (numpy.arange(24).reshape(2, 3, 4) * 10 + 10000).astype(dtype)
            for opset in (11, 13, 18):
                for axis in (None, 0, 1, -1):
                    with self.subTest(dtype=dtype, opset=opset, axis=axis):
                        attributes = {} if axis is None else {"axis": axis}
                        actual = self.run_node("LogSoftmax", {"X": data}, opset, **attributes)
                        selected_axis = (1 if opset < 13 else -1) if axis is None else axis
                        values = data.astype(numpy.float64)
                        if opset < 13:
                            selected_axis %= data.ndim
                            values = values.reshape(
                                int(numpy.prod(data.shape[:selected_axis])), -1
                            )
                            selected_axis = 1
                        values -= numpy.max(values, axis=selected_axis, keepdims=True)
                        expected = values - numpy.log(
                            numpy.sum(numpy.exp(values), axis=selected_axis, keepdims=True)
                        )
                        self.assertEqual(actual.dtype, data.dtype)
                        numpy.testing.assert_allclose(
                            actual,
                            expected.reshape(data.shape).astype(dtype),
                            atol=1e-6,
                            rtol=1e-6,
                        )
            for opset in (11, 18):
                for axis in (0, 1):
                    data = numpy.empty((2, 0), dtype=dtype)
                    actual = self.run_node("LogSoftmax", {"X": data}, opset, axis=axis)
                    self.assertEqual(actual.shape, data.shape)
                with self.assertRaisesRegex(ValueError, "axis.*range"):
                    self.run_node("LogSoftmax", {"X": data}, opset, axis=2)

    def test_where_broadcast_signed_zero_and_invalid_types(self):
        condition = numpy.array([[True, False, True], [False, True, False]])
        for dtype in (numpy.float16, numpy.float32, numpy.float64):
            x = numpy.array([-0.0, 1 + 1e-12, 10000], dtype=dtype)
            y = numpy.array([[0.0], [-2.0]], dtype=dtype)
            actual = self.run_node("Where", {"C": condition, "X": x, "Y": y})
            expected = numpy.where(condition, x, y)
            self.assertEqual(actual.dtype, expected.dtype)
            numpy.testing.assert_array_equal(actual, expected)
            numpy.testing.assert_array_equal(numpy.signbit(actual), numpy.signbit(expected))
            empty = numpy.empty((0, 3), dtype=dtype)
            actual = self.run_node(
                "Where", {"C": numpy.ones((0, 1), dtype=numpy.bool_), "X": empty, "Y": x}
            )
            self.assertEqual(actual.shape, empty.shape)
        with self.assertRaisesRegex(ValueError, "BOOL condition"):
            self.run_node("Where", {"C": numpy.ones(3), "X": x, "Y": x})
        with self.assertRaisesRegex(ValueError, "same dtype"):
            self.run_node("Where", {"C": condition, "X": x, "Y": x.astype(numpy.float32)})


if __name__ == "__main__":
    unittest.main()
