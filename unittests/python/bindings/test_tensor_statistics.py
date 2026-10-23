# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest

import ml_dtypes
import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
tensor_statistics = import_or_skip("onnx_light.onnx_core.tensor_statistics")
TensorStatistics = tensor_statistics.TensorStatistics
compute_tensor_statistics = tensor_statistics.compute_tensor_statistics


class TestTensorStatistics(ExtTestCase):
    def test_runtime_value_is_canonical_shared_quantized_value_alias(self):
        self.assertEqual(runtime.RuntimeValue.__name__, "RuntimeValue")
        self.assertIs(runtime.SharedQuantizedValue, runtime.RuntimeValue)

    def test_numpy_tensor_and_proto(self):
        array = numpy.arange(24, dtype=numpy.float32).reshape(4, 6)
        proto = onh.from_array(array)
        tensor = runtime.tensor_from_numpy(
            "X", onnx.TensorProto.FLOAT, list(array.shape), array.view(numpy.uint8).ravel()
        )
        for value in (array, proto, tensor):
            with self.subTest(value_type=type(value)):
                result = compute_tensor_statistics(
                    value, quantiles=[0, 0.25, 0.75, 1], median=True, rectangle_shape=[2, 3]
                )
                self.assertIsInstance(result, TensorStatistics)
                self.assertEqual(result.dtype, onnx.TensorProto.FLOAT)
                self.assertEqual(result.shape, [4, 6])
                self.assertEqual(result.size_bytes, array.nbytes)
                self.assertEqual(result.count, array.size)
                self.assertEqual(result.min, 0)
                self.assertEqual(result.max, 23)
                self.assertEqual(result.mean, 11.5)
                self.assertIsNone(result.distinct_count)
                self.assertEqual(result.median, 11.5)
                numpy.testing.assert_allclose(
                    result.quantiles, numpy.quantile(array, [0, 0.25, 0.75, 1])
                )
                expected_map = array.reshape(2, 2, 3, 2).mean(axis=(1, 3))
                self.assertEqual(result.rectangle_shape, [2, 3])
                numpy.testing.assert_allclose(
                    numpy.array(result.rectangle_map).reshape(2, 3), expected_map
                )
                counts, edges = numpy.histogram(array, bins=10)
                numpy.testing.assert_array_equal(result.distribution, counts)
                numpy.testing.assert_allclose(result.bin_edges, edges)

    def test_runtime_value_wrapping_tensor(self):
        array = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
        tensor = runtime.tensor_from_numpy(
            "X", onnx.TensorProto.FLOAT, list(array.shape), array.view(numpy.uint8).ravel()
        )
        value = runtime.RuntimeValue(tensor)
        result = compute_tensor_statistics(value)
        self.assertEqual(result.shape, [2, 3])
        self.assertEqual(result.min, 0)
        self.assertEqual(result.max, 5)
        self.assertEqual(result.mean, 2.5)

    def test_histogram_thresholds(self):
        for count, bins in ((0, 0), (9, 0), (10, 10), (500, 10), (501, 50)):
            with self.subTest(count=count):
                result = compute_tensor_statistics(numpy.arange(count, dtype=numpy.int32))
                self.assertEqual(len(result.distribution), bins)
                self.assertEqual(sum(result.distribution), count if bins else 0)
                self.assertEqual(result.distinct_count, count)
                self.assertIsNone(result.median)
                self.assertEqual(result.quantiles, [])
                self.assertEqual(result.rectangle_map, [])
        result = compute_tensor_statistics(numpy.ones(12), bins=3)
        self.assertEqual(len(result.distribution), 3)
        self.assertEqual(sum(result.distribution), 12)
        self.assertEqual(result.min, result.max)
        numpy.testing.assert_array_equal(
            result.distribution, numpy.histogram(numpy.ones(12), bins=3)[0]
        )

    def test_numeric_types_and_exact_integer_distinct(self):
        types = (
            numpy.bool_,
            numpy.int8,
            numpy.uint8,
            numpy.int16,
            numpy.uint16,
            numpy.int32,
            numpy.uint32,
            numpy.int64,
            numpy.uint64,
            numpy.float16,
            numpy.float32,
            numpy.float64,
            ml_dtypes.bfloat16,
            ml_dtypes.float8_e4m3fn,
            ml_dtypes.int4,
            ml_dtypes.uint4,
        )
        for dtype in types:
            with self.subTest(dtype=dtype):
                values = numpy.array([0, 1, 1, 2], dtype=dtype)
                result = compute_tensor_statistics(onh.from_array(values))
                expected = values.astype(numpy.float64)
                self.assertEqual(result.min, expected.min())
                self.assertEqual(result.max, expected.max())
                self.assertEqual(result.mean, expected.mean())
                direct = compute_tensor_statistics(values)
                self.assertEqual(direct.min, result.min)
                self.assertEqual(direct.max, result.max)
                self.assertEqual(direct.mean, result.mean)
                self.assertEqual(direct.size_bytes, result.size_bytes)
                if dtype in (numpy.bool_,):
                    self.assertEqual(result.distinct_count, 2)
                elif dtype not in (
                    numpy.float16,
                    numpy.float32,
                    numpy.float64,
                    ml_dtypes.bfloat16,
                    ml_dtypes.float8_e4m3fn,
                ):
                    self.assertEqual(result.distinct_count, 3)
        for dtype, base in ((numpy.int64, 2**62), (numpy.uint64, 2**64 - 4)):
            values = numpy.array([base, base + 1, base + 1, base + 2], dtype=dtype)
            self.assertEqual(compute_tensor_statistics(values).distinct_count, 3)

    def test_quantized_values(self):
        values = numpy.array([-8, -4, 0, 1, 2, 7] * 2, dtype=numpy.float32).reshape(3, 4)
        plan = runtime.make_quantization_plan(runtime.QuantizationFormat.INT4, values.size, 4)
        encoded = runtime.quantize_tensor_proto(onh.from_array(values), plan)
        result = compute_tensor_statistics(encoded, median=True, quantiles=[0.25, 0.75])
        reconstructed = onh.to_array(runtime.dequantize_tensor_proto(encoded))
        self.assertEqual(result.dtype, onnx.TensorProto.FLOAT)
        self.assertEqual(result.shape, [3, 4])
        self.assertEqual(result.count, values.size)
        self.assertEqual(result.size_bytes, len(encoded.raw_data))
        self.assertEqual(result.min, reconstructed.min())
        self.assertEqual(result.max, reconstructed.max())
        self.assertAlmostEqual(result.mean, reconstructed.astype(numpy.float64).mean())
        self.assertEqual(result.distinct_count, len(numpy.unique(reconstructed)))
        self.assertEqual(result.median, numpy.median(reconstructed))
        numpy.testing.assert_allclose(
            result.quantiles, numpy.quantile(reconstructed, [0.25, 0.75])
        )
        self.assertEqual(sum(result.distribution), values.size)

    def test_shared_quantization(self):
        from onnx_light.onnx_core.quantization import add_quantization_parameters

        values = numpy.array([-8, -4, 0, 7, -8, -2, 4, 7], dtype=numpy.float32)
        plan = runtime.make_quantization_plan(runtime.QuantizationFormat.INT4, values.size, 4)
        model = oh.make_model(oh.make_graph([], "shared", [], []))
        add_quantization_parameters(
            model,
            "common",
            runtime.make_quantization_type(plan),
            oh.make_tensor_type_proto(onnx.TensorProto.FLOAT, [values.size]),
            scales=numpy.array([1, 2], dtype=numpy.float64),
        )
        tensor = runtime.tensor_from_numpy(
            "X", onnx.TensorProto.FLOAT, [values.size], values.view(numpy.uint8)
        )
        shared = runtime.quantize_tensor_shared(tensor, model, "common")
        expected = compute_tensor_statistics(runtime.dequantize_tensor(shared), median=True)
        for value in (shared, shared.encoded):
            result = compute_tensor_statistics(value, model=model, median=True)
            self.assertEqual(result.min, expected.min)
            self.assertEqual(result.max, expected.max)
            self.assertEqual(result.mean, expected.mean)
            self.assertEqual(result.median, expected.median)
            self.assertEqual(result.size_bytes, len(shared.raw_data))
            result = compute_tensor_statistics(
                value, model=model, bins=1, median=True, max_output_bytes=32
            )
            self.assertEqual(result.median, expected.median)
            for options in (
                {"bins": 1, "median": True, "max_output_bytes": 31},
                {"bins": 2**30},
                {"rectangle_shape": [1, 2**30]},
            ):
                with self.assertRaisesRegex(ValueError, "max_output_bytes"):
                    compute_tensor_statistics(value, model=model, **options)
                with self.assertRaisesRegex(ValueError, "max_output_bytes"):
                    compute_tensor_statistics(value, **options)
        with self.assertRaises(ValueError):
            compute_tensor_statistics(shared.encoded)

    def test_encoded_values_require_logical_tensor_metadata(self):
        values = numpy.arange(12, dtype=numpy.float32)
        plan = runtime.make_quantization_plan(runtime.QuantizationFormat.INT4, values.size, 4)
        encoded = runtime.quantize_tensor_proto(onh.from_array(values), plan)
        encoded.ClearField("logical_type")
        with self.assertRaises(ValueError):
            compute_tensor_statistics(encoded)
        encoded.logical_type.sequence_type.elem_type.tensor_type.elem_type = (
            onnx.TensorProto.FLOAT
        )
        with self.assertRaises(ValueError):
            compute_tensor_statistics(encoded)

    def test_nonfinite_and_result_lifetime(self):
        values = numpy.array([numpy.nan, -numpy.inf, numpy.inf, 0, 1, 2, 3, 4, 5, 6])
        result = compute_tensor_statistics(values, bins=2, quantiles=[0, 0.5, 1], median=True)
        self.assertTrue(numpy.isnan([result.min, result.max, result.mean]).all())
        self.assertTrue(numpy.isnan(result.quantiles).all())
        self.assertTrue(numpy.isnan(result.median))
        self.assertEqual(sum(result.distribution), 7)
        result = compute_tensor_statistics(numpy.arange(10, dtype=numpy.int32), median=True)
        self.assertEqual(result.mean, 4.5)
        self.assertEqual(result.distinct_count, 10)
        self.assertEqual(sum(result.distribution), 10)

    def test_output_allocation_budget_boundaries(self):
        count_bytes = numpy.dtype(numpy.int64).itemsize
        float_bytes = numpy.dtype(numpy.float64).itemsize
        one_bin_bytes = count_bytes + 2 * float_bytes
        # Two counts, three edges, two rectangle cells, one quantile, and the median.
        combined_bytes = 2 * count_bytes + (3 + 2 + 1 + 1) * float_bytes
        values = numpy.arange(10, dtype=numpy.float32)
        proto = onh.from_array(values)
        tensor = runtime.tensor_from_numpy(
            "X", onnx.TensorProto.FLOAT, [10], values.view(numpy.uint8)
        )
        plan = runtime.make_quantization_plan(runtime.QuantizationFormat.INT8, 10)
        encoded = runtime.quantize_tensor_proto(proto, plan)
        for value in (values, proto, tensor, runtime.RuntimeValue(tensor), encoded):
            with self.subTest(value_type=type(value)):
                options = dict(bins=2, rectangle_shape=[1, 2], quantiles=[0.5], median=True)
                result = compute_tensor_statistics(
                    value, **options, max_output_bytes=combined_bytes
                )
                self.assertEqual(len(result.distribution), 2)
                self.assertEqual(len(result.bin_edges), 3)
                self.assertEqual(len(result.rectangle_map), 2)
                self.assertEqual(len(result.quantiles), 1)
                self.assertIsNotNone(result.median)
                with self.assertRaisesRegex(ValueError, "max_output_bytes"):
                    compute_tensor_statistics(
                        value, **options, max_output_bytes=combined_bytes - 1
                    )
        for options, budget in (
            ({}, 50 * count_bytes + 51 * float_bytes),  # Maximum automatic histogram.
            ({"bins": 3}, 3 * count_bytes + 4 * float_bytes),
            ({"bins": 1, "rectangle_shape": [2, 2]}, one_bin_bytes + 4 * float_bytes),
            ({"bins": 1, "quantiles": [0.5]}, one_bin_bytes + float_bytes),
            ({"bins": 1, "median": True}, one_bin_bytes + float_bytes),
        ):
            with self.subTest(options=options):
                compute_tensor_statistics(values, **options, max_output_bytes=budget)
                with self.assertRaisesRegex(ValueError, "max_output_bytes"):
                    compute_tensor_statistics(values, **options, max_output_bytes=budget - 1)
        for budget in (0, one_bin_bytes - 1):
            with self.assertRaisesRegex(ValueError, "max_output_bytes"):
                compute_tensor_statistics(values, bins=1, max_output_bytes=budget)
        for budget in (-1, 2**65):
            with self.assertRaises((TypeError, OverflowError)):
                compute_tensor_statistics(values, max_output_bytes=budget)

        empty = numpy.array([], dtype=numpy.float32)
        default_budget = 64 * 1024 * 1024
        maximum_bins = (default_budget - float_bytes) // (count_bytes + float_bytes)
        maximum_cells = (default_budget - one_bin_bytes) // float_bytes
        compute_tensor_statistics(empty, bins=maximum_bins)
        compute_tensor_statistics(
            empty,
            bins=maximum_bins + 1,
            max_output_bytes=default_budget + count_bytes + float_bytes,
        )
        compute_tensor_statistics(empty, bins=1, rectangle_shape=[1, maximum_cells])
        for options in (
            {"bins": maximum_bins + 1},
            {"bins": 1, "rectangle_shape": [1, maximum_cells + 1]},
        ):
            with self.assertRaisesRegex(ValueError, "max_output_bytes"):
                compute_tensor_statistics(empty, **options)

    def test_output_budget_rejected_before_input_conversion(self):
        values = numpy.arange(10, dtype=numpy.float32)
        tensor = runtime.tensor_from_numpy(
            "X", onnx.TensorProto.FLOAT, [10], values.view(numpy.uint8)
        )
        plan = runtime.make_quantization_plan(runtime.QuantizationFormat.INT8, 10)
        encoded = runtime.quantize_tensor(tensor, plan)
        malformed = onnx.TensorProto(data_type=onnx.TensorProto.FLOAT, dims=[10])
        for value in (
            object(),
            numpy.array([], dtype=numpy.float32),
            values,
            malformed,
            tensor,
            runtime.RuntimeValue(tensor),
            encoded,
            onnx.EncodedValueProto(),
        ):
            for options in (
                {"bins": 2**30},
                {"rectangle_shape": [1, 2**30]},
                {"bins": 2, "max_output_bytes": 39},
                {"bins": 1, "rectangle_shape": [2, 2], "max_output_bytes": 55},
            ):
                with (
                    self.subTest(value_type=type(value), options=options),
                    self.assertRaisesRegex(ValueError, "max_output_bytes"),
                ):
                    compute_tensor_statistics(value, **options)

    def test_scalar_empty_and_invalid_options(self):
        scalar = compute_tensor_statistics(numpy.array(3.0), median=True, rectangle_shape=[1, 2])
        self.assertEqual(scalar.shape, [])
        self.assertEqual(scalar.count, 1)
        self.assertEqual(scalar.median, 3)
        self.assertEqual(scalar.rectangle_map, [3, 3])
        empty = compute_tensor_statistics(numpy.array([], dtype=numpy.float64))
        self.assertTrue(numpy.isnan([empty.min, empty.max, empty.mean]).all())
        strided = numpy.arange(12).reshape(3, 4)[:, ::2]
        self.assertEqual(compute_tensor_statistics(strided).mean, strided.mean())
        for values in (numpy.array([], dtype=numpy.float32), numpy.arange(12)):
            for options in (
                {"bins": -1},
                {"quantiles": [-0.1]},
                {"quantiles": [1.1]},
                {"quantiles": [float("nan")]},
                {"rectangle_shape": [0, 2]},
                {"rectangle_shape": [2]},
                {"rectangle_shape": [2**62, 4]},
            ):
                with (
                    self.subTest(options=options, count=values.size),
                    self.assertRaises(ValueError),
                ):
                    compute_tensor_statistics(values, **options)
        for values in (numpy.array(["text"]), numpy.array([1 + 2j])):
            with self.assertRaises((ValueError, TypeError)):
                compute_tensor_statistics(values)
        truncated = onnx.TensorProto(
            data_type=onnx.TensorProto.FLOAT, dims=[2], raw_data=b"\x00\x00\x00\x00"
        )
        with self.assertRaises(ValueError):
            compute_tensor_statistics(truncated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
