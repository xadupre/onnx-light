# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests independent wire goldens, not quantize/dequantize round trips."""

import struct
import unittest

import ml_dtypes
import numpy

from onnx_light import onnx
import onnx_light.onnx.numpy_helper as onh
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


class TestQuantizationPrecision(ExtTestCase):
    output_types = (
        (onnx.TensorProto.FLOAT16, numpy.float16),
        (onnx.TensorProto.BFLOAT16, ml_dtypes.bfloat16),
        (onnx.TensorProto.FLOAT, numpy.float32),
        (onnx.TensorProto.DOUBLE, numpy.float64),
    )

    def encoded(
        self,
        output_type,
        *,
        precision=onnx.TensorProto.FLOAT,
        scale=1,
        offset=0,
        code=3,
        codebook=None,
        cast_type=None,
        cast_value=1,
    ):
        """Constructs an independently packed single-element portable value."""
        plan = runtime.make_quantization_plan(runtime.QuantizationFormat.INT4, 1, 1)
        plan.reconstruction_dtype = precision
        run = plan.run(0)
        if codebook is not None:
            run.layout.method = runtime.QuantizationMethod.CODEBOOK
            run.layout.bits = 1
            run.layout.books = len(codebook)
            run.layout.entries = 1
            code = 0
        if cast_type is not None:
            run.layout.method = runtime.QuantizationMethod.CAST
            run.layout.cast_type = cast_type
        plan.set_run(0, run)
        value = onnx.EncodedValueProto()
        value.struct_type.CopyFrom(runtime.make_quantization_type(plan))
        value.logical_type.tensor_type.elem_type = output_type
        value.logical_type.tensor_type.shape.dim.add(dim_value=1)
        payload = b"\0" + struct.pack("<ddd", scale, 0, offset)
        if codebook is not None:
            payload += struct.pack(f"<{len(codebook)}d", *codebook)
        if cast_type is not None:
            formats = {
                onnx.TensorProto.DOUBLE: "<d",
                onnx.TensorProto.FLOAT: "<f",
                onnx.TensorProto.FLOAT16: "<e",
                onnx.TensorProto.BFLOAT16: "<H",
            }
            if cast_type == onnx.TensorProto.BFLOAT16:
                cast_value = int(
                    numpy.array(cast_value, dtype=ml_dtypes.bfloat16).view(numpy.uint16)
                )
            payload += struct.pack(formats[cast_type], cast_value)
        else:
            payload += bytes([code & 15])
        value.raw_data = payload
        return value

    def assert_decoded(self, value, expected, dtype):
        """Checks proto and runtime consumers against identical expected bytes."""
        wire = onnx.EncodedValueProto()
        wire.ParseFromString(value.SerializeToString())
        result = runtime.dequantize_tensor_proto(wire)
        self.assertEqual(bytes(result.raw_data), numpy.array([expected], dtype=dtype).tobytes())
        tensor = runtime.dequantize_tensor(wire)
        self.assertEqual(runtime.tensor_to_numpy(tensor).tobytes(), bytes(result.raw_data))

    def test_affine_independent_rounding_golden(self):
        # 3*(1+2^-23) is halfway between FLOAT values; rounds upward before subtracting 3.
        scale = 1 + 2**-23
        float_golden = 2**-21
        double_golden = 3 * 2**-23
        centered = numpy.subtract(numpy.float32(3), numpy.float32(0), dtype=numpy.float32)
        product = numpy.multiply(centered, numpy.float32(scale), dtype=numpy.float32)
        reference = numpy.add(product, numpy.float32(-3), dtype=numpy.float32)
        self.assertEqual(float(reference), float_golden)
        self.assertNotEqual(float(numpy.float32(double_golden)), float_golden)
        for output_type, dtype in self.output_types:
            for precision, golden in (
                (onnx.TensorProto.FLOAT, float_golden),
                (onnx.TensorProto.DOUBLE, double_golden),
            ):
                with self.subTest(output=output_type, precision=precision):
                    value = self.encoded(output_type, precision=precision, scale=scale, offset=-3)
                    version = 2 if precision == onnx.TensorProto.FLOAT else 1
                    self.assertEqual(
                        value.struct_type.name, f"onnx_light.quantization.v{version}/int4"
                    )
                    self.assert_decoded(value, golden, dtype)

    def test_additive_codebook_independent_rounding_golden(self):
        books = [2**24, 1, -(2**24)]
        reference = numpy.float32(0)
        for entry in books:
            reference = numpy.add(reference, numpy.float32(entry), dtype=numpy.float32)
        self.assertEqual(float(reference), 0)
        self.assertEqual(sum(books), 1)
        for output_type, dtype in self.output_types:
            for precision, golden in ((onnx.TensorProto.FLOAT, 0), (onnx.TensorProto.DOUBLE, 1)):
                with self.subTest(output=output_type, precision=precision):
                    value = self.encoded(output_type, precision=precision, codebook=books)
                    self.assert_decoded(value, golden, dtype)

    def test_cast_operand_rounds_before_scale(self):
        for output_type, dtype in self.output_types:
            for cast_type, _ in self.output_types:
                with self.subTest(output=output_type, cast=cast_type):
                    # The DOUBLE storage operand rounds to 1 before multiplication.
                    value = self.encoded(
                        output_type, scale=2, cast_type=cast_type, cast_value=1 + 2**-24
                    )
                    self.assert_decoded(value, 2, dtype)

    def test_inverse_transform_and_outlier_rounding(self):
        forward = [2**-24, 0, 0, -(2**-24), 1, 0, 1, 0, 1]
        inverse = [2**24, 0, 0, 1, 1, 0, -(2**24), 0, 1]
        for precision, golden in ((onnx.TensorProto.FLOAT, 0), (onnx.TensorProto.DOUBLE, 1)):
            plan = runtime.make_quantization_plan(runtime.QuantizationFormat.INT4, 3, 3)
            plan.reconstruction_dtype = precision
            plan.transform_size = 3
            plan.forward = forward
            plan.inverse = inverse
            value = onnx.EncodedValueProto()
            value.struct_type.CopyFrom(runtime.make_quantization_type(plan))
            value.logical_type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
            value.logical_type.tensor_type.shape.dim.add(dim_value=3)
            value.raw_data = (
                b"\0"
                + struct.pack("<18d", *forward, *inverse)
                + struct.pack("<ddd", 1, 0, 0)
                + b"\x11\x01"
            )
            result = onh.to_array(runtime.dequantize_tensor_proto(value))
            self.assertEqualArray(result, numpy.array([golden, 1, 1], dtype=numpy.float64))

            # Outliers are restored after the inverse transform, rounding only in v2.
            plan.outliers = [0]
            value.struct_type.CopyFrom(runtime.make_quantization_type(plan))
            raw = bytes(value.raw_data)
            exception = 1 + 2**-24
            value.raw_data = raw[:145] + struct.pack("<qd", 0, exception) + raw[145:]
            result = onh.to_array(runtime.dequantize_tensor_proto(value))
            self.assertEqual(result[0], 1 if precision == onnx.TensorProto.FLOAT else exception)

    def test_subnormals_and_ties_to_even(self):
        for output_type, dtype in self.output_types:
            for exponent in (-150, -149, -134, -133, -25, -24):
                with self.subTest(output=output_type, exponent=exponent):
                    value = self.encoded(output_type, scale=2.0**exponent, code=1)
                    expected = numpy.float32(2.0**exponent)
                    self.assert_decoded(value, expected, dtype)
        for offset, golden in ((2**-24, 1), (3 * 2**-24, 1 + 2**-22)):
            value = self.encoded(onnx.TensorProto.DOUBLE, code=1, offset=offset)
            self.assert_decoded(value, golden, numpy.float64)

    def test_nonfinite_and_overflow_are_rejected(self):
        for output_type, _ in self.output_types:
            for parameters in (
                {"scale": float("inf")},
                {"scale": float("nan")},
                {"offset": float("nan")},
                {"scale": 2.0**128},
                {"scale": 2.0**127},  # Finite operands, overflowing product.
                {"codebook": [float("inf")]},
                {"codebook": [2.0**128]},
                {"cast_type": onnx.TensorProto.DOUBLE, "cast_value": float("nan")},
                {"cast_type": onnx.TensorProto.DOUBLE, "cast_value": 2.0**128},
            ):
                with self.subTest(output=output_type, parameters=parameters):
                    value = self.encoded(output_type, **parameters)
                    for decoder in (runtime.dequantize_tensor, runtime.dequantize_tensor_proto):
                        with self.assertRaises(ValueError):
                            decoder(value)
        # A finite FLOAT reconstruction can still overflow the requested output.
        for output_type, scale in (
            (onnx.TensorProto.FLOAT16, 2.0**16),
            (onnx.TensorProto.BFLOAT16, float(numpy.finfo(numpy.float32).max)),
        ):
            value = self.encoded(output_type, code=1, scale=scale)
            with self.assertRaises(ValueError):
                runtime.dequantize_tensor_proto(value)

    def test_precision_validation_and_ort_gate(self):
        plan = runtime.make_quantization_plan(runtime.QuantizationFormat.INT4, 1, 1)
        self.assertEqual(plan.reconstruction_dtype, onnx.TensorProto.DOUBLE)
        plan.reconstruction_dtype = onnx.TensorProto.FLOAT16
        with self.assertRaises(ValueError):
            runtime.make_quantization_type(plan)
        plan = runtime.make_matmul_nbits_plan(
            runtime.QuantizationFormat.ORT_MATMULNBITS_INT4, 16, 1, 16
        )
        plan.reconstruction_dtype = onnx.TensorProto.FLOAT
        with self.assertRaises(ValueError):
            runtime.quantize_tensor_proto(
                onh.from_array(numpy.zeros((16, 1), numpy.float32)), plan
            )
        value = self.encoded(onnx.TensorProto.FLOAT)
        value.struct_type.name = "onnx_light.quantization.v3/int4"
        with self.assertRaises(ValueError):
            runtime.dequantize_tensor_proto(value)


if __name__ == "__main__":
    unittest.main()
