# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests complete NVFP4 weights against source-derived compressed-tensors fixtures.

The reference is eb058d630f7344099459b6795b1e3eeffdda4b40:
compressors/nvfp4/helpers.py, quantization/utils/fp4_utils.py and
tests/test_quantization/test_utils/test_fp4_utils.py.
"""

import unittest

import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")


class TestNvfp4Weights(ExtTestCase):
    @staticmethod
    def tensor(array, dtype=onnx.TensorProto.FLOAT):
        """Returns an owning native tensor."""
        array = numpy.ascontiguousarray(array)
        return runtime.tensor_from_numpy(
            "W", dtype, list(array.shape), array.view(numpy.uint8).reshape(-1)
        )

    @staticmethod
    def array(tensor):
        """Returns an independent NumPy array."""
        return numpy.from_dlpack(tensor).copy()

    def test_rounding_packing_and_wire_parameters(self):
        # compressed-tensors eb058d6: helpers.quantize_and_pack_fp4 CPU path,
        # _round_to_fp4 ties-to-even and pack_fp4_to_uint8 low-nibble first.
        boundaries = numpy.array(
            [
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5,
                6,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5,
                -6,
            ],
            dtype=numpy.float32,
        )
        weights = numpy.stack([numpy.concatenate([boundaries, boundaries])])
        scales = numpy.array([[1.0625, 1.1875]], dtype=numpy.float32)
        encoded = runtime.quantize_nvfp4_weights(self.tensor(weights), self.tensor(scales), 1.0)
        self.assertEqual(encoded.struct_type.name, "onnx_light.nvfp4_weights.v1")
        parsed = onnx.EncodedValueProto()
        parsed.ParseFromString(encoded.SerializeToString())
        parameters = runtime.export_nvfp4_weights(parsed)
        self.assertEqual(parameters.weight_scale.data_type, onnx.TensorProto.FLOAT8E4M3FN)
        self.assertEqual(parameters.weight_scale.raw_data(), bytes([0x38, 0x3A]))
        self.assertEqual(parameters.weight_global_scale, 1.0)
        self.assertEqual(
            parameters.weight_packed.raw_data()[:8],
            bytes([0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE]),
        )
        imported = runtime.import_nvfp4_weights(parameters)
        self.assertEqual(imported.raw_data, parsed.raw_data)
        actual = self.array(runtime.dequantize_tensor(parsed))
        restored = self.array(runtime.dequantize_tensor(imported))
        numpy.testing.assert_array_equal(actual, restored)
        numpy.testing.assert_array_equal(
            actual[0, :16],
            numpy.array(
                [0, 1, 1, 2, 2, 4, 4, 6, -0.0, -1, -1, -2, -2, -4, -4, -6], dtype=numpy.float32
            ),
        )
        proto = runtime.dequantize_tensor_proto(parsed)
        self.assertEqual(proto.data_type, onnx.TensorProto.FLOAT)
        self.assertEqual(proto.raw_data, actual.tobytes())

    def test_fp8_rounding_precedes_quantization(self):
        weights = self.tensor(numpy.full((1, 16), 0.26, dtype=numpy.float32))
        local = self.tensor(numpy.array([[1.0625]], dtype=numpy.float32))
        encoded = runtime.quantize_nvfp4_weights(weights, local, 1.0)
        parameters = runtime.export_nvfp4_weights(encoded)
        # 1.0625 rounds to FP8 1.0: .26/1 > .25 gives .5, whereas
        # quantizing with the unrounded "effective scale" gives zero.
        self.assertEqual(parameters.weight_packed.raw_data(), bytes([0x11] * 8))
        numpy.testing.assert_array_equal(
            self.array(runtime.dequantize_tensor(encoded)),
            numpy.full((1, 16), 0.5, dtype=numpy.float32),
        )
        self.assertLess(numpy.float32(0.26) / numpy.float32(1.0625), 0.25)
        # Passing the exported FP8 tensor does not round it again.
        again = runtime.quantize_nvfp4_weights(weights, parameters.weight_scale, 1.0)
        self.assertEqual(encoded.raw_data, again.raw_data)

    def test_global_scale_direction_and_row_groups(self):
        weights = numpy.full((2, 32), 3, dtype=numpy.float32)
        local = numpy.array([[1, 2], [4, 8]], dtype=numpy.float32)
        encoded = runtime.quantize_nvfp4_weights(self.tensor(weights), self.tensor(local), 2.0)
        parameters = runtime.export_nvfp4_weights(encoded)
        self.assertEqual(parameters.weight_global_scale, 2.0)
        self.assertEqual(parameters.weight_scale.raw_data(), bytes([0x38, 0x40, 0x48, 0x50]))
        self.assertEqual(
            parameters.weight_packed.raw_data(),
            bytes([0x77] * 8 + [0x55] * 8 + [0x33] * 8 + [0x22] * 8),
        )
        expected = numpy.full((2, 32), 3, dtype=numpy.float32)
        expected[1, 16:] = 4
        numpy.testing.assert_array_equal(self.array(runtime.dequantize_tensor(encoded)), expected)

    def test_preset_calibrated_reference_fixture(self):
        # Source-derived fixture at compressed-tensors eb058d6:
        # generate_gparam(amax=12) = 2688/12 = 224;
        # calculate_qparams: FP8(224 * (group_amax/6)) = [448, 112].
        levels = numpy.array(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=numpy.float32,
        )
        weights = numpy.concatenate([levels * 2, levels * 0.5]).reshape(1, 32)
        encoded = runtime.quantize_nvfp4_weights(
            self.tensor(weights),
            self.tensor(numpy.array([[448, 112]], dtype=numpy.float32)),
            224.0,
        )
        parameters = runtime.export_nvfp4_weights(encoded)
        self.assertEqual(parameters.weight_scale.raw_data(), bytes([0x7E, 0x6E]))
        self.assertEqual(parameters.weight_global_scale, 224.0)
        self.assertEqual(
            parameters.weight_packed.raw_data(),
            bytes([0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE] * 2),
        )
        numpy.testing.assert_array_equal(self.array(runtime.dequantize_tensor(encoded)), weights)

    def test_float32_division_order(self):
        weights = numpy.full((1, 16), 2**-11, dtype=numpy.float32)
        local = numpy.array([[2**-9]], dtype=numpy.float32)
        encoded = runtime.quantize_nvfp4_weights(self.tensor(weights), self.tensor(local), 7.0)
        parameters = runtime.export_nvfp4_weights(encoded)
        self.assertEqual(parameters.weight_scale.raw_data(), b"\x01")
        self.assertEqual(parameters.weight_packed.raw_data(), bytes([0x33] * 8))
        effective = numpy.float32(numpy.float32(2**-9) / numpy.float32(7))
        self.assertLess(numpy.float32(numpy.float32(2**-11) / effective), 1.75)
        self.assertEqual(
            numpy.float32(numpy.float32(numpy.float32(2**-11) * 7) / numpy.float32(2**-9)), 1.75
        )
        numpy.testing.assert_array_equal(
            self.array(runtime.dequantize_tensor(encoded)),
            numpy.full((1, 16), numpy.float32(1.5) * effective, dtype=numpy.float32),
        )

    def test_global_scale_bits_survive_parameter_export(self):
        global_scale = float(numpy.float32(0.3))
        encoded = runtime.quantize_nvfp4_weights(
            self.tensor(numpy.ones((1, 16), dtype=numpy.float32)),
            self.tensor(numpy.ones((1, 1), dtype=numpy.float32)),
            global_scale,
        )
        self.assertEqual(encoded.raw_data[-4:], b"\x9a\x99\x99\x3e")
        parameters = runtime.export_nvfp4_weights(encoded)
        self.assertEqual(parameters.weight_global_scale, global_scale)
        self.assertEqual(runtime.import_nvfp4_weights(parameters).raw_data, encoded.raw_data)

    def test_adjacent_fp4_and_fp8_rounding_values(self):
        midpoints = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]
        tie_codes = [0, 2, 2, 4, 4, 6, 6]
        for index, (midpoint, tie_code) in enumerate(zip(midpoints, tie_codes)):
            with self.subTest(midpoint=midpoint):
                middle = numpy.float32(midpoint)
                values = numpy.zeros((1, 16), dtype=numpy.float32)
                values[0, :3] = [
                    numpy.nextafter(middle, numpy.float32(0)),
                    middle,
                    numpy.nextafter(middle, numpy.float32(numpy.inf)),
                ]
                encoded = runtime.quantize_nvfp4_weights(
                    self.tensor(values), self.tensor(numpy.ones((1, 1), dtype=numpy.float32)), 1.0
                )
                self.assertEqual(
                    runtime.export_nvfp4_weights(encoded).weight_packed.raw_data(),
                    bytes([index | (tie_code << 4), index + 1] + [0] * 6),
                )
        middle = numpy.float32(1.0625)
        for scale, byte in (
            (numpy.nextafter(middle, numpy.float32(0)), 0x38),
            (middle, 0x38),
            (numpy.nextafter(middle, numpy.float32(numpy.inf)), 0x39),
        ):
            with self.subTest(scale=scale):
                encoded = runtime.quantize_nvfp4_weights(
                    self.tensor(numpy.zeros((1, 16), dtype=numpy.float32)),
                    self.tensor(numpy.array([[scale]], dtype=numpy.float32)),
                    1.0,
                )
                self.assertEqual(
                    runtime.export_nvfp4_weights(encoded).weight_scale.raw_data(), bytes([byte])
                )

    def test_zero_groups_and_nonfinite_policy(self):
        weights = numpy.zeros((1, 16), dtype=numpy.float32)
        weights[0, 0] = -0.0
        scales = numpy.ones((1, 1), dtype=numpy.float32)
        encoded = runtime.quantize_nvfp4_weights(self.tensor(weights), self.tensor(scales), 1.0)
        self.assertEqual(runtime.export_nvfp4_weights(encoded).weight_packed.raw_data(), bytes(8))
        for value in (0, -1, numpy.inf, numpy.nan, numpy.float32(2**-20)):
            with self.subTest(local=value), self.assertRaises(ValueError):
                runtime.quantize_nvfp4_weights(
                    self.tensor(weights),
                    self.tensor(numpy.array([[value]], dtype=numpy.float32)),
                    1.0,
                )
        for value in (0, -1, numpy.inf, numpy.nan):
            with self.subTest(global_scale=value), self.assertRaises(ValueError):
                runtime.quantize_nvfp4_weights(self.tensor(weights), self.tensor(scales), value)
        for value in (numpy.nan, numpy.inf, -numpy.inf):
            weights[0, 0] = value
            with self.subTest(weight=value), self.assertRaises(ValueError):
                runtime.quantize_nvfp4_weights(self.tensor(weights), self.tensor(scales), 1.0)

    def test_normalization_underflow_uses_quotient_sign(self):
        weights = numpy.full((1, 16), -1e-30, dtype=numpy.float32)
        encoded = runtime.quantize_nvfp4_weights(
            self.tensor(weights),
            self.tensor(numpy.ones((1, 1), dtype=numpy.float32)),
            float(numpy.finfo(numpy.float32).tiny),
        )
        self.assertEqual(runtime.export_nvfp4_weights(encoded).weight_packed.raw_data(), bytes(8))
        self.assertFalse(numpy.signbit(self.array(runtime.dequantize_tensor(encoded))).any())

    def test_shapes_and_malformed_wire(self):
        scales = self.tensor(numpy.ones((1, 1), dtype=numpy.float32))
        for shape in ((16,), (1, 15), (1, 17), (0, 16), (1, 0), (1, 1, 16)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                runtime.quantize_nvfp4_weights(
                    self.tensor(numpy.zeros(shape, dtype=numpy.float32)), scales, 1.0
                )
        weights = self.tensor(numpy.ones((1, 16), dtype=numpy.float32))
        for shape in ((1,), (1, 2), (2, 1)):
            with self.subTest(scale_shape=shape), self.assertRaises(ValueError):
                runtime.quantize_nvfp4_weights(
                    weights, self.tensor(numpy.ones(shape, dtype=numpy.float32)), 1.0
                )
        encoded = runtime.quantize_nvfp4_weights(weights, scales, 1.0)
        for payload in (encoded.raw_data[:-1], encoded.raw_data + b"\x00"):
            broken = onnx.EncodedValueProto()
            broken.CopyFrom(encoded)
            broken.raw_data = payload
            with self.assertRaises(ValueError):
                runtime.export_nvfp4_weights(broken)
            with self.assertRaises(ValueError):
                runtime.dequantize_tensor(broken)
        parameters = runtime.export_nvfp4_weights(encoded)
        parameters.weight_global_scale = 0.0
        with self.assertRaises(ValueError):
            runtime.import_nvfp4_weights(parameters)

    def test_public_module_exports(self):
        from onnx_light.onnx_core import quantization

        for name in (
            "Nvfp4Weights",
            "quantize_nvfp4_weights",
            "export_nvfp4_weights",
            "import_nvfp4_weights",
        ):
            self.assertIs(getattr(quantization, name), getattr(runtime, name))
            self.assertIn(name, quantization.__all__)


if __name__ == "__main__":
    unittest.main()
