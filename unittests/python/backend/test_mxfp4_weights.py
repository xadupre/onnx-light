# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests complete MXFP4 weights against source-derived compressed-tensors fixtures."""

import numpy as np

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


class TestMxfp4Weights(ExtTestCase):
    @staticmethod
    def tensor(array):
        """Returns an owning native FLOAT tensor."""
        array = np.ascontiguousarray(array, dtype=np.float32)
        return runtime.tensor_from_numpy(
            "W", onnx.TensorProto.FLOAT, list(array.shape), array.view(np.uint8).reshape(-1)
        )

    @staticmethod
    def array(tensor):
        """Returns an independent NumPy array."""
        return np.from_dlpack(tensor).copy()

    def test_pinned_values_packing_and_wire_parameters(self):
        # Source-derived from compressed-tensors eb058d630f7344099459b6795b1e3eeffdda4b40.
        levels = np.array(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
        )
        weights = self.tensor(np.concatenate([levels, levels]).reshape(1, 32))
        encoded = quantization.quantize_mxfp4_weights(weights)
        self.assertEqual(encoded.struct_type.name, "onnx_light.mxfp4_weights.v1")

        parsed = onnx.EncodedValueProto()
        parsed.ParseFromString(encoded.SerializeToString())
        parameters = quantization.export_mxfp4_weights(parsed)
        self.assertEqual(parameters.weight_scale.data_type, onnx.TensorProto.UINT8)
        self.assertEqual(parameters.weight_scale.raw_data(), bytes([127]))
        self.assertEqual(
            parameters.weight_packed.raw_data(),
            bytes([0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE] * 2),
        )

        imported = quantization.import_mxfp4_weights(parameters)
        self.assertEqual(imported.raw_data, parsed.raw_data)
        np.testing.assert_array_equal(
            self.array(runtime.dequantize_tensor(imported)),
            np.concatenate([levels, levels]).reshape(1, 32),
        )
        proto = runtime.dequantize_tensor_proto(parsed)
        self.assertEqual(proto.data_type, onnx.TensorProto.FLOAT)

    def test_reference_scale_selection(self):
        weights = self.tensor(np.ones((1, 32), dtype=np.float32))
        parameters = quantization.export_mxfp4_weights(
            quantization.quantize_mxfp4_weights(weights)
        )
        self.assertEqual(parameters.weight_scale.raw_data(), bytes([125]))
        self.assertEqual(parameters.weight_packed.raw_data(), bytes([0x66] * 16))

    def test_scale_boundaries_zero_groups_and_validation(self):
        zeros = self.tensor(np.zeros((1, 32), dtype=np.float32))
        encoded = quantization.quantize_mxfp4_weights(zeros)
        parameters = quantization.export_mxfp4_weights(encoded)
        self.assertEqual(parameters.weight_scale.raw_data(), bytes([127]))
        self.assertEqual(parameters.weight_packed.raw_data(), bytes(16))
        np.testing.assert_array_equal(
            self.array(runtime.dequantize_tensor(encoded)), np.zeros((1, 32), dtype=np.float32)
        )

        small = np.zeros((1, 32), dtype=np.float32)
        small[0, 0] = np.float32(6 * 2.0**-127)
        parameters = quantization.export_mxfp4_weights(
            quantization.quantize_mxfp4_weights(self.tensor(small))
        )
        self.assertEqual(parameters.weight_scale.raw_data(), bytes([0]))

        incomplete = self.tensor(np.zeros((1, 31), dtype=np.float32))
        with self.assertRaises(ValueError):
            quantization.quantize_mxfp4_weights(incomplete)

        parameters.weight_scale = runtime.tensor_from_numpy(
            "weight_scale", onnx.TensorProto.UINT8, [1, 1], np.array([255], dtype=np.uint8)
        )
        with self.assertRaises(ValueError):
            quantization.import_mxfp4_weights(parameters)
