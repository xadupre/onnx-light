# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks the pinned native FP8 ABI without requiring Torch or GPTQModel."""

import unittest

import ml_dtypes
import numpy

import onnx_light.onnx.numpy_helper as onh
from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

fp8 = import_or_skip("onnx_light.onnx_core.gptqmodel_fp8")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


class TestGptqModelFp8(ExtTestCase):
    def test_native_bytes_grouping_and_inverse_arithmetic(self):
        # These are scalar FP8 ABI bytes, not values requantized by this adapter.
        cases = [
            ("float8_e4m3fn", [0x38, 0xBC, 0x40, 0xB0], [1, -1.5, 2, -0.5]),
            ("float8_e4m3fnuz", [0x40, 0xC4, 0x48, 0xB8], [1, -1.5, 2, -0.5]),
            ("float8_e5m2", [0x3C, 0xBE, 0x40, 0xB8], [1, -1.5, 2, -0.5]),
            ("float8_e5m2fnuz", [0x40, 0xC2, 0x44, 0xBC], [1, -1.5, 2, -0.5]),
            ("float8_e8m0fnu", [127, 128, 126, 125], [1, 2, 0.5, 0.25]),
        ]
        for fmt, codes, decoded in cases:
            for method, block in [
                ("tensor", None),
                ("row", None),
                ("block", [1, 2]),
                ("block", [2, 1]),
                ("block", [2, 2]),
                ("block", [4, 4]),
            ]:
                with self.subTest(format=fmt, method=method, block=block):
                    dtype = getattr(ml_dtypes, fmt)
                    weight = numpy.tile(numpy.array(codes, dtype=numpy.uint8), (4, 1)).view(dtype)
                    if method == "tensor":
                        scale = numpy.array(0.3, dtype=numpy.float32)
                        expanded = scale
                    elif method == "row":
                        scale = numpy.array([0.3, 1.3, 3, 7], dtype=numpy.float32)
                        expanded = scale[:, None]
                    else:
                        shape = (4 // block[0], 4 // block[1])
                        scale = numpy.arange(1, numpy.prod(shape) + 1, dtype=numpy.float32)
                        scale = (scale * numpy.float32(0.3)).reshape(shape)
                        expanded = scale.repeat(block[0], axis=0).repeat(block[1], axis=1)
                    tensors = {"weight": weight, "weight_scale_inv": scale}
                    encoded = fp8.import_gptqmodel_fp8_weights(
                        tensors,
                        in_features=4,
                        out_features=4,
                        format=fmt,
                        weight_scale_method=method,
                        weight_block_size=block,
                    )
                    self.assertEqual(
                        encoded.raw_data[weight.size :], scale.astype("<f4").tobytes()
                    )
                    restored = onnx.EncodedValueProto()
                    restored.ParseFromString(encoded.SerializeToString())
                    # Mirrors fp8.py's explicit inverse path at target_dtype=float32.
                    expected = numpy.tile(numpy.array(decoded, dtype=numpy.float32), (4, 1))
                    expected = expected / expanded
                    actual = fp8.reconstruct_gptqmodel_fp8_weights(restored)
                    self.assertEqual(actual.dtype, numpy.float32)
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                    self.assertEqual(
                        onh.to_array(quantization.dequantize_tensor_proto(restored)).tobytes(),
                        expected.tobytes(),
                    )
                    exported = fp8.export_gptqmodel_fp8_weights(restored)
                    self.assertEqual(exported["format"], fmt)
                    self.assertEqual(exported["weight_scale_method"], method)
                    self.assertEqual(exported["weight_scale_semantics"], "inverse")
                    self.assertEqual(exported["weight_block_size"], block)
                    self.assertEqual(exported["upstream_revision"], fp8.UPSTREAM_REVISION)
                    for name, tensor in tensors.items():
                        self.assertEqual(exported["tensors"][name].dtype, tensor.dtype)
                        self.assertEqual(exported["tensors"][name].shape, tensor.shape)
                        self.assertEqual(exported["tensors"][name].tobytes(), tensor.tobytes())
                    self.assertEqual(
                        fp8.import_gptqmodel_fp8_weights(**exported).SerializeToString(),
                        encoded.SerializeToString(),
                    )
                    weight.fill(0)
                    scale.fill(1)
                    self.assertEqual(
                        fp8.reconstruct_gptqmodel_fp8_weights(restored).tobytes(),
                        expected.tobytes(),
                    )

    def test_nonreversible_scale_and_division_precision(self):
        inverse = numpy.array([0x3E350913], dtype=numpy.uint32).view(numpy.float32).reshape(())
        reciprocal = numpy.float32(1) / inverse
        self.assertNotEqual((numpy.float32(1) / reciprocal).tobytes(), inverse.tobytes())
        weight = numpy.array([[1.25, 1]], dtype=ml_dtypes.float8_e4m3fn)
        tensors = {"weight": weight, "weight_scale_inv": inverse}
        encoded = fp8.import_gptqmodel_fp8_weights(
            tensors, in_features=2, out_features=1, weight_scale_method="tensor"
        )
        self.assertEqual(
            fp8.export_gptqmodel_fp8_weights(encoded)["tensors"]["weight_scale_inv"].tobytes(),
            inverse.tobytes(),
        )
        expected = weight.astype(numpy.float32) / inverse
        self.assertNotEqual(
            expected.tobytes(), (weight.astype(numpy.float32) * reciprocal).tobytes()
        )
        self.assertEqual(
            fp8.reconstruct_gptqmodel_fp8_weights(encoded).tobytes(), expected.tobytes()
        )

    def test_rejections(self):
        good = {
            "weight": numpy.ones((2, 4), dtype=ml_dtypes.float8_e4m3fn),
            "weight_scale_inv": numpy.array([0.3, 1.3], dtype=numpy.float32),
        }

        def rejects(tensors, **kwargs):
            with self.assertRaises((ValueError, RuntimeError, TypeError, OverflowError)):
                fp8.import_gptqmodel_fp8_weights(tensors, in_features=4, out_features=2, **kwargs)

        for kwargs in [
            {"format": "mxfp8"},
            {"weight_scale_method": "channel"},
            {"weight_scale_semantics": "direct"},
            {"weight_scale_semantics": "heuristic"},
            {"weight_scale_method": "block"},
            {"weight_scale_method": "block", "weight_block_size": [3, 2]},
            {"weight_scale_method": "block", "weight_block_size": [1, 3]},
            {"weight_scale_method": "block", "weight_block_size": [0, 2]},
            {"weight_block_size": [1, 2]},
            {"upstream_revision": "unverified"},
            {"max_bytes": 0},
            {"max_bytes": 31},
        ]:
            rejects(good, **kwargs)
        for scale in [
            numpy.ones((2, 1), dtype=numpy.float32),
            numpy.ones(2, dtype=numpy.float16),
            numpy.ones(2, dtype=">f4" if numpy.little_endian else "<f4"),
            numpy.array([0, 1], dtype=numpy.float32),
            numpy.array([-1, 1], dtype=numpy.float32),
            numpy.array([numpy.nan, 1], dtype=numpy.float32),
            numpy.array([numpy.inf, 1], dtype=numpy.float32),
            numpy.array([numpy.finfo(numpy.float32).smallest_subnormal, 1], dtype=numpy.float32),
        ]:
            rejects({**good, "weight_scale_inv": scale})
        rejects({"weight": good["weight"]})
        rejects({**good, "bias": numpy.ones(2, dtype=numpy.float16)})
        rejects({**good, "weight": good["weight"].view(numpy.uint8)})
        rejects({**good, "weight": good["weight"].reshape(4, 2)})
        encoded = fp8.import_gptqmodel_fp8_weights(
            good, in_features=4, out_features=2, max_bytes=32
        )
        encoded.raw_data = encoded.raw_data[:-1]
        with self.assertRaises((ValueError, RuntimeError)):
            fp8.export_gptqmodel_fp8_weights(encoded)


if __name__ == "__main__":
    unittest.main()
