# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks that historical Marlin tokens do not imply transformed serialized buffers."""

import unittest

import numpy

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

gptq = import_or_skip("onnx_light.onnx_core.gptq")
awq = import_or_skip("onnx_light.onnx_core.awq")
marlin = import_or_skip("onnx_light.onnx_core.marlin")


class TestMarlin(ExtTestCase):
    def test_historical_gptq_is_ordinary_v1_not_a_tiled_abi(self):
        for bits in (4, 8):
            codes = (numpy.arange(64 * 64) % (1 << bits)).astype(numpy.uint8).reshape(64, 64)
            groups = numpy.repeat(numpy.arange(2, dtype=numpy.int32), 32)
            zeros = numpy.full((2, 64), 1 << (bits - 1), dtype=numpy.int16)
            scales = numpy.ones((2, 64), dtype=numpy.float16)
            args = dict(in_features=64, out_features=64, group_size=32, bits=bits)
            ordinary = gptq.pack_gptq_weights(codes, zeros, scales, groups, format="gptq", **args)
            tensors = gptq.export_gptq_weights(ordinary)["tensors"]
            imported = marlin.import_marlin_weights(
                tensors, method="gptq", layout="gptq-v1", **args
            )
            self.assertEqual(imported.SerializeToString(), ordinary.SerializeToString())
            for change in (
                {"layout": "marlin"},
                {"layout": "tiled-v1"},
                {"layout": "awq-gemm"},
                {"method": "awq"},
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    marlin.import_marlin_weights(
                        tensors, **{**args, "method": "gptq", "layout": "gptq-v1", **change}
                    )
            for change in (
                {"g_idx": groups[::-1].copy()},
                {"qzeros": numpy.zeros_like(tensors["qzeros"])},
                {"qweight": tensors["qweight"].reshape(4, -1)},
            ):
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    marlin.import_marlin_weights(
                        {**tensors, **change}, method="gptq", layout="gptq-v1", **args
                    )

    def test_historical_awq_is_ordinary_gemm(self):
        tensors = {
            "qweight": numpy.full((64, 8), 0x76543210, dtype=numpy.int32),
            "qzeros": numpy.full((2, 8), 0x33333333, dtype=numpy.int32),
            "scales": numpy.ones((2, 64), dtype=numpy.float16),
        }
        args = dict(in_features=64, out_features=64, group_size=32)
        ordinary = awq.import_awq_weights(tensors, variant="gemm", **args)
        imported = marlin.import_marlin_weights(tensors, method="awq", layout="awq-gemm", **args)
        self.assertEqual(imported.SerializeToString(), ordinary.SerializeToString())
        # A runtime Marlin tile has the same byte count, but not the loading shape.
        with self.assertRaises(ValueError):
            marlin.import_marlin_weights(
                {**tensors, "qweight": tensors["qweight"].reshape(4, 128)},
                method="awq",
                layout="awq-gemm",
                **args,
            )

    def test_historical_awq_eight_bit_natural_order(self):
        # Pinned packing_utils.pack_awq uses natural byte order outside INT4.
        tensors = {
            "qweight": numpy.full((64, 16), 0x7F0080FF, dtype=numpy.int32),
            "qzeros": numpy.full((2, 16), 0x03020100, dtype=numpy.int32),
            "scales": numpy.ones((2, 64), dtype=numpy.float16),
        }
        args = dict(in_features=64, out_features=64, group_size=32, bits=8)
        value = marlin.import_marlin_weights(tensors, method="awq", layout="awq-gemm", **args)
        codes = numpy.repeat(
            numpy.tile(numpy.array([255, 128, 0, 127], dtype=numpy.uint8), 16)[:, None],
            64,
            axis=1,
        )
        numpy.testing.assert_array_equal(awq.unpack_awq_codes(value), codes)
        expected = (
            codes.astype(numpy.float32)
            - numpy.tile(numpy.arange(4, dtype=numpy.float32), 16)[:, None]
        )
        numpy.testing.assert_array_equal(awq.reconstruct_awq_weights(value), expected)
        exported = awq.export_awq_weights(value)
        self.assertEqual(exported["bits"], 8)
        for name, tensor in tensors.items():
            self.assertEqual(exported["tensors"][name].tobytes(), tensor.tobytes())
        self.assertEqual(
            awq.import_awq_weights(**exported).SerializeToString(), value.SerializeToString()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
