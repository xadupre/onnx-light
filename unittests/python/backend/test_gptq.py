# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks the exact native GPTQModel adapter's Python boundary."""

import unittest

import numpy

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

gptq = import_or_skip("onnx_light.onnx_core.gptq")


def logical_fixture(bits=4, n=32, k=32, group_size=16):
    """Returns exact codes and asymmetric parameters with a noncontiguous group map."""
    h = (k + group_size - 1) // group_size
    row, col = numpy.indices((n, k))
    codes = ((row * 3 + col * 5) % (1 << bits)).astype(numpy.uint8)
    zeros = ((numpy.arange(h * n).reshape(h, n) % ((1 << bits) - 1)) + 1).astype(numpy.int16)
    scales = numpy.resize(numpy.array([0.125, 0.5, 2], dtype=numpy.float16), (h, n))
    groups = (numpy.arange(k) % h).astype(numpy.int32)
    return codes, zeros, scales, groups


class TestGptq(ExtTestCase):
    def test_upstream_continuous_three_bit_fixture(self):
        # Executed NumPy packing stages of upstream pack_original at UPSTREAM_REVISION;
        # only logical positions 10 and 21 contain seven, exercising both crossings.
        words = numpy.array([0xC0000000, 0x80000001, 3], dtype=numpy.uint32)
        tensors = {
            "qweight": numpy.repeat(words[:, None], 32, axis=1).view(numpy.int32),
            "qzeros": words[None, :].view(numpy.int32),
            "scales": numpy.ones((1, 32), dtype=numpy.float16),
            "g_idx": numpy.zeros(32, dtype=numpy.int32),
        }
        args = dict(format="gptq_v2", in_features=32, out_features=32, group_size=32, bits=3)
        encoded = gptq.import_gptq_weights(tensors, **args)
        x = numpy.zeros(32, dtype=numpy.uint8)
        x[[10, 21]] = 7
        codes = numpy.repeat(x[None, :], 32, axis=0)
        zeros = x[None, :].astype(numpy.int16)
        numpy.testing.assert_array_equal(gptq.unpack_gptq_codes(encoded), codes)
        numpy.testing.assert_array_equal(gptq.unpack_gptq_zeros(encoded), zeros)
        numpy.testing.assert_array_equal(
            gptq.reconstruct_gptq_weights(encoded),
            x.astype(numpy.float32)[None, :] - x.astype(numpy.float32)[:, None],
        )
        repacked = gptq.pack_gptq_weights(
            codes, zeros, tensors["scales"], tensors["g_idx"], **args
        )
        for name, actual in gptq.export_gptq_weights(repacked)["tensors"].items():
            self.assertEqual(actual.tobytes(), tensors[name].tobytes())

    def test_partial_words_and_partial_groups(self):
        for bits in (2, 4, 8):
            for word_dtype, numpy_dtype in (
                (3, numpy.int8),
                (5, numpy.int16),
                (6, numpy.int32),
                (7, numpy.int64),
            ):
                for n, k in ((1, 1), (19, 63)):
                    with self.subTest(bits=bits, word_dtype=word_dtype, n=n, k=k):
                        codes, zeros, scales, groups = logical_fixture(bits, n, k, 17)
                        encoded = gptq.pack_gptq_weights(
                            codes,
                            zeros,
                            scales,
                            groups,
                            format="gptq_v2",
                            in_features=k,
                            out_features=n,
                            group_size=17,
                            bits=bits,
                            word_dtype=word_dtype,
                        )
                        numpy.testing.assert_array_equal(gptq.unpack_gptq_codes(encoded), codes)
                        numpy.testing.assert_array_equal(gptq.unpack_gptq_zeros(encoded), zeros)
                        checkpoint = gptq.export_gptq_weights(encoded)
                        tensors = checkpoint["tensors"]
                        self.assertEqual(tensors["qweight"].dtype, numpy_dtype)
                        word_bits = numpy.dtype(numpy_dtype).itemsize * 8
                        self.assertEqual(
                            tensors["qweight"].shape, ((k * bits + word_bits - 1) // word_bits, n)
                        )
                        self.assertEqual(
                            tensors["qzeros"].shape,
                            ((k + 16) // 17, (n * bits + word_bits - 1) // word_bits),
                        )
                        self.assertEqual(
                            gptq.import_gptq_weights(**checkpoint).SerializeToString(),
                            encoded.SerializeToString(),
                        )

    def test_exact_native_roundtrip_and_group_map(self):
        for format_name, bits in (
            ("gptq", 2),
            ("gptq", 3),
            ("gptq_v2", 4),
            ("gptq_v2", 8),
            ("gptq_p", 5),
            ("gptq_p", 6),
            ("gptq_p", 7),
        ):
            with self.subTest(format=format_name, bits=bits):
                codes, zeros, scales, groups = logical_fixture(bits)
                encoded = gptq.pack_gptq_weights(
                    codes,
                    zeros,
                    scales,
                    groups,
                    format=format_name,
                    in_features=32,
                    out_features=32,
                    group_size=16,
                    bits=bits,
                )
                numpy.testing.assert_array_equal(gptq.unpack_gptq_codes(encoded), codes)
                numpy.testing.assert_array_equal(gptq.unpack_gptq_zeros(encoded), zeros)
                expected = (codes.astype(numpy.float32) - zeros[groups].T) * scales[
                    groups
                ].T.astype(numpy.float32)
                numpy.testing.assert_array_equal(gptq.reconstruct_gptq_weights(encoded), expected)
                exported = gptq.export_gptq_weights(encoded)
                self.assertEqual(exported["format"], format_name)
                self.assertEqual(exported["upstream_revision"], gptq.UPSTREAM_REVISION)
                numpy.testing.assert_array_equal(exported["tensors"]["g_idx"], groups)
                self.assertEqual(exported["tensors"]["scales"].tobytes(), scales.tobytes())
                restored = gptq.import_gptq_weights(**exported)
                self.assertEqual(restored.SerializeToString(), encoded.SerializeToString())
                again = gptq.export_gptq_weights(restored)
                for name, tensor in exported["tensors"].items():
                    self.assertEqual(again["tensors"][name].dtype, tensor.dtype)
                    self.assertEqual(again["tensors"][name].shape, tensor.shape)
                    self.assertEqual(again["tensors"][name].tobytes(), tensor.tobytes())
                # Import owns its snapshot rather than retaining caller-owned storage.
                exported["tensors"]["qweight"].fill(0)
                numpy.testing.assert_array_equal(gptq.unpack_gptq_codes(restored), codes)

    def test_binding_validation(self):
        codes, zeros, scales, groups = logical_fixture()
        args = dict(format="gptq_v2", in_features=32, out_features=32, group_size=16)
        encoded = gptq.pack_gptq_weights(codes, zeros, scales, groups, **args)
        exported = gptq.export_gptq_weights(encoded)
        for update in (
            {"format": "portable_gptq"},
            {"group_axis": 0},
            {"upstream_revision": "unverified"},
            {"max_bytes": 1},
            {"in_features": -1},
            {"in_features": 0},
            {"word_dtype": 1},
            {"word_dtype": 11},
            {"bits": 5},
        ):
            with self.subTest(update=update), self.assertRaises((ValueError, TypeError)):
                gptq.import_gptq_weights(**{**exported, **update})
        for bad_codes in (codes[:, ::-1], codes.astype(numpy.float32), codes.tolist()):
            with self.subTest(kind=type(bad_codes)), self.assertRaises((ValueError, TypeError)):
                gptq.pack_gptq_weights(bad_codes, zeros, scales, groups, **args)
        tensors = exported["tensors"]
        for change in (
            {"g_idx": groups.astype(numpy.int64)},
            {"g_idx": numpy.full_like(groups, -1)},
            {"g_idx": numpy.full_like(groups, 2)},
            {"qweight": tensors["qweight"].astype(">i4" if numpy.little_endian else "<i4")},
            {"qzeros": tensors["qzeros"][:, :-1].copy()},
            {"scales": scales.T},
        ):
            with self.subTest(change=list(change)), self.assertRaises((ValueError, TypeError)):
                gptq.import_gptq_weights(**{**exported, "tensors": {**tensors, **change}})
        for invalid_scale in (0, -1, numpy.inf, numpy.nan):
            with self.subTest(scale=invalid_scale), self.assertRaises(ValueError):
                gptq.pack_gptq_weights(
                    codes, zeros, numpy.full_like(scales, invalid_scale), groups, **args
                )
        with self.assertRaises(ValueError):
            gptq.import_gptq_weights(**{**exported, "tensors": {**tensors, "bias": scales}})

    def test_legacy_zero_conversion_is_explicit_and_lossless_only(self):
        codes, zeros, scales, groups = logical_fixture()
        args = dict(in_features=32, out_features=32, group_size=16)
        zeros.fill(0)
        modern = gptq.pack_gptq_weights(codes, zeros, scales, groups, format="gptq_v2", **args)
        numpy.testing.assert_array_equal(gptq.unpack_gptq_zeros(modern), zeros)
        with self.assertRaises(ValueError):
            gptq.pack_gptq_weights(codes, zeros, scales, groups, format="gptq", **args)
        zeros.fill(16)
        for format_name in ("gptq", "gptq_v2", "gptq_p"):
            with self.subTest(format=format_name), self.assertRaises(ValueError):
                gptq.pack_gptq_weights(codes, zeros, scales, groups, format=format_name, **args)


if __name__ == "__main__":
    unittest.main()
