# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks the pinned GPTQModel ParoLinear serialization and reference contract."""

import unittest

import ml_dtypes
import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

paroquant = import_or_skip("onnx_light.onnx_core.paroquant")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


def fixture():
    """Returns tensors following the pinned AWQ producer and ParoLinear buffers."""
    k, n, group, steps = 32, 8, 16, 2
    row, col = numpy.indices((k, n))
    codes = ((row * 5 + col * 3) % 16).astype(numpy.uint32)
    zeros = numpy.resize(numpy.arange(8, dtype=numpy.uint32), (k // group, n))
    words = numpy.zeros((k, n // 8), dtype=numpy.uint32)
    zero_words = numpy.zeros((k // group, n // 8), dtype=numpy.uint32)
    # GPTQModel AWQ GEMM packs output lanes, not input channels.
    for lane, output in enumerate((0, 2, 4, 6, 1, 3, 5, 7)):
        words |= codes[:, output::8] << (lane * 4)
        zero_words |= zeros[:, output::8] << (lane * 4)
    scales = numpy.resize(numpy.array([0.125, 0.5, 2], dtype=numpy.float16), (2, n))
    pairs = numpy.tile(numpy.arange(group, dtype=numpy.int16), (steps, k // group))
    pairs[1] = numpy.roll(pairs[1].reshape(2, group), 1, axis=1).reshape(-1)
    theta = numpy.resize(
        numpy.array([0.25, -0.5, 0.75, -1], dtype=numpy.float16), (steps, k // 2)
    )
    tensors = dict(
        qweight=words.view(numpy.int32),
        qzeros=zero_words.view(numpy.int32),
        scales=scales,
        pairs=pairs,
        theta=theta,
        channel_scales=numpy.resize(numpy.array([0.5, 2, 1, 1.5], dtype=numpy.float16), (1, k)),
        bias=numpy.arange(n, dtype=numpy.float16) / 8,
    )
    parameters = dict(in_features=k, out_features=n, group_size=group, krot=steps)
    weight = (codes.astype(numpy.float32) - numpy.repeat(zeros, group, axis=0)) * numpy.repeat(
        scales.astype(numpy.float32), group, axis=0
    )
    return tensors, parameters, codes.T.astype(numpy.uint8), weight.T.copy()


def reference(input, tensors, group):
    """Applies the pinned utils/paroquant.py reference in independent NumPy form."""
    transformed = input.reshape(-1, input.shape[-1]) * tensors["channel_scales"].astype(
        numpy.float32
    )
    for pairs, theta in zip(tensors["pairs"], tensors["theta"]):
        previous = transformed.copy()
        for g in range(input.shape[-1] // group):
            local = pairs[g * group : (g + 1) * group].reshape(-1, 2)
            angles = theta[g * group // 2 : (g + 1) * group // 2].astype(numpy.float32)
            i, j = (local + g * group).T
            c, s = numpy.cos(angles), numpy.sin(angles)
            transformed[:, i] = previous[:, i] * c + previous[:, j] * s
            transformed[:, j] = -previous[:, i] * s + previous[:, j] * c
    return transformed


class TestParoQuant(ExtTestCase):
    def test_pinned_dense_weight_materialization_rounding(self):
        tensors, parameters, codes, _ = fixture()
        input = numpy.arange(32, dtype=numpy.float32).reshape(1, 32) / 8
        # packing_utils.dequantize_gemm multiplies in the saved scale dtype;
        # ParoLinear's dense fallback only then widens the weight to input dtype.
        for dtype in (numpy.float16, ml_dtypes.bfloat16):
            with self.subTest(dtype=dtype):
                tensors["scales"] = numpy.full((2, 8), 0.3, dtype=dtype)
                algebraic = (
                    codes.astype(numpy.float32) - numpy.arange(8, dtype=numpy.float32)[:, None]
                ) * numpy.repeat(tensors["scales"].T.astype(numpy.float32), 16, axis=1)
                expected_weight = algebraic.astype(dtype).astype(numpy.float32)
                self.assertFalse(numpy.array_equal(expected_weight, algebraic))
                encoded = paroquant.import_paroquant_weights(tensors, **parameters)
                numpy.testing.assert_array_equal(
                    paroquant.reconstruct_paroquant_weights(encoded), expected_weight
                )
                expected = reference(input, tensors, 16) @ expected_weight.T + tensors["bias"]
                numpy.testing.assert_allclose(
                    paroquant.apply_paroquant_linear(encoded, input),
                    expected,
                    rtol=2e-6,
                    atol=1e-4,
                )

    def test_native_bytes_reconstruction_and_reference_operator(self):
        tensors, parameters, codes, weight = fixture()
        self.assertEqual(int(tensors["qweight"][0, 0].view(numpy.uint32)), 0x5F932C60)
        self.assertEqual(int(tensors["qzeros"][0, 0].view(numpy.uint32)), 0x75316420)
        encoded = paroquant.import_paroquant_weights(tensors, **parameters)
        restored = onnx.EncodedValueProto()
        restored.ParseFromString(encoded.SerializeToString())
        numpy.testing.assert_array_equal(paroquant.unpack_paroquant_codes(restored), codes)
        numpy.testing.assert_array_equal(
            paroquant.reconstruct_paroquant_weights(restored), weight
        )
        exported = paroquant.export_paroquant_weights(restored)
        for name, value in tensors.items():
            actual = exported["tensors"][name]
            self.assertEqual(actual.dtype, value.dtype)
            self.assertEqual(actual.shape, value.shape)
            self.assertEqual(actual.tobytes(), value.tobytes())
        self.assertEqual(
            paroquant.import_paroquant_weights(**exported).SerializeToString(),
            encoded.SerializeToString(),
        )
        input = numpy.arange(2 * 3 * 32, dtype=numpy.float32).reshape(2, 3, 32) / 32
        rotated = reference(input, tensors, parameters["group_size"])
        expected = (rotated @ weight.T + tensors["bias"]).reshape(2, 3, 8)
        actual = paroquant.apply_paroquant_linear(restored, input)
        numpy.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-4)
        self.assertFalse(numpy.allclose(actual, input @ weight.T + tensors["bias"]))
        reversed_state = {
            **tensors,
            "theta": tensors["theta"][::-1].copy(),
            "pairs": tensors["pairs"][::-1].copy(),
        }
        reversed_encoded = paroquant.import_paroquant_weights(reversed_state, **parameters)
        self.assertFalse(
            numpy.allclose(actual, paroquant.apply_paroquant_linear(reversed_encoded, input))
        )
        # Neither generic dequantization nor AWQ may discard the required state.
        with self.assertRaises(ValueError):
            quantization.dequantize_tensor_proto(restored)
        tensors["theta"].fill(0)
        tensors["channel_scales"].fill(1)
        numpy.testing.assert_array_equal(
            paroquant.apply_paroquant_linear(restored, input), actual
        )

    def test_bfloat16_state_and_optional_bias(self):
        tensors, parameters, _, weight = fixture()
        del tensors["bias"]
        for name in ("scales", "theta", "channel_scales"):
            tensors[name] = tensors[name].astype(ml_dtypes.bfloat16)
        encoded = paroquant.import_paroquant_weights(tensors, **parameters)
        input = numpy.ones((1, 32), dtype=numpy.float32)
        expected = reference(input, tensors, 16) @ weight.T
        numpy.testing.assert_allclose(
            paroquant.apply_paroquant_linear(encoded, input), expected, rtol=2e-6, atol=1e-5
        )
        exported = paroquant.export_paroquant_weights(encoded)
        self.assertNotIn("bias", exported["tensors"])
        for name, value in tensors.items():
            self.assertEqual(exported["tensors"][name].tobytes(), value.tobytes())
            self.assertEqual(exported["tensors"][name].dtype, value.dtype)

    def test_invalid_state_even_for_identity(self):
        tensors, parameters, _, _ = fixture()
        tensors["theta"].fill(0)
        tensors["channel_scales"].fill(1)
        for name in tensors:
            if name == "bias":
                continue
            with self.subTest(missing=name), self.assertRaises(ValueError):
                paroquant.import_paroquant_weights(
                    {key: value for key, value in tensors.items() if key != name}, **parameters
                )
        for index in (-1, 16, 32767, 1):
            pairs = tensors["pairs"].copy()
            pairs[0, 0] = index
            with self.subTest(pair=index), self.assertRaises(ValueError):
                paroquant.import_paroquant_weights({**tensors, "pairs": pairs}, **parameters)
        for name, value in (
            ("pairs", tensors["pairs"].astype(numpy.int32)),
            ("pairs", tensors["pairs"][:1].copy()),
            ("theta", tensors["theta"][:, :-1].copy()),
            ("theta", tensors["theta"].astype(numpy.float32)),
            ("theta", numpy.full_like(tensors["theta"], numpy.nan)),
            ("channel_scales", tensors["channel_scales"].reshape(32)),
            ("channel_scales", tensors["channel_scales"].astype(numpy.float32)),
            ("channel_scales", numpy.full_like(tensors["channel_scales"], numpy.inf)),
            ("bias", numpy.zeros(7, dtype=numpy.float16)),
            ("bias", tensors["bias"].astype(numpy.float32)),
            ("bias", numpy.full(8, numpy.inf, dtype=numpy.float16)),
            ("rotation_matrix", numpy.eye(32, dtype=numpy.float16)),
            ("theta", tensors["theta"].astype(tensors["theta"].dtype.newbyteorder("S"))),
            ("theta", tensors["theta"].T),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                paroquant.import_paroquant_weights({**tensors, name: value}, **parameters)

    def test_metadata_limits_and_inputs(self):
        tensors, parameters, _, _ = fixture()
        for key, value in (
            ("bits", 8),
            ("group_axis", 0),
            ("krot", 0),
            ("krot", 1 << 62),
            ("in_features", 0),
            ("in_features", 1 << 62),
            ("out_features", 9),
            ("out_features", 1 << 62),
            ("group_size", 3),
            ("group_size", 1 << 40),
            ("upstream_revision", "main"),
            ("max_bytes", 0),
            ("max_bytes", 32 * 8 * 4 - 1),
        ):
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                paroquant.import_paroquant_weights(tensors, **{**parameters, key: value})
        encoded = paroquant.import_paroquant_weights(tensors, **parameters)
        for input in (
            numpy.ones(32, dtype=numpy.float32),
            numpy.ones((1, 31), dtype=numpy.float32),
            numpy.ones((1, 32), dtype=numpy.float16),
            numpy.full((1, 32), numpy.nan, dtype=numpy.float32),
            numpy.full((1, 32), numpy.finfo(numpy.float32).max, dtype=numpy.float32),
        ):
            with self.subTest(shape=input.shape), self.assertRaises(ValueError):
                paroquant.apply_paroquant_linear(encoded, input)
        with self.assertRaises(ValueError):
            paroquant.apply_paroquant_linear(
                encoded, numpy.ones((32, 32), dtype=numpy.float32), max_bytes=1024
            )

    def test_rejects_encoded_corruption(self):
        tensors, parameters, _, _ = fixture()
        encoded = paroquant.import_paroquant_weights(tensors, **parameters)
        for payload in (encoded.raw_data[:-1], encoded.raw_data + b"\0"):
            changed = onnx.EncodedValueProto()
            changed.ParseFromString(encoded.SerializeToString())
            changed.raw_data = payload
            for consumer in (
                paroquant.export_paroquant_weights,
                paroquant.unpack_paroquant_codes,
                paroquant.reconstruct_paroquant_weights,
            ):
                with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                    consumer(changed)
            with self.assertRaises(ValueError):
                paroquant.apply_paroquant_linear(
                    changed, numpy.ones((1, 32), dtype=numpy.float32)
                )
        changed = onnx.EncodedValueProto()
        changed.ParseFromString(encoded.SerializeToString())
        changed.struct_type.name = "onnx_light.paroquant_weights.v999"
        with self.assertRaises(ValueError):
            paroquant.export_paroquant_weights(changed)


if __name__ == "__main__":
    unittest.main()
