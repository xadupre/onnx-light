# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks native GPTQModel AWQ layouts independently of portable affine packing."""

import unittest

import ml_dtypes
import numpy

import onnx_light.onnx.numpy_helper as onh
from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

awq = import_or_skip("onnx_light.onnx_core.awq")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


def gemm_fixture(n=128, k=128, group=32):
    """Returns output-packed words from the pinned GEMM producer's lane order."""
    row, col = numpy.indices((n, k))
    codes = ((row * 3 + col * 5 + col // 8) % 16).astype(numpy.uint8)
    zeros = ((numpy.arange(k // group)[:, None] * 5 + numpy.arange(n)) % 16).astype(numpy.uint8)
    scales = numpy.resize(numpy.array([0.125, 0.5, 2], dtype=numpy.float16), (k // group, n))
    words = numpy.zeros((k, n // 8), dtype=numpy.uint32)
    zero_words = numpy.zeros((k // group, n // 8), dtype=numpy.uint32)
    for lane, output in enumerate((0, 2, 4, 6, 1, 3, 5, 7)):
        words |= codes[output::8].T.astype(numpy.uint32) << (lane * 4)
        zero_words |= zeros[:, output::8].astype(numpy.uint32) << (lane * 4)
    tensors = {
        "qweight": words.view(numpy.int32),
        "qzeros": zero_words.view(numpy.int32),
        "scales": scales,
    }
    parameters = dict(variant="gemm", in_features=k, out_features=n, group_size=group)
    expected = (
        codes.astype(numpy.float32) - numpy.repeat(zeros.T, group, axis=1)
    ) * numpy.repeat(scales.T.astype(numpy.float32), group, axis=1)
    return tensors, parameters, codes, expected


def vector_fixture(variant, group=64, k=128):
    """Returns source-derived GEMV or four-row interleaved GEMV_FAST storage."""
    n = 64
    groups = k // group
    multiplier = 1 if group >= 128 else 128 // group
    words = ((groups + 7) // 8 + multiplier - 1) // multiplier * multiplier
    padded = words * 8
    row, col = numpy.indices((n, k))
    codes = ((row * 7 + col * 3 + col // 16) % 16).astype(numpy.uint8)
    zeros = ((numpy.arange(n)[:, None] + numpy.arange(groups) * 3) % 16).astype(numpy.uint8)
    scales = numpy.full((n, padded), 7, dtype=numpy.float16)
    scales[:, :groups] = numpy.resize(
        numpy.array([0.3, 1.125, 0.0625], dtype=numpy.float16), (n, groups)
    )
    parameters = dict(variant=variant, in_features=k, out_features=n, group_size=group)
    if variant == "gemv":
        weights = numpy.zeros((n, k // 8), dtype=numpy.uint32)
        padded_zeros = numpy.full((n, padded), 13, dtype=numpy.uint8)
        padded_zeros[:, :groups] = zeros
        zero_words = numpy.zeros((n, words), dtype=numpy.uint32)
        for lane in range(8):
            weights |= codes[:, lane::8].astype(numpy.uint32) << (lane * 4)
            zero_words |= padded_zeros[:, lane::8].astype(numpy.uint32) << (lane * 4)
        tensors = {
            "qweight": weights.view(numpy.int32),
            "qzeros": zero_words.view(numpy.int32),
            "scales": scales,
        }
        expected = (
            codes.astype(numpy.float32) - numpy.repeat(zeros, group, axis=1)
        ) * numpy.repeat(scales[:, :groups].astype(numpy.float32), group, axis=1)
    else:
        # Source pack_intweight uses tensor reshapes/transposes, independently
        # of the C++ decoder's index-address calculation.
        shuffled = codes.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
        shuffled = shuffled.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
        shuffled = shuffled.reshape(n // 4, 4, k // 64, 64).transpose(0, 2, 1, 3)
        lanes = shuffled.reshape(n // 4, k // 64, 64, 4).astype(numpy.uint16)
        weights = sum(lanes[..., lane] << (lane * 4) for lane in range(4))
        offsets = numpy.full((n, padded), -3, dtype=numpy.float16)
        offsets[:, :groups] = -(scales[:, :groups].astype(numpy.float32) * zeros).astype(
            numpy.float16
        )
        tensors = {
            "qweight": weights.reshape(n // 4, k).view(numpy.int16),
            "scaled_zeros" if variant == "llm-awq" else "qzeros": offsets.T.copy(),
            "scales": scales.T.copy(),
        }
        expected = codes.astype(numpy.float32) * numpy.repeat(
            scales[:, :groups].astype(numpy.float32), group, axis=1
        ) + numpy.repeat(offsets[:, :groups].astype(numpy.float32), group, axis=1)
    return tensors, parameters, codes, expected


class TestAwq(ExtTestCase):
    def test_checkpoint_groups_are_not_gpu_launch_restrictions(self):
        fixtures = [gemm_fixture(n=16, k=120, group=24)]
        for variant in ("gemm", "gemv", "gemv_fast", "llm-awq"):
            fixtures.append(
                gemm_fixture(k=256, group=256)
                if variant == "gemm"
                else vector_fixture(variant, group=256, k=256)
            )
        fixtures.extend(vector_fixture("gemv", group=group) for group in (16, 32))
        for tensors, parameters, codes, expected in fixtures:
            for full_group_sentinel in (False, True):
                if full_group_sentinel:
                    if parameters["group_size"] != parameters["in_features"]:
                        continue
                    parameters = {**parameters, "group_size": -1}
                with self.subTest(parameters=parameters):
                    encoded = awq.import_awq_weights(tensors, **parameters)
                    numpy.testing.assert_array_equal(awq.unpack_awq_codes(encoded), codes)
                    numpy.testing.assert_array_equal(
                        awq.reconstruct_awq_weights(encoded), expected
                    )
                    self.assertEqual(
                        awq.export_awq_weights(encoded)["group_size"], parameters["group_size"]
                    )

    def test_vector_layouts_padding_and_scaled_zeros(self):
        for variant, group in (("gemv", 64), ("gemv_fast", 16), ("llm-awq", 32)):
            with self.subTest(variant=variant):
                tensors, parameters, codes, expected = vector_fixture(variant, group)
                if variant == "gemv":
                    self.assertEqual(int(tensors["qweight"][0, 0].view(numpy.uint32)), 0x52FC9630)
                else:
                    packed = tensors["qweight"].view(numpy.uint16)
                    for column, word in ((0, 0x9180), (8, 0xB3A2), (64, 0xD5C4)):
                        self.assertEqual(int(packed[0, column]), word)
                encoded = awq.import_awq_weights(tensors, **parameters)
                numpy.testing.assert_array_equal(awq.unpack_awq_codes(encoded), codes)
                numpy.testing.assert_array_equal(awq.reconstruct_awq_weights(encoded), expected)
                exported = awq.export_awq_weights(encoded)
                for name, value in tensors.items():
                    actual = exported["tensors"][name]
                    self.assertEqual(actual.dtype, value.dtype)
                    self.assertEqual(actual.shape, value.shape)
                    self.assertEqual(actual.tobytes(), value.tobytes())
                self.assertEqual(
                    awq.import_awq_weights(**exported).SerializeToString(),
                    encoded.SerializeToString(),
                )
                if variant != "gemv":
                    wrong_name = "qzeros" if variant == "llm-awq" else "scaled_zeros"
                    right_name = "scaled_zeros" if variant == "llm-awq" else "qzeros"
                    renamed = {**tensors, wrong_name: tensors[right_name]}
                    del renamed[right_name]
                    with self.assertRaises(ValueError):
                        awq.import_awq_weights(renamed, **parameters)
                with self.assertRaises(ValueError):
                    awq.import_awq_weights(tensors, **{**parameters, "variant": "gemm"})

    def test_gemm_bfloat16_parameters(self):
        tensors, parameters, codes, expected = gemm_fixture()
        tensors["scales"] = tensors["scales"].astype(ml_dtypes.bfloat16)
        encoded = awq.import_awq_weights(tensors, **parameters)
        numpy.testing.assert_array_equal(awq.unpack_awq_codes(encoded), codes)
        numpy.testing.assert_array_equal(awq.reconstruct_awq_weights(encoded), expected)
        exported = awq.export_awq_weights(encoded)["tensors"]["scales"]
        self.assertEqual(exported.dtype, tensors["scales"].dtype)
        self.assertEqual(exported.tobytes(), tensors["scales"].tobytes())

    def test_gemm_exact_words_parameters_and_reconstruction(self):
        tensors, parameters, codes, expected = gemm_fixture()
        # Literal words catch changes in producer order independently of round trips.
        self.assertEqual(int(tensors["qweight"][0, 0].view(numpy.uint32)), 0x5F932C60)
        self.assertEqual(int(tensors["qzeros"][0, 0].view(numpy.uint32)), 0x75316420)
        encoded = awq.import_awq_weights(tensors, **parameters)
        restored = onnx.EncodedValueProto()
        restored.ParseFromString(encoded.SerializeToString())
        numpy.testing.assert_array_equal(awq.unpack_awq_codes(restored), codes)
        numpy.testing.assert_array_equal(awq.reconstruct_awq_weights(restored), expected)
        numpy.testing.assert_array_equal(
            onh.to_array(quantization.dequantize_tensor_proto(restored)), expected
        )
        exported = awq.export_awq_weights(restored)
        for key, value in parameters.items():
            self.assertEqual(exported[key], value)
        for name, value in tensors.items():
            actual = exported["tensors"][name]
            self.assertEqual(actual.dtype, value.dtype)
            self.assertEqual(actual.shape, value.shape)
            self.assertEqual(actual.tobytes(), value.tobytes())
        self.assertEqual(
            awq.import_awq_weights(**exported).SerializeToString(), encoded.SerializeToString()
        )
        # Import owns a snapshot, not a mutable alias of the Python arrays.
        tensors["qweight"].fill(0)
        numpy.testing.assert_array_equal(awq.unpack_awq_codes(encoded), codes)

    def test_rejects_metadata_and_arrays(self):
        tensors, parameters, _, _ = gemm_fixture()
        for key, value in (
            ("variant", "awq"),
            ("variant", "GEMM"),
            ("variant", "marlin"),
            ("bits", 8),
            ("group_axis", 0),
            ("group_axis", -1),
            ("in_features", 0),
            ("in_features", -128),
            ("out_features", 127),
            ("group_size", 0),
            ("group_size", 31),
            ("upstream_revision", "main"),
            ("max_bytes", 0),
            ("max_bytes", 128 * 128 * 4 - 1),
        ):
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                awq.import_awq_weights(tensors, **{**parameters, key: value})
        for name, value in (
            ("qweight", tensors["qweight"].astype(numpy.int64)),
            ("qweight", tensors["qweight"][:-1].copy()),
            ("qweight", tensors["qweight"].T),
            ("qweight", tensors["qweight"].astype(tensors["qweight"].dtype.newbyteorder("S"))),
            ("qzeros", tensors["qzeros"][:, :-1].copy()),
            ("scales", tensors["scales"].astype(numpy.float32)),
            ("scales", numpy.full_like(tensors["scales"], numpy.inf)),
            ("scales", numpy.full_like(tensors["scales"], numpy.nan)),
            ("bias", numpy.zeros(128, dtype=numpy.float16)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                awq.import_awq_weights({**tensors, name: value}, **parameters)
        for name in tensors:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                awq.import_awq_weights(
                    {key: value for key, value in tensors.items() if key != name}, **parameters
                )

    def test_rejects_payload_and_schema_mutations(self):
        tensors, parameters, _, _ = gemm_fixture()
        encoded = awq.import_awq_weights(tensors, **parameters)
        for payload in (encoded.raw_data[:-1], encoded.raw_data + b"\0"):
            altered = onnx.EncodedValueProto()
            altered.ParseFromString(encoded.SerializeToString())
            altered.raw_data = payload
            for consumer in (
                awq.export_awq_weights,
                awq.unpack_awq_codes,
                awq.reconstruct_awq_weights,
            ):
                with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                    consumer(altered)
        altered = onnx.EncodedValueProto()
        altered.ParseFromString(encoded.SerializeToString())
        altered.struct_type.name = "onnx_light.awq_weights.v999"
        with self.assertRaises(ValueError):
            awq.reconstruct_awq_weights(altered)


if __name__ == "__main__":
    unittest.main()
