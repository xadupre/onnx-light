# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks the native GPTQModel c1976918 EXL3 contract, not the affine exl3 profile."""

import unittest

import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

exl3 = import_or_skip("onnx_light.onnx_core.exl3")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


def reference_fixture(
    bits, book, shape=(128, 256), native_half=False, dense=False, exhaustive=False
):
    """Returns native words, cyclic states and source-derived reference weights.

    The pinned upstream mlx_exl3.py packer writes big-endian symbol strings as
    little-endian uint32 words. This intentionally uses strings, not the native
    decoder's two-word sliding window. Codebook formulas are from the pinned
    quantization/mlx_exl3.py; tile ordering is from nn_modules/qlinear/mlx_exl3.py.
    """
    rows, columns = shape
    payload = bytearray()
    states = []
    for tile in range(rows * columns // 256):
        if exhaustive:
            symbols = [
                symbol
                for low in range((tile % 2) * 128, (tile % 2 + 1) * 128)
                for symbol in (tile // 2, low)
            ]
        else:
            symbols = [((i * 37 + tile * 13) ^ (i >> 2)) % (1 << bits) for i in range(256)]
        stream = "".join(f"{symbol:0{bits}b}" for symbol in symbols)
        payload.extend(
            b"".join(
                int(stream[i : i + 32], 2).to_bytes(4, "little")
                for i in range(0, len(stream), 32)
            )
        )
        cyclic = stream[-16:] + stream
        states.extend(int(cyclic[(i + 1) * bits : (i + 1) * bits + 16], 2) for i in range(256))
    states = numpy.array(states, dtype=numpy.uint64)
    if book < 2:
        raw = (states * 89226354 + 64248484 if book == 0 else states * 0xCBAC1FED) & 0xFFFFFFFF
        raw = 0x3B603B60 ^ (raw & 0x8FFF8FFF)
        low = (raw & 65535).astype(numpy.uint16).view(numpy.float16)
        high = (raw >> 16).astype(numpy.uint16).view(numpy.float16)
        decoded = (low + high).astype(numpy.float32)
    else:
        raw = (states * 0x83DCD12D) & 0xFFFFFFFF
        byte_sum = sum((raw >> shift) & 255 for shift in (0, 8, 16, 24))
        accumulator = (byte_sum + 0x6400).astype(numpy.uint16).view(numpy.float16)
        factor, bias = numpy.array([0x1EEE, 0xC931], dtype=numpy.uint16).view(numpy.float16)
        # Exact binary16 operands, one fused binary16 rounding.
        decoded = (
            (accumulator.astype(numpy.float64) * float(factor) + float(bias))
            .astype(numpy.float16)
            .astype(numpy.float32)
        )
    matrix = numpy.empty(shape, dtype=numpy.float32)
    for tile, values in enumerate(decoded.reshape(-1, 256)):
        for lane in range(32):
            for item in range(8):
                row = (
                    tile // (columns // 16) * 16
                    + (lane % 4) * 2
                    + (item % 2)
                    + (item // 2 % 2) * 8
                )
                col = tile % (columns // 16) * 16 + lane // 4 + (item // 4) * 8
                matrix[row, col] = values[lane * 8 + item]
    tensors = {
        "trellis": numpy.frombuffer(payload, dtype=numpy.int16)
        .reshape(rows // 16, columns // 16, 16 * bits)
        .copy(),
        "suh": numpy.resize(numpy.array([-0.3, 0.7, 1.125, -2.25], dtype=numpy.float16), rows),
        "svh": numpy.resize(numpy.array([2.125, -1.3, 0.55], dtype=numpy.float16), columns),
    }
    if book:
        tensors["mcg" if book == 1 else "mul1"] = numpy.array(
            0xCBAC1FED if book == 1 else 0x83DCD12D, dtype=numpy.uint32
        ).view(numpy.int32)

    def transform(value, axis):
        """Applies the pinned ascending-stride normalized FP32 butterflies."""
        value = numpy.moveaxis(value, axis, -1).astype(numpy.float32)
        if dense:
            hadamard = numpy.ones((1, 1), dtype=numpy.float32)
            while hadamard.shape[0] < 128:
                hadamard = numpy.block([[hadamard, hadamard], [hadamard, -hadamard]])
            hadamard *= numpy.float32(0.08838834764831845)
            value = (value.reshape(-1, 128) @ hadamard).reshape(value.shape)
            if native_half:
                value = value.astype(numpy.float16)
            return numpy.moveaxis(value, -1, axis)
        for width in (1, 2, 4, 8, 16, 32, 64):
            blocks = value.reshape(*value.shape[:-1], -1, width * 2)
            a, b = blocks[..., :width].copy(), blocks[..., width:].copy()
            blocks[..., :width] = a + b
            blocks[..., width:] = a - b
        value *= numpy.float32(0.08838834764831845)
        if native_half:
            value = value.astype(numpy.float16)
        return numpy.moveaxis(value, -1, axis)

    expected = transform(matrix, 0) * tensors["suh"][:, None]
    expected = transform(expected, 1) * tensors["svh"]
    return tensors, states, expected


class TestExl3(ExtTestCase):
    def test_exhaustive_codebook_states_in_native_streams(self):
        # Alternating high/low symbols cover every uint16 state, including the
        # signed boundary, through the real packed-stream reconstruction path.
        for book in range(3):
            tensors, states, expected = reference_fixture(
                8, book, shape=(128, 1024), exhaustive=True
            )
            self.assertEqual(len(numpy.unique(states)), 65536)
            encoded = exl3.import_exl3_weights(tensors)
            numpy.testing.assert_array_equal(exl3.reconstruct_exl3_weights(encoded), expected)

    def test_dense_upstream_materialization_reference(self):
        # Native get_weight_tensor uses dense normalized Hadamards rather than
        # butterflies. Reduction order is backend-dependent, not a byte contract.
        for half in (False, True):
            tensors, _, expected = reference_fixture(3, 1, native_half=half, dense=True)
            encoded = exl3.import_exl3_weights(tensors)
            actual = exl3.reconstruct_exl3_weights(
                encoded,
                reconstruction_dtype=onnx.TensorProto.FLOAT16 if half else onnx.TensorProto.FLOAT,
            )
            numpy.testing.assert_allclose(
                actual, expected, rtol=0.002 if half else 0.00002, atol=0.002 if half else 0.00002
            )

    def test_native_reference_rates_codebooks_and_transform_boundaries(self):
        for bits in range(1, 9):
            for book in range(3):
                with self.subTest(bits=bits, book=book):
                    tensors, states, expected = reference_fixture(bits, book)
                    self.assertTrue(numpy.any(states > 32767))
                    encoded = exl3.import_exl3_weights(tensors)
                    restored = onnx.EncodedValueProto()
                    restored.ParseFromString(encoded.SerializeToString())
                    numpy.testing.assert_array_equal(
                        exl3.unpack_exl3_states(restored),
                        states.astype(numpy.uint16).view(numpy.int16).reshape(8, 16, 256),
                    )
                    numpy.testing.assert_array_equal(
                        exl3.reconstruct_exl3_weights(restored), expected
                    )
                    exported = exl3.export_exl3_weights(restored)
                    self.assertEqual(set(exported), set(tensors))
                    for name, value in tensors.items():
                        self.assertEqual(exported[name].dtype, value.dtype)
                        self.assertEqual(exported[name].shape, value.shape)
                        self.assertEqual(exported[name].tobytes(), value.tobytes())
                    self.assertEqual(
                        exl3.import_exl3_weights(exported).SerializeToString(),
                        encoded.SerializeToString(),
                    )
                    _, _, expected_half = reference_fixture(bits, book, native_half=True)
                    numpy.testing.assert_array_equal(
                        exl3.reconstruct_exl3_weights(
                            restored, reconstruction_dtype=onnx.TensorProto.FLOAT16
                        ),
                        expected_half,
                    )

    def test_rejects_malformed_checkpoint_and_limits(self):
        tensors, _, _ = reference_fixture(3, 1)
        for name in ("trellis", "suh", "svh"):
            with self.subTest(missing=name), self.assertRaises(ValueError):
                exl3.import_exl3_weights(
                    {key: value for key, value in tensors.items() if key != name}
                )
        for name, value in (
            ("trellis", tensors["trellis"].astype(numpy.uint16)),
            ("trellis", tensors["trellis"][:, :, :-1]),
            ("trellis", tensors["trellis"][:4]),
            ("trellis", tensors["trellis"][:0]),
            ("suh", tensors["suh"].astype(numpy.float32)),
            ("suh", numpy.full(128, numpy.inf, dtype=numpy.float16)),
            ("svh", numpy.full(256, numpy.nan, dtype=numpy.float16)),
            ("mcg", numpy.array(0, dtype=numpy.int32)),
            ("mcg", tensors["mcg"].reshape(1)),
            ("mul1", numpy.array(0, dtype=numpy.int32)),
            ("bias", numpy.zeros(256, dtype=numpy.float16)),
        ):
            with self.subTest(name=name, shape=value.shape), self.assertRaises(ValueError):
                exl3.import_exl3_weights({**tensors, name: value})
        with self.assertRaisesRegex(ValueError, "revision"):
            exl3.import_exl3_weights(tensors, upstream_revision="unsupported")
        for rate in (0, 9):
            with self.assertRaisesRegex(ValueError, "rate"):
                exl3.import_exl3_weights(
                    {**tensors, "trellis": numpy.zeros((8, 16, 16 * rate), dtype=numpy.int16)}
                )
        for limit in (0, 1, 128 * 256 * 4 - 1):
            with self.assertRaises(ValueError):
                exl3.import_exl3_weights(tensors, max_bytes=limit)
        encoded = exl3.import_exl3_weights(tensors)
        with self.assertRaisesRegex(ValueError, "reconstruction_dtype"):
            exl3.reconstruct_exl3_weights(encoded, reconstruction_dtype=onnx.TensorProto.DOUBLE)
        overflow = {
            **tensors,
            "suh": numpy.full(128, 65504, dtype=numpy.float16),
            "svh": numpy.full(256, 65504, dtype=numpy.float16),
        }
        overflow = exl3.import_exl3_weights(overflow)
        self.assertTrue(numpy.isfinite(exl3.reconstruct_exl3_weights(overflow)).all())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            exl3.reconstruct_exl3_weights(overflow, reconstruction_dtype=onnx.TensorProto.FLOAT16)
        for consumer in (exl3.export_exl3_weights, exl3.reconstruct_exl3_weights):
            with self.assertRaises(ValueError):
                consumer(encoded, max_bytes=1)
            for payload in (encoded.raw_data[:-1], encoded.raw_data + b"\0"):
                damaged = onnx.EncodedValueProto()
                damaged.ParseFromString(encoded.SerializeToString())
                damaged.raw_data = payload
                with self.assertRaises(ValueError):
                    consumer(damaged)

    def test_rejects_metadata_and_preserves_portable_profile(self):
        tensors, _, _ = reference_fixture(2, 0, (128, 128))
        original = exl3.import_exl3_weights(tensors)
        for index, value in ((0, 0), (0, 9), (1, 3), (2, 64)):
            encoded = onnx.EncodedValueProto()
            encoded.ParseFromString(original.SerializeToString())
            encoded.struct_type.structure.field[index].constant.int64_data[0] = value
            with self.assertRaises(ValueError):
                exl3.reconstruct_exl3_weights(encoded)
        original.struct_type.name = "onnx_light.exl3_weights.v2"
        with self.assertRaisesRegex(ValueError, "version"):
            exl3.export_exl3_weights(original)
        plan = quantization.make_quantization_plan(quantization.QuantizationFormat.EXL3, 16, 8)
        self.assertEqual(plan.run(0).layout.method, quantization.QuantizationMethod.AFFINE)


if __name__ == "__main__":
    unittest.main()
