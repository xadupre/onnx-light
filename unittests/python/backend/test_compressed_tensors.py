# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Exercises layouts pinned to compressed-tensors eb058d630f7344099459b6795b1e3eeffdda4b40."""

import unittest

import ml_dtypes
import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.numpy_helper as onh

quantization = import_or_skip("onnx_light.onnx_core.quantization")
compressed_tensors = import_or_skip("onnx_light.onnx_core.compressed_tensors")
CompressedTensorsWeights = compressed_tensors.CompressedTensorsWeights
export_compressed_tensors_weights = compressed_tensors.export_compressed_tensors_weights
import_compressed_tensors_weights = compressed_tensors.import_compressed_tensors_weights


class TestCompressedTensors(ExtTestCase):
    def test_native_integer_packing_validation(self):
        runtime = quantization.runtime
        for bits in range(1, 9):
            codes = numpy.array(
                [i % (1 << bits) - (1 << (bits - 1)) for i in range(77)], dtype=numpy.int8
            ).reshape(7, 11)
            for axis in (0, 1):
                packed = runtime._pack_compressed_tensor_int32(codes.tobytes(), 7, 11, bits, axis)
                self.assertEqual(
                    runtime._unpack_compressed_tensor_int32(packed, 7, 11, bits, axis),
                    codes.tobytes(),
                )
                with self.assertRaises(ValueError):
                    runtime._unpack_compressed_tensor_int32(packed[:-1], 7, 11, bits, axis)
            portable = runtime._pack_quantization_codes(codes.tobytes(), bits)
            self.assertEqual(
                runtime._unpack_quantization_codes(portable, codes.size, bits), codes.tobytes()
            )
        with self.assertRaisesRegex(ValueError, "padding"):
            runtime._unpack_compressed_tensor_int32(b"\x80\0\0\0", 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "padding"):
            runtime._unpack_quantization_codes(b"\x80", 1, 1)
        with self.assertRaisesRegex(ValueError, "range"):
            runtime._pack_compressed_tensor_int32(b"\x02", 1, 1, 2, 1)
        with self.assertRaisesRegex(ValueError, "overflow"):
            runtime._validate_compressed_tensor_affine(b"\x7f", numpy.finfo(numpy.float32).max, 0)

    def integer_fixture(self):
        """Returns a source-derived cross-word asymmetric channel fixture."""
        return CompressedTensorsWeights(
            "pack-quantized",
            dict(num_bits=3, type="int", symmetric=False, strategy="channel"),
            {
                "weight_packed": numpy.array(
                    [[-923089272, 1], [923089271, 0]], dtype=numpy.int32
                ),
                "weight_shape": numpy.array([2, 11], dtype=numpy.int64),
                "weight_scale": numpy.array([[0.5], [2]], dtype=numpy.float32),
                "weight_zero_point": numpy.array([[51]], dtype=numpy.int32),
            },
        )

    def check_roundtrip(self, checkpoint, expected):
        """Checks native decoding and exact checkpoint code/parameter export."""
        encoded = import_compressed_tensors_weights(checkpoint)
        wire = onnx.EncodedValueProto()
        wire.ParseFromString(encoded.SerializeToString())
        numpy.testing.assert_array_equal(
            onh.to_array(quantization.dequantize_tensor_proto(wire)), expected
        )
        exported = export_compressed_tensors_weights(
            wire,
            compression_format=checkpoint.compression_format,
            quantization_args=checkpoint.quantization_args,
        )
        self.assertEqual(exported.upstream_revision, checkpoint.upstream_revision)
        self.assertEqual(exported.quantization_args, checkpoint.quantization_args)
        self.assertEqual(set(exported.tensors), set(checkpoint.tensors))
        for name, original in checkpoint.tensors.items():
            actual = exported.tensors[name]
            self.assertEqual(original.dtype, actual.dtype)
            self.assertEqual(original.shape, actual.shape)
            self.assertEqual(original.tobytes(), actual.tobytes())
        self.assertEqual(import_compressed_tensors_weights(exported).raw_data, encoded.raw_data)
        return encoded

    def test_integer_cross_word_asymmetric_channel(self):
        self.check_roundtrip(
            self.integer_fixture(),
            numpy.array(
                [
                    [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, -1.5, -1, 2],
                    [2, 0, -2, -4, -6, -8, -10, -12, 2, 0, -12],
                ],
                dtype=numpy.float32,
            ),
        )

    def test_all_integer_widths_and_group_boundaries(self):
        for bits in range(1, 9):
            for strategy in ("tensor", "channel", "group"):
                for symmetric in (True, False):
                    with self.subTest(bits=bits, strategy=strategy, symmetric=symmetric):
                        # Independent fixture: 0x55 repeating codes in upstream offset-binary.
                        # K=32 makes exactly `bits` int32 words per row.
                        codes = [
                            (
                                (
                                    0x5555555555555555555555555555555555555555555555555555555555555555
                                    >> (i * bits)
                                )
                                & ((1 << bits) - 1)
                            )
                            - (1 << (bits - 1))
                            for i in range(32)
                        ]
                        args = dict(
                            num_bits=bits, type="int", symmetric=symmetric, strategy=strategy
                        )
                        if strategy == "group":
                            args["group_size"] = 8
                        parameter_shape = (
                            (1,) if strategy == "tensor" else (2, 4 if strategy == "group" else 1)
                        )
                        scales = numpy.arange(
                            1, 1 + numpy.prod(parameter_shape), dtype=numpy.float32
                        ).reshape(parameter_shape)
                        tensors = {
                            "weight_packed": numpy.full((2, bits), 0x55555555, dtype=numpy.int32),
                            "weight_shape": numpy.array([2, 32], dtype=numpy.int64),
                            "weight_scale": scales,
                        }
                        if not symmetric:
                            # Stored zero words of zero decode to the minimum signed code.
                            tensors["weight_zero_point"] = (
                                numpy.array([-(1 << (bits - 1))], dtype=numpy.int8)
                                if strategy == "tensor"
                                else numpy.zeros(
                                    ((2 * bits + 31) // 32, parameter_shape[1]), dtype=numpy.int32
                                )
                            )
                        centered = numpy.tile(codes, (2, 1)).astype(numpy.float32)
                        if not symmetric:
                            centered += 1 << (bits - 1)
                        expanded = (
                            scales
                            if strategy == "tensor"
                            else numpy.repeat(scales, 8 if strategy == "group" else 32, axis=1)
                        )
                        self.check_roundtrip(
                            CompressedTensorsWeights("pack-quantized", args, tensors),
                            centered * expanded,
                        )

    def test_zero_point_row_axis_cross_word(self):
        fixture = self.integer_fixture()
        fixture.quantization_args.update(strategy="group", group_size=1)
        fixture.tensors = {
            "weight_shape": numpy.array([11, 2], dtype=numpy.int64),
            "weight_packed": numpy.full((11, 1), 36, dtype=numpy.int32),
            "weight_scale": numpy.ones((11, 2), dtype=numpy.float32),
            "weight_zero_point": numpy.array(
                [[-923089272, 923089271], [1, 0]], dtype=numpy.int32
            ),
        }
        zeros = numpy.array(
            [[-4, -3, -2, -1, 0, 1, 2, 3, -4, -3, 3], [3, 2, 1, 0, -1, -2, -3, -4, 3, 2, -4]],
            dtype=numpy.float32,
        ).T
        encoded = self.check_roundtrip(fixture, -zeros)
        self.assertEqual(
            import_compressed_tensors_weights(fixture, max_bytes=len(encoded.raw_data)).raw_data,
            encoded.raw_data,
        )

    def mx_fixture(self, bits):
        """Returns source-derived MX weight codes with E8M0 scale one."""
        if bits == 4:
            tensors = {"weight_packed": numpy.full((1, 16), 0xA2, dtype=numpy.uint8)}
            expected = numpy.tile([1, -1], 16).reshape(1, 32)
        elif bits == 6:
            tensors = {
                "weight_packed": numpy.tile(
                    numpy.array([0x48, 0xF8, 0x01], dtype=numpy.uint8), 8
                ).reshape(1, 24)
            }
            expected = numpy.tile([1, -0.125, 7.5, 0], 8).reshape(1, 32)
        else:
            tensors = {
                "weight": numpy.full((1, 32), 0x38, dtype=numpy.uint8).view(
                    ml_dtypes.float8_e4m3fn
                )
            }
            expected = numpy.ones((1, 32))
        tensors["weight_scale"] = numpy.array([[127]], dtype=numpy.uint8)
        return CompressedTensorsWeights(
            f"mxfp{bits}-{'quantized' if bits == 8 else 'pack-quantized'}",
            dict(
                num_bits=bits,
                type="float",
                symmetric=True,
                strategy="group",
                group_size=32,
                scale_dtype="torch.uint8",
                zp_dtype="torch.uint8",
            ),
            tensors,
        ), expected.astype(numpy.float32)

    def test_mx_formats_and_e8m0(self):
        for bits in (4, 6, 8):
            fixture, expected = self.mx_fixture(bits)
            for exponent in (0, 126, 127, 128):
                with self.subTest(bits=bits, exponent=exponent):
                    fixture.tensors["weight_scale"][0, 0] = exponent
                    self.check_roundtrip(
                        fixture, expected * numpy.float32(2.0 ** (exponent - 127))
                    )

    def test_nvfp4_global_division_and_signed_zero(self):
        fixture = self.nv_fixture()
        expected = numpy.array([[0, -0.0] + [0.5, -0.5] * 7], dtype=numpy.float32)
        encoded = self.check_roundtrip(fixture, expected)
        actual = onh.to_array(quantization.dequantize_tensor_proto(encoded))
        numpy.testing.assert_array_equal(numpy.signbit(actual), numpy.signbit(expected))

    def nv_fixture(self):
        """Returns an NVFP4 fixture whose global scale is not one."""
        return CompressedTensorsWeights(
            "nvfp4-pack-quantized",
            dict(
                num_bits=4, type="float", symmetric=True, strategy="tensor_group", group_size=16
            ),
            {
                "weight_packed": numpy.array([[0x80] + [0xA2] * 7], dtype=numpy.uint8),
                "weight_scale": numpy.array([[2]], dtype=ml_dtypes.float8_e4m3fn),
                "weight_global_scale": numpy.array([4], dtype=numpy.float32),
            },
        )

    def test_optional_upstream_decoders(self):
        torch = import_or_skip("torch")
        upstream = import_or_skip("compressed_tensors")
        upstream_quantization = import_or_skip("compressed_tensors.quantization")
        helpers = import_or_skip("compressed_tensors.compressors.pack_quantized.helpers")
        fixture = self.integer_fixture()
        encoded = import_compressed_tensors_weights(fixture)
        exported = export_compressed_tensors_weights(
            encoded,
            compression_format=fixture.compression_format,
            quantization_args=fixture.quantization_args,
        )
        tensors = {key: torch.from_numpy(value) for key, value in exported.tensors.items()}
        codes = helpers.unpack_from_int32(tensors["weight_packed"], 3, torch.Size([2, 11]))
        zeros = helpers.unpack_from_int32(
            tensors["weight_zero_point"], 3, torch.Size([2, 1]), packed_dim=0
        )
        numpy.testing.assert_array_equal(
            ((codes.float() - zeros) * tensors["weight_scale"]).numpy(),
            onh.to_array(quantization.dequantize_tensor_proto(encoded)),
        )
        classes = {
            "mxfp4-pack-quantized": upstream.MXFP4PackedCompressor,
            "mxfp6-pack-quantized": upstream.MXFP6PackedCompressor,
            "mxfp8-quantized": upstream.MXFP8QuantizationCompressor,
            "nvfp4-pack-quantized": upstream.NVFP4PackedCompressor,
        }
        fixtures = [self.mx_fixture(bits)[0] for bits in (4, 6, 8)] + [self.nv_fixture()]
        for fixture in fixtures:
            with self.subTest(format=fixture.compression_format):
                encoded = import_compressed_tensors_weights(fixture)
                exported = export_compressed_tensors_weights(
                    encoded,
                    compression_format=fixture.compression_format,
                    quantization_args=fixture.quantization_args,
                )
                tensors = {}
                for key, value in exported.tensors.items():
                    if value.dtype == numpy.dtype(ml_dtypes.float8_e4m3fn):
                        tensors[key] = torch.from_numpy(value.view(numpy.uint8)).view(
                            torch.float8_e4m3fn
                        )
                    else:
                        tensors[key] = torch.from_numpy(value)
                nvfp4 = fixture.compression_format.startswith("nvfp4")
                args = dict(exported.quantization_args)
                args["scale_dtype"] = torch.float8_e4m3fn if nvfp4 else torch.uint8
                args["zp_dtype"] = args["scale_dtype"]
                scheme = upstream_quantization.QuantizationScheme(
                    targets=["Linear"],
                    format=fixture.compression_format,
                    weights=upstream_quantization.QuantizationArgs(**args),
                )
                actual = classes[fixture.compression_format].decompress(tensors, scheme)["weight"]
                numpy.testing.assert_array_equal(
                    actual.float().numpy(),
                    onh.to_array(quantization.dequantize_tensor_proto(encoded)),
                )

    def test_export_native_affine_and_reject_lossy_parameters(self):
        args = dict(num_bits=4, type="int", symmetric=True, strategy="group", group_size=4)
        plan = quantization.make_quantization_plan(quantization.QuantizationFormat.INT4, 8, 4)
        source = numpy.array([[-8, -1, 0, 7], [7, 0, -1, -8]], dtype=numpy.float32)
        encoded = quantization.quantize_tensor_proto(onh.from_array(source), plan)
        exported = export_compressed_tensors_weights(
            encoded, compression_format="pack-quantized", quantization_args=args
        )
        numpy.testing.assert_array_equal(
            onh.to_array(
                quantization.dequantize_tensor_proto(import_compressed_tensors_weights(exported))
            ),
            source,
        )
        run = plan.run(0)
        block = run.block(0)
        block.scale = 0.1
        run.set_block(0, block)
        plan.set_run(0, run)
        lossy = quantization.quantize_tensor_proto(onh.from_array(source), plan)
        with self.assertRaisesRegex(ValueError, "not be lossless"):
            export_compressed_tensors_weights(
                lossy, compression_format="pack-quantized", quantization_args=args
            )

    def test_rejects_malformed_metadata_and_extents(self):
        changes = [
            ("num_bits", 0),
            ("num_bits", 9),
            ("num_bits", True),
            ("type", "float"),
            ("strategy", "block"),
            ("dynamic", True),
            ("actorder", "group"),
            ("scale_dtype", "float16"),
            ("zp_dtype", "uint8"),
            ("input_activations", {}),
            ("profile", "INT4"),
            ("block_structure", [2, 2]),
            ("group_size", 3),
        ]
        for key, value in changes:
            fixture = self.integer_fixture()
            fixture.quantization_args[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                import_compressed_tensors_weights(fixture)
        for name, value in (
            ("weight_shape", numpy.array([2, 10**12], dtype=numpy.int64)),
            ("weight_shape", numpy.array([-1, 11], dtype=numpy.int64)),
            ("weight_packed", numpy.zeros((2, 1), dtype=numpy.int32)),
            ("weight_packed", numpy.zeros((2, 2), dtype=numpy.uint32)),
            ("weight_scale", numpy.array([[0], [1]], dtype=numpy.float32)),
            ("weight_scale", numpy.array([[numpy.nan], [1]], dtype=numpy.float32)),
            ("weight_zero_point", numpy.array([[51], [0]], dtype=numpy.int32)),
            ("weight_packed", "../../weights.bin"),
        ):
            fixture = self.integer_fixture()
            fixture.tensors[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                import_compressed_tensors_weights(fixture)
        fixture = self.integer_fixture()
        fixture.tensors["weight_packed"][0, 1] = 3
        with self.assertRaisesRegex(ValueError, "padding"):
            import_compressed_tensors_weights(fixture)
        fixture = self.integer_fixture()
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            import_compressed_tensors_weights(fixture, max_bytes=16)
        fixture.upstream_revision = "latest"
        with self.assertRaisesRegex(ValueError, "revision"):
            import_compressed_tensors_weights(fixture)

    def test_rejects_wrong_float_contracts(self):
        for bits in (4, 6, 8):
            for key, value in (
                ("group_size", 16),
                ("symmetric", False),
                ("type", "int"),
                ("scale_dtype", "float32"),
            ):
                fixture, _ = self.mx_fixture(bits)
                fixture.quantization_args[key] = value
                with self.subTest(bits=bits, key=key), self.assertRaises(ValueError):
                    import_compressed_tensors_weights(fixture)
            fixture, _ = self.mx_fixture(bits)
            fixture.tensors["weight_scale"][0, 0] = 255
            with self.assertRaises(ValueError):
                import_compressed_tensors_weights(fixture)
        fixture, _ = self.mx_fixture(8)
        fixture.compression_format = "float-quantized"
        with self.assertRaisesRegex(ValueError, "ordinary FP8"):
            import_compressed_tensors_weights(fixture)
        legacy = quantization.make_quantization_plan(quantization.QuantizationFormat.MXFP6, 32)
        encoded = quantization.quantize_tensor_proto(
            onh.from_array(numpy.ones((1, 32), dtype=numpy.float32)), legacy
        )
        fixture, _ = self.mx_fixture(6)
        with self.assertRaises(ValueError):
            export_compressed_tensors_weights(
                encoded,
                compression_format=fixture.compression_format,
                quantization_args=fixture.quantization_args,
            )

    def test_rejects_partial_groups_and_malformed_exports(self):
        fixture = self.integer_fixture()
        encoded = import_compressed_tensors_weights(fixture)
        fixture.quantization_args.update(strategy="group", group_size=4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            import_compressed_tensors_weights(fixture)
        fixture = self.integer_fixture()
        for raw in (
            encoded.raw_data[:-1],
            encoded.raw_data + b"\0",
            b"\1" + encoded.raw_data[1:],
        ):
            malformed = onnx.EncodedValueProto()
            malformed.CopyFrom(encoded)
            malformed.raw_data = raw
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                export_compressed_tensors_weights(
                    malformed,
                    compression_format=fixture.compression_format,
                    quantization_args=fixture.quantization_args,
                )
        malformed = onnx.EncodedValueProto()
        malformed.CopyFrom(encoded)
        malformed.logical_type.tensor_type.shape.dim[0].dim_value = 10**12
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            export_compressed_tensors_weights(
                malformed,
                compression_format=fixture.compression_format,
                quantization_args=fixture.quantization_args,
            )
        malformed.CopyFrom(encoded)
        malformed.parameter_ref = "unresolved"
        with self.assertRaisesRegex(ValueError, "materialize"):
            export_compressed_tensors_weights(
                malformed,
                compression_format=fixture.compression_format,
                quantization_args=fixture.quantization_args,
            )

    def test_rejects_invalid_nvfp4_scales(self):
        for local, global_scale in ((0, 4), (-1, 4), (numpy.nan, 4), (2, 0), (2, numpy.inf)):
            fixture = self.nv_fixture()
            fixture.tensors["weight_scale"][0, 0] = local
            fixture.tensors["weight_global_scale"][0] = global_scale
            with (
                self.subTest(local=local, global_scale=global_scale),
                self.assertRaises(ValueError),
            ):
                import_compressed_tensors_weights(fixture)

    def test_integer_actual_code_overflow_and_external_envelope(self):
        args = dict(num_bits=8, type="int", symmetric=True, strategy="tensor")
        fixture = CompressedTensorsWeights(
            "pack-quantized",
            args,
            {
                "weight_shape": numpy.array([1, 2], dtype=numpy.int64),
                "weight_packed": numpy.array([[0xFF00]], dtype=numpy.int32),
                "weight_scale": numpy.array([2e36], dtype=numpy.float32),
            },
        )
        self.check_roundtrip(
            fixture,
            numpy.array([[-128, 127]], dtype=numpy.float32) * fixture.tensors["weight_scale"],
        )
        # A zero code must not bypass scale validation or wrongly require that all
        # possible (unused) codes reconstruct finitely.
        fixture.tensors["weight_packed"][0, 0] = 0x8080
        fixture.tensors["weight_scale"][0] = numpy.finfo(numpy.float32).max
        encoded = self.check_roundtrip(fixture, numpy.zeros((1, 2), dtype=numpy.float32))
        fixture.tensors["weight_packed"][0, 0] = 0xFF00
        with self.assertRaisesRegex(ValueError, "overflow"):
            import_compressed_tensors_weights(fixture)
        for external_location in (False, True):
            malformed = onnx.EncodedValueProto()
            malformed.CopyFrom(encoded)
            if external_location:
                malformed.data_location = onnx.TensorProto.EXTERNAL
            else:
                malformed.external_data.append(
                    onnx.StringStringEntryProto(key="location", value="../../weights.bin")
                )
            with (
                self.subTest(external_location=external_location),
                self.assertRaisesRegex(ValueError, "inline storage"),
            ):
                export_compressed_tensors_weights(
                    malformed,
                    compression_format=fixture.compression_format,
                    quantization_args=fixture.quantization_args,
                )


if __name__ == "__main__":
    unittest.main()
