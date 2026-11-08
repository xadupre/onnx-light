# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks exact GGUF payloads against pinned native GGML fixtures."""

import base64
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
import zlib

import numpy

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light import onnx

gguf = import_or_skip("onnx_light.onnx_core.gguf")


class TestGguf(ExtTestCase):
    @staticmethod
    def fixtures():
        """Returns native-produced goldens without requiring upstream at test time."""
        path = Path(__file__).parent / "data" / "gguf_native_v1.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def raw(fixture, field):
        """Returns the losslessly stored little-endian golden bytes."""
        return zlib.decompress(base64.b64decode(fixture[field + "_zlib_base64"], validate=True))

    def test_native_fixtures_and_checkpoint_roundtrip(self):
        fixtures = self.fixtures()
        self.assertEqual(fixtures["gptqmodel_revision"], gguf.UPSTREAM_REVISION)
        self.assertEqual(fixtures["ggml_revision"], gguf.REFERENCE_REVISION)
        self.assertEqual(fixtures["byte_order"], "little")
        self.assertEqual(
            {layout.subtype for layout in gguf.gguf_block_layouts()},
            {fixture["subtype"] for fixture in fixtures["fixtures"]},
        )
        for fixture in fixtures["fixtures"]:
            with self.subTest(subtype=fixture["subtype"]):
                raw = self.raw(fixture, "data")
                payload = gguf.import_gguf_tensor_payload(
                    raw, fixture["shape"], fixture["subtype"]
                )
                self.assertIsInstance(payload, onnx.EncodedValueProto)
                self.assertEqual(payload.raw_data, raw)
                self.assertEqual(
                    payload.struct_type.name,
                    "onnx_light.gguf.ggml-blocks-v1/" + fixture["subtype"],
                )
                restored = onnx.EncodedValueProto()
                restored.ParseFromString(payload.SerializeToString())
                payload = restored
                self.assertEqual(gguf.export_gguf_tensor_payload(payload), raw)
                actual = (
                    gguf.runtime.tensor_to_numpy(gguf.dequantize_gguf_tensor_payload(payload))
                    .view(numpy.float32)
                    .reshape(fixture["shape"])
                )
                expected = (
                    numpy.frombuffer(self.raw(fixture, "float32"), dtype="<f4")
                    .reshape(fixture["shape"])
                    .astype(numpy.float32)
                )
                numpy.testing.assert_array_equal(
                    actual.view(numpy.uint32), expected.view(numpy.uint32)
                )
                generic = gguf.runtime.dequantize_tensor_proto(payload)
                numpy.testing.assert_array_equal(
                    numpy.frombuffer(generic.raw_data, dtype=numpy.float32).view(numpy.uint32),
                    expected.ravel().view(numpy.uint32),
                )
                for field in ("scales", "minima"):
                    expected_parameters = (
                        numpy.frombuffer(self.raw(fixture, field), dtype="<f4")
                        .reshape(4, -1)
                        .astype(numpy.float32)
                    )
                    for block in range(4):
                        parameters = gguf.read_gguf_block_parameters(payload, block)
                        numpy.testing.assert_array_equal(
                            numpy.asarray(getattr(parameters, field), dtype=numpy.float32).view(
                                numpy.uint32
                            ),
                            expected_parameters[block].view(numpy.uint32),
                        )
                checkpoint = gguf.export_gptqmodel_gguf_weights(payload)
                self.assertEqual(checkpoint.qweight.dtype, numpy.uint8)
                self.assertEqual(checkpoint.qweight.shape, (2, 2 * fixture["type_size"]))
                self.assertEqual(checkpoint.shape, fixture["shape"])
                self.assertEqual(checkpoint.subtype, fixture["subtype"])
                self.assertEqual(checkpoint.layout_version, gguf.LAYOUT_VERSION)
                self.assertEqual(checkpoint.upstream_revision, gguf.UPSTREAM_REVISION)
                self.assertEqual(checkpoint.qweight.tobytes(), raw)
                self.assertEqual(gguf.import_gptqmodel_gguf_weights(checkpoint).raw_data, raw)
                strided = numpy.empty((2, 4 * fixture["type_size"]), dtype=numpy.uint8)
                strided[:, ::2] = checkpoint.qweight
                checkpoint.qweight = strided[:, ::2]
                self.assertEqual(gguf.import_gptqmodel_gguf_weights(checkpoint).raw_data, raw)
                with self.assertRaises(ValueError):
                    gguf.read_gguf_block_parameters(payload, 4)
                for malformed in (raw[:-1], raw + b"\0"):
                    with self.assertRaises(ValueError):
                        gguf.import_gguf_tensor_payload(
                            malformed, fixture["shape"], fixture["subtype"]
                        )

    def test_staged_support_rejects_every_other_subtype(self):
        unsupported = [
            "Q1_0",
            "Q1_0_g128",
            "Q2_0",
            "Q4_1",
            "Q5_0",
            "Q5_1",
            "Q8_1",
            "Q2_K",
            "Q3_K",
            "Q8_K",
            "IQ1_S",
            "IQ1_M",
            "IQ2_XXS",
            "IQ2_XS",
            "IQ2_S",
            "IQ3_XXS",
            "IQ3_S",
            "IQ4_NL",
            "IQ4_XS",
            "TQ1_0",
            "TQ2_0",
            "MXFP4",
            "NVFP4",
            "F32",
            "F16",
            "BF16",
            "I8",
            "I16",
            "I32",
            "I64",
            "F64",
            "Q4_0_4_4",
            "q4_0",
            "q4_k_m",
            "gguf",
            "",
        ]
        for subtype in unsupported:
            with self.subTest(subtype=subtype), self.assertRaises(ValueError):
                gguf.import_gguf_tensor_payload(b"\0" * 18, [32], subtype)

    def test_metadata_size_and_finite_policies(self):
        raw = b"\0" * 18
        for shape in ([], [0], [-32], [1, 31], [1, 33], [2**62, 32], [32, 0]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                gguf.import_gguf_tensor_payload(raw, shape, "Q4_0")
        for keyword, value in (
            ("layout_version", "ggml-blocks-v2"),
            ("upstream_revision", "unknown"),
            ("max_bytes", 0),
            ("max_bytes", 17),
            ("max_bytes", 127),
        ):
            with self.subTest(keyword=keyword, value=value), self.assertRaises(ValueError):
                gguf.import_gguf_tensor_payload(raw, [32], "Q4_0", **{keyword: value})
        payload = gguf.import_gguf_tensor_payload(raw, [32], "Q4_0", max_bytes=128)
        for consumer in (gguf.export_gguf_tensor_payload, gguf.dequantize_gguf_tensor_payload):
            with self.assertRaises(ValueError):
                consumer(payload, max_bytes=127)
            name = payload.struct_type.name
            payload.struct_type.name = "onnx_light.gguf.ggml-blocks-v2/Q4_0"
            with self.assertRaises(ValueError):
                consumer(payload)
            payload.struct_type.name = name
        for layout in gguf.gguf_block_layouts():
            offsets = [208] if layout.subtype == "Q6_K" else [0]
            if layout.subtype in ("Q4_K", "Q5_K"):
                offsets.append(2)
            for offset in offsets:
                for bits in (0x7C00, 0xFC00, 0x7E00):
                    bad = bytearray(layout.type_size)
                    bad[offset : offset + 2] = bits.to_bytes(2, "little")
                    with (
                        self.subTest(subtype=layout.subtype, offset=offset, bits=bits),
                        self.assertRaises(ValueError),
                    ):
                        gguf.import_gguf_tensor_payload(
                            bytes(bad), [layout.block_size], layout.subtype
                        )
            zero = gguf.import_gguf_tensor_payload(
                b"\0" * layout.type_size, [layout.block_size], layout.subtype
            )
            values = gguf.runtime.tensor_to_numpy(gguf.dequantize_gguf_tensor_payload(zero)).view(
                numpy.float32
            )
            numpy.testing.assert_array_equal(values, numpy.zeros(layout.block_size))
        payload.raw_data = b"\0\x7e" + raw[2:]
        for consumer in (
            gguf.export_gguf_tensor_payload,
            gguf.dequantize_gguf_tensor_payload,
            gguf.export_gptqmodel_gguf_weights,
        ):
            with self.assertRaises(ValueError):
                consumer(payload)
        with self.assertRaises(ValueError):
            gguf.read_gguf_block_parameters(payload, 0)

    def test_checkpoint_validation(self):
        qweight = numpy.zeros((1, 18), dtype=numpy.uint8)
        for array, shape in (
            (qweight.astype(numpy.int8), [1, 32]),
            (qweight.astype(numpy.int32), [1, 32]),
            (qweight.tolist(), [1, 32]),
            (qweight.reshape(-1), [1, 32]),
            (numpy.zeros((1, 19), dtype=numpy.uint8), [1, 32]),
            (qweight, [2, 32]),
            (qweight, [1, 31]),
            (qweight, [32]),
        ):
            with self.subTest(shape=shape):
                checkpoint = gguf.GptqModelGgufWeights(array, shape, "Q4_0")
                with self.assertRaises(ValueError):
                    gguf.import_gptqmodel_gguf_weights(checkpoint)
        checkpoint = gguf.GptqModelGgufWeights(qweight, [1, 32], "Q4_0")
        with self.assertRaises(ValueError):
            gguf.import_gptqmodel_gguf_weights(checkpoint, max_bytes=17)
        checkpoint.upstream_revision = "unknown"
        with self.assertRaises(ValueError):
            gguf.import_gptqmodel_gguf_weights(checkpoint)
        with self.assertRaises(ValueError):
            gguf.export_gptqmodel_gguf_weights(
                gguf.import_gguf_tensor_payload(qweight.tobytes(), [32], "Q4_0")
            )

    def test_checkpoint_subclass_metadata_is_not_trusted(self):
        class MisleadingArray(numpy.ndarray):
            @property
            def shape(self):
                return (2, 18)

            @property
            def nbytes(self):
                return 36

            @property
            def dtype(self):
                return numpy.dtype("uint8")

        storage = numpy.zeros((1, 18), dtype=numpy.uint8)
        subclass = storage.view(MisleadingArray)
        valid = gguf.GptqModelGgufWeights(subclass, [1, 32], "Q4_0")
        self.assertEqual(gguf.import_gptqmodel_gguf_weights(valid).raw_data, storage.tobytes())
        invalid = gguf.GptqModelGgufWeights(subclass, [2, 32], "Q4_0")
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            gguf.import_gptqmodel_gguf_weights(invalid)
        wrong_dtype = numpy.zeros((1, 18), dtype=numpy.int32).view(MisleadingArray)
        invalid.qweight = wrong_dtype
        with self.assertRaisesRegex(ValueError, "UINT8 dtype"):
            gguf.import_gptqmodel_gguf_weights(invalid)

    @unittest.skipUnless(sys.platform == "linux", "Requires Linux address-space accounting.")
    def test_decoded_budget_is_checked_before_contiguous_copy(self):
        code = textwrap.dedent("""
            import os
            import resource
            import unittest
            import numpy
            from onnx_light.onnx_core import gguf

            qweight = numpy.broadcast_to(numpy.zeros(1, dtype=numpy.uint8), (8192, 36864))
            checkpoint = gguf.GptqModelGgufWeights(qweight, [8192, 65536], "Q4_K")
            with open("/proc/self/statm") as stream:
                virtual_size = int(stream.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (virtual_size + 16 * 2**20, hard))
            with unittest.TestCase().assertRaisesRegex(ValueError, "decoded byte limit"):
                gguf.import_gptqmodel_gguf_weights(checkpoint)
            """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
