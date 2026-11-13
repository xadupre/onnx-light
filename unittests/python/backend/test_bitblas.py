# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks explicit row-packed BitBLAS checkpoints against pinned native readers."""

import json
from pathlib import Path
import unittest

import ml_dtypes
import numpy

import onnx_light.onnx.numpy_helper as onh
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

bitblas = import_or_skip("onnx_light.onnx_core.bitblas")
gptq = import_or_skip("onnx_light.onnx_core.gptq")
awq = import_or_skip("onnx_light.onnx_core.awq")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


def fixtures():
    """Returns independently generated native serialized fixtures."""

    def decode(record):
        return (
            numpy.frombuffer(
                bytes.fromhex(record["hex"]), dtype="<" + numpy.dtype(record["dtype"]).str[1:]
            )
            .astype(record["dtype"])
            .reshape(record["shape"])
        )

    records = json.loads(
        Path(__file__).with_name("data").joinpath("bitblas_native_v1.json").read_text()
    )
    for record in records:
        expected = decode(record.pop("expected"))
        codes = decode(record.pop("codes"))
        record.pop("source_sha256")
        record["tensors"] = {name: decode(value) for name, value in record["tensors"].items()}
        yield record, codes, expected


class TestBitblas(ExtTestCase):
    def test_ordinary_checkpoints_are_not_bitblas_rows(self):
        checkpoint, codes, _ = next(fixtures())
        metadata = {name: val for name, val in checkpoint.items() if name != "tensors"}
        ordinary = gptq.pack_gptq_weights(
            codes,
            numpy.ones((2, 32), dtype=numpy.int16),
            numpy.ascontiguousarray(checkpoint["tensors"]["scales"].T),
            numpy.repeat(numpy.arange(2, dtype=numpy.int32), 32),
            format="gptq_v2",
            in_features=64,
            out_features=32,
            group_size=32,
            bits=2,
        )
        tensors = gptq.export_gptq_weights(ordinary)["tensors"]
        tensors.pop("g_idx")
        with self.assertRaises(ValueError):
            bitblas.import_bitblas_weights(tensors, **metadata)
        tensors = {
            "qweight": numpy.full((64, 4), 0x76543210, dtype=numpy.int32),
            "qzeros": numpy.full((2, 4), 0x33333333, dtype=numpy.int32),
            "scales": numpy.ones((2, 32), dtype=numpy.float16),
        }
        source = awq.import_awq_weights(
            tensors, variant="gemm", in_features=64, out_features=32, group_size=32
        )
        with self.assertRaises(ValueError):
            bitblas.import_bitblas_weights(
                awq.export_awq_weights(source)["tensors"],
                **{**metadata, "method": "awq", "bits": 4},
            )

    def test_native_fixtures(self):
        for checkpoint, codes, expected in fixtures():
            with self.subTest(
                method=checkpoint["method"],
                bits=checkpoint["bits"],
                symmetric=checkpoint["symmetric"],
            ):
                value = bitblas.import_bitblas_weights(**checkpoint)
                numpy.testing.assert_array_equal(bitblas.unpack_bitblas_codes(value), codes)
                numpy.testing.assert_array_equal(
                    bitblas.reconstruct_bitblas_weights(value), expected
                )
                exported = bitblas.export_bitblas_weights(value)
                self.assertEqual(exported.keys(), checkpoint.keys())
                for name, tensor in checkpoint["tensors"].items():
                    actual = exported["tensors"][name]
                    self.assertEqual(actual.dtype, tensor.dtype)
                    self.assertEqual(actual.shape, tensor.shape)
                    self.assertEqual(actual.tobytes(), tensor.tobytes())
                self.assertEqual(
                    bitblas.import_bitblas_weights(**exported).SerializeToString(),
                    value.SerializeToString(),
                )
                metadata = {name: val for name, val in checkpoint.items() if name != "tensors"}
                packed = bitblas.pack_bitblas_weights(
                    codes,
                    checkpoint["tensors"]["scales"],
                    checkpoint["tensors"]["qzeros"],
                    **metadata,
                )
                self.assertEqual(packed.SerializeToString(), value.SerializeToString())
                numpy.testing.assert_array_equal(
                    onh.to_array(quantization.dequantize_tensor_proto(value)), expected
                )
                checkpoint["tensors"]["qweight"].fill(0)
                numpy.testing.assert_array_equal(bitblas.unpack_bitblas_codes(value), codes)

    def test_row_reader_requires_fp16_side_tensors(self):
        checkpoint, _, _ = next(fixtures())
        tensors = checkpoint["tensors"]
        for dtype in (ml_dtypes.bfloat16, numpy.float32):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                bitblas.import_bitblas_weights(
                    **{
                        **checkpoint,
                        "tensors": {**tensors, "scales": tensors["scales"].astype(dtype)},
                    }
                )

    def test_rejects_ambiguous_mislabeled_and_invalid_layouts(self):
        checkpoint, codes, _ = next(fixtures())
        for change in (
            {"layout": "bitblas"},
            {"layout": "marlin"},
            {"layout": "row-packed-v2"},
            {"layout": "cuda-ladder"},
            {"method": "unknown"},
            {"bits": 3},
            {"method": "awq", "bits": 1},
            {"upstream_revision": "unknown"},
            {"in_features": 63},
            {"out_features": 31},
            {"group_size": 128},
            {"in_features": 0},
            {"max_bytes": 1},
            {"group_size": 0},
            {"in_features": 1 << 62, "out_features": 1 << 62},
        ):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                bitblas.import_bitblas_weights(**{**checkpoint, **change})
        tensors = checkpoint["tensors"]
        for change in (
            {"qweight": tensors["qweight"].astype(numpy.int32)},
            {"qweight": tensors["qweight"].reshape(4, -1)},
            {"qweight": tensors["qweight"][:, :-1].copy()},
            {"qweight": tensors["qweight"][:, ::-1]},
            {"scales": tensors["scales"].T.copy()},
            {"scales": tensors["scales"].astype(">f2" if numpy.little_endian else "<f2")},
            {"qzeros": numpy.empty(0, dtype=numpy.int8)},
            {"g_idx": numpy.zeros(64, dtype=numpy.int32)},
        ):
            with self.subTest(change=list(change)), self.assertRaises((ValueError, TypeError)):
                bitblas.import_bitblas_weights(**{**checkpoint, "tensors": {**tensors, **change}})
        for scale in (0, -1, numpy.inf, numpy.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                bitblas.import_bitblas_weights(
                    **{
                        **checkpoint,
                        "tensors": {
                            **tensors,
                            "scales": numpy.full_like(tensors["scales"], scale),
                        },
                    }
                )
        with self.assertRaises(ValueError):
            bitblas.import_bitblas_weights(
                **{
                    **checkpoint,
                    "tensors": {
                        **tensors,
                        "qweight": numpy.zeros_like(tensors["qweight"]),
                        "scales": numpy.full_like(tensors["scales"], numpy.nan),
                    },
                }
            )
        metadata = {name: val for name, val in checkpoint.items() if name != "tensors"}
        codes[0, 0] = 4
        with self.assertRaises(ValueError):
            bitblas.pack_bitblas_weights(codes, tensors["scales"], tensors["qzeros"], **metadata)

    def test_rejects_mutated_encoded_payload(self):
        checkpoint, _, _ = next(fixtures())
        original = bitblas.import_bitblas_weights(**checkpoint)
        for raw in (original.raw_data[:-1], original.raw_data + b"\0"):
            value = type(original)()
            value.CopyFrom(original)
            value.raw_data = raw
            for operation in (
                bitblas.export_bitblas_weights,
                bitblas.unpack_bitblas_codes,
                bitblas.reconstruct_bitblas_weights,
            ):
                with self.subTest(operation=operation), self.assertRaises(ValueError):
                    operation(value)
        for field in ("layout", "method", "gptqmodel_revision"):
            value = type(original)()
            value.CopyFrom(original)
            entry = next(item for item in value.struct_type.structure.field if item.name == field)
            entry.constant.raw_data = b"unknown"
            with self.assertRaises(ValueError):
                bitblas.reconstruct_bitblas_weights(value)
        value = type(original)()
        value.CopyFrom(original)
        value.struct_type.name = "onnx_light.bitblas_weights.v999"
        with self.assertRaises(ValueError):
            bitblas.export_bitblas_weights(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
