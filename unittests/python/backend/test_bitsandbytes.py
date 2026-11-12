# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks native GPTQModel bitsandbytes state against pinned upstream fixtures."""

import json
from pathlib import Path
import unittest

import ml_dtypes
import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.numpy_helper as onh

bitsandbytes = import_or_skip("onnx_light.onnx_core.bitsandbytes")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


def loaded_array(record):
    """Returns an owned native-endian reference array."""
    return (
        numpy.frombuffer(bytes.fromhex(record["hex"]), dtype=numpy.dtype(record["dtype"]))
        .reshape(record["shape"])
        .copy()
    )


def fixtures():
    """Returns real bitsandbytes 0.50.2 CPU outputs, without importing upstream."""
    path = Path(__file__).parent / "data" / "bitsandbytes_native_v1.json"
    records = json.loads(path.read_text(encoding="ascii"))
    return [
        (
            record["name"],
            {k: loaded_array(v) for k, v in record["tensors"].items()},
            loaded_array(record["expected"]),
        )
        for record in records
    ]


def with_state(tensors, state=None, raw=None):
    """Returns a checkpoint with replaced packed JSON bytes."""
    result = dict(tensors)
    if raw is None:
        raw = json.dumps(state).encode("ascii")
    result["weight_quant_state"] = numpy.frombuffer(raw, dtype=numpy.uint8).copy()
    return result


def state_of(tensors):
    """Returns the packed metadata dictionary."""
    return json.loads(tensors["weight_quant_state"].tobytes())


class TestBitsAndBytes(ExtTestCase):
    def test_native_fixtures_exact_codes_state_and_reconstruction(self):
        for name, tensors, expected in fixtures():
            with self.subTest(name=name):
                encoded = bitsandbytes.import_bitsandbytes_weights(tensors)
                restored = onnx.EncodedValueProto()
                restored.ParseFromString(encoded.SerializeToString())
                actual = bitsandbytes.reconstruct_bitsandbytes_weights(restored)
                numpy.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.tobytes(), expected.tobytes())
                numpy.testing.assert_array_equal(
                    onh.to_array(quantization.dequantize_tensor_proto(restored)), expected
                )
                exported = bitsandbytes.export_bitsandbytes_weights(restored)
                self.assertEqual(set(exported), set(tensors))
                for key, tensor in tensors.items():
                    self.assertEqual(exported[key].dtype, tensor.dtype)
                    self.assertEqual(exported[key].shape, tensor.shape)
                    self.assertEqual(exported[key].tobytes(), tensor.tobytes())
                self.assertEqual(
                    encoded.SerializeToString(),
                    bitsandbytes.import_bitsandbytes_weights(exported).SerializeToString(),
                )
                if name == "int8":
                    codes = tensors["weight"]
                else:
                    packed = tensors["weight"].reshape(-1)
                    codes = (
                        numpy.column_stack((packed >> 4, packed & 15))
                        .reshape(-1)[:99]
                        .reshape(3, 33)
                    )
                    expected_codes = set(range(16))
                    if name.startswith("fp4-nested"):
                        expected_codes.remove(8)  # Upstream maps both zero entries to code 0.
                    self.assertEqual(set(codes.reshape(-1)), expected_codes)
                numpy.testing.assert_array_equal(
                    bitsandbytes.unpack_bitsandbytes_codes(restored), codes
                )
                for dtype, numpy_type in (
                    (onnx.TensorProto.FLOAT16, numpy.float16),
                    (onnx.TensorProto.BFLOAT16, ml_dtypes.bfloat16),
                ):
                    numpy.testing.assert_array_equal(
                        bitsandbytes.reconstruct_bitsandbytes_weights(
                            restored, reconstruction_dtype=dtype
                        ),
                        expected.astype(numpy_type),
                    )

    def test_partial_nested_groups_zero_scales_and_saved_dtype(self):
        _, template, _ = fixtures()[1]
        for quant_type in ("nf4", "fp4"):
            table = fixtures()[0 if quant_type == "nf4" else 2][1]["weight_quant_map"]
            for block_size in (32, 64, 128, 256, 512, 1024, 2048, 4096):
                count = block_size * 257 + 1
                state = state_of(template)
                state.update(quant_type=quant_type, blocksize=block_size, shape=[1, count])
                tensors = dict(template)
                tensors["weight_quant_map"] = table
                tensors["weight"] = numpy.full(((count + 1) // 2, 1), 0x0F, dtype=numpy.uint8)
                tensors["weight_absmax"] = numpy.resize(numpy.arange(256, dtype=numpy.uint8), 258)
                tensors["weight_nested_absmax"] = numpy.array([0.0, 0.25], dtype=numpy.float32)
                tensors = with_state(tensors, state)
                encoded = bitsandbytes.import_bitsandbytes_weights(tensors)
                decoded_scales = (
                    tensors["weight_nested_quant_map"][tensors["weight_absmax"]]
                    * numpy.repeat(tensors["weight_nested_absmax"], 256)[:258]
                ) + numpy.float32(state["nested_offset"])
                expected = numpy.resize(numpy.array([table[0], table[15]]), count)
                expected *= numpy.repeat(decoded_scales, block_size)[:count]
                numpy.testing.assert_array_equal(
                    bitsandbytes.reconstruct_bitsandbytes_weights(encoded).reshape(-1), expected
                )
        _, tensors, expected = fixtures()[0]
        for saved_dtype in ("float32", "float16", "bfloat16"):
            state = state_of(tensors)
            state["dtype"] = saved_dtype
            changed = with_state(tensors, state)
            encoded = bitsandbytes.import_bitsandbytes_weights(changed)
            numpy.testing.assert_array_equal(
                bitsandbytes.reconstruct_bitsandbytes_weights(encoded), expected
            )
            self.assertEqual(
                bitsandbytes.export_bitsandbytes_weights(encoded)["weight_quant_state"].tobytes(),
                changed["weight_quant_state"].tobytes(),
            )

    def test_padding_snapshot_and_json_order_are_preserved(self):
        _, tensors, expected = fixtures()[0]
        tensors["weight"][-1, 0] |= 13
        state = state_of(tensors)
        raw = json.dumps(dict(reversed(list(state.items()))), indent=2).encode("ascii")
        tensors = with_state(tensors, raw=raw)
        encoded = bitsandbytes.import_bitsandbytes_weights(tensors)
        before = {name: value.tobytes() for name, value in tensors.items()}
        for value in tensors.values():
            value[...] = 0
        exported = bitsandbytes.export_bitsandbytes_weights(encoded)
        self.assertEqual({name: value.tobytes() for name, value in exported.items()}, before)
        numpy.testing.assert_array_equal(
            bitsandbytes.reconstruct_bitsandbytes_weights(encoded), expected
        )

    def test_rejects_malformed_missing_state_and_side_tensors(self):
        _, tensors, _ = fixtures()[1]
        for name in tensors:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                bitsandbytes.import_bitsandbytes_weights(
                    {k: v for k, v in tensors.items() if k != name}
                )
        for name, value in (
            ("weight", tensors["weight"].astype(numpy.int8)),
            ("weight", tensors["weight"].reshape(1, -1)),
            ("weight", tensors["weight"][:-1]),
            ("weight_absmax", tensors["weight_absmax"].astype(numpy.float32)),
            ("weight_quant_map", numpy.zeros(16, dtype=numpy.float32)),
            ("weight_nested_quant_map", numpy.full(256, numpy.nan, dtype=numpy.float32)),
            ("weight_nested_absmax", numpy.array([-1], dtype=numpy.float32)),
            ("bias", numpy.zeros(3, dtype=numpy.float32)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                bitsandbytes.import_bitsandbytes_weights({**tensors, name: value})
        for field, invalid_values in (
            ("quant_type", ("af4", "int4", 4, None)),
            ("dtype", ("float64", "int8", None)),
            ("blocksize", (0, 16, -32, 32.0, True)),
            ("shape", ([0, 33], [-1, 33], [3], [3, 33, 1], [3, 34], [2**63, 3], [2**62, 16])),
            ("nested_blocksize", (0, 128, 256.0)),
            ("nested_dtype", ("float16", "bfloat16")),
            ("nested_offset", (numpy.nan, numpy.inf, -1, 1e100, "2", None)),
            ("unknown", (0,)),
        ):
            for invalid in invalid_values:
                state = state_of(tensors)
                state[field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    bitsandbytes.import_bitsandbytes_weights(with_state(tensors, state))
        for field in state_of(tensors):
            state = state_of(tensors)
            del state[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                bitsandbytes.import_bitsandbytes_weights(with_state(tensors, state))
        for raw in (
            b"",
            b" " * 4097,
            b"{}",
            b"[]",
            b'{"quant_type":"nf4","quant_type":"fp4"}',
            b'{"blocksize":01}',
            b'{"blocksize":+32}',
            b'{"nested_offset":1.}',
            b'{"nested_offset":1e}',
            b'{"shape":[3,33],}',
            tensors["weight_quant_state"].tobytes() + b"null",
            tensors["weight_quant_state"].tobytes()[:-1],
            b"\xff",
        ):
            with self.subTest(raw=raw[:50]), self.assertRaises(ValueError):
                bitsandbytes.import_bitsandbytes_weights(with_state(tensors, raw=raw))

    def test_limits_schema_validation_and_overflow(self):
        _, tensors, _ = fixtures()[1]
        encoded = bitsandbytes.import_bitsandbytes_weights(tensors)
        for limit in (0, 1, 3 * 33 * 4 - 1, len(encoded.raw_data) - 1):
            with self.assertRaises(ValueError):
                bitsandbytes.import_bitsandbytes_weights(tensors, max_bytes=limit)
            for function in (
                bitsandbytes.export_bitsandbytes_weights,
                bitsandbytes.unpack_bitsandbytes_codes,
                bitsandbytes.reconstruct_bitsandbytes_weights,
            ):
                with self.assertRaises(ValueError):
                    function(encoded, max_bytes=limit)
        with self.assertRaisesRegex(ValueError, "revision"):
            bitsandbytes.import_bitsandbytes_weights(tensors, upstream_revision="future")
        for replacement in (encoded.raw_data[:-1], encoded.raw_data + b"\0"):
            changed = onnx.EncodedValueProto()
            changed.CopyFrom(encoded)
            changed.raw_data = replacement
            with self.assertRaises(ValueError):
                bitsandbytes.export_bitsandbytes_weights(changed)
        for index, value in ((0, 3), (1, 2**62), (3, 16), (4, 128), (5, 2**32 + 1), (6, 4097)):
            changed = onnx.EncodedValueProto()
            changed.CopyFrom(encoded)
            constant = changed.struct_type.structure.field[index].constant
            constant.ClearField("int64_data")
            constant.int64_data.extend([value])
            with self.assertRaises(ValueError):
                bitsandbytes.export_bitsandbytes_weights(changed)
        # A payload cannot override the schema's shape or group metadata.
        changed = onnx.EncodedValueProto()
        changed.CopyFrom(encoded)
        changed.raw_data = encoded.raw_data.replace(b'"shape": [3, 33]', b'"shape": [1, 99]')
        with self.assertRaises(ValueError):
            bitsandbytes.export_bitsandbytes_weights(changed)
        _, int8_tensors, _ = fixtures()[-1]
        int8_tensors["weight"].fill(-128)
        int8_tensors["weight_scb"].fill(numpy.finfo(numpy.float32).max)
        overflow = bitsandbytes.import_bitsandbytes_weights(int8_tensors)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            bitsandbytes.reconstruct_bitsandbytes_weights(overflow)
        _, tensors, _ = fixtures()[1]
        tensors["weight_nested_absmax"].fill(numpy.finfo(numpy.float32).max)
        tensors["weight_nested_quant_map"].fill(numpy.finfo(numpy.float32).max)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            bitsandbytes.import_bitsandbytes_weights(tensors)
        _, tensors, _ = fixtures()[0]
        tensors["weight_absmax"].fill(1e10)
        encoded = bitsandbytes.import_bitsandbytes_weights(tensors)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            bitsandbytes.reconstruct_bitsandbytes_weights(
                encoded, reconstruction_dtype=onnx.TensorProto.FLOAT16
            )
        with self.assertRaisesRegex(ValueError, "reconstruction_dtype"):
            bitsandbytes.reconstruct_bitsandbytes_weights(
                encoded, reconstruction_dtype=onnx.TensorProto.DOUBLE
            )

    def test_import_requires_loaded_contiguous_native_arrays(self):
        _, tensors, _ = fixtures()[0]
        for value in (
            tensors["weight_absmax"][::-1],
            tensors["weight_absmax"].astype(">f4"),
            tensors["weight_absmax"].tolist(),
        ):
            with self.assertRaises(ValueError):
                bitsandbytes.import_bitsandbytes_weights({**tensors, "weight_absmax": value})
        for value in tensors.values():
            value.flags.writeable = False
        encoded = bitsandbytes.import_bitsandbytes_weights(tensors)
        self.assertEqual(
            bitsandbytes.export_bitsandbytes_weights(encoded)["weight"].tobytes(),
            tensors["weight"].tobytes(),
        )


if __name__ == "__main__":
    unittest.main()
