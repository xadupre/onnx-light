# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks the native QQQ weight ABI against the pinned upstream producer."""

import json
import unittest
from pathlib import Path

import numpy

import onnx_light.onnx.numpy_helper as onh
from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

qqq = import_or_skip("onnx_light.onnx_core.qqq")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


def fixtures():
    """Returns native fixture arrays without requiring Torch or GPTQModel."""
    path = Path(__file__).with_name("data") / "qqq_native_v1.json"

    def array(record):
        dtype = numpy.dtype(record["dtype"])
        return (
            numpy.frombuffer(bytes.fromhex(record["hex"]), dtype=dtype.newbyteorder("<"))
            .reshape(record["shape"])
            .astype(dtype, copy=True)
        )

    for record in json.loads(path.read_text(encoding="ascii")):
        yield (
            record["name"],
            {name: array(value) for name, value in record["tensors"].items()},
            {name: record[name] for name in ("in_features", "out_features", "group_size")},
            array(record["codes"]),
            array(record["expected"]),
        )


def restored(value):
    """Returns an independent protobuf wire round trip."""
    result = onnx.EncodedValueProto()
    result.ParseFromString(value.SerializeToString())
    return result


class TestQqq(ExtTestCase):
    def test_upstream_bytes_scales_codes_and_reference(self):
        for name, tensors, parameters, codes, expected in fixtures():
            with self.subTest(name=name):
                encoded = qqq.import_qqq_weights(tensors, **parameters)
                self.assertEqual(encoded.struct_type.name, "onnx_light.qqq_weights.v1")
                value = restored(encoded)
                actual_codes = qqq.unpack_qqq_codes(value)
                self.assertEqual(actual_codes.dtype, numpy.uint8)
                numpy.testing.assert_array_equal(actual_codes, codes)
                actual = qqq.reconstruct_qqq_weights(value)
                self.assertEqual(actual.dtype, numpy.float32)
                # Byte comparison includes signed zero and FP32 subnormal results.
                self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertEqual(
                    onh.to_array(quantization.dequantize_tensor_proto(value)).tobytes(),
                    expected.tobytes(),
                )
                exported = qqq.export_qqq_weights(value)
                for key, parameter in parameters.items():
                    self.assertEqual(exported[key], parameter)
                self.assertEqual(exported["upstream_revision"], qqq.UPSTREAM_REVISION)
                for key, tensor in tensors.items():
                    actual = exported["tensors"][key]
                    self.assertEqual(actual.dtype, tensor.dtype)
                    self.assertEqual(actual.shape, tensor.shape)
                    self.assertEqual(actual.tobytes(), tensor.tobytes())
                packed = qqq.pack_qqq_weights(
                    codes, tensors["s_channel"], tensors["s_group"], **parameters
                )
                self.assertEqual(packed.SerializeToString(), encoded.SerializeToString())
                self.assertEqual(
                    qqq.import_qqq_weights(**exported).SerializeToString(),
                    encoded.SerializeToString(),
                )
                for tensor in tensors.values():
                    tensor.fill(0)
                self.assertEqual(qqq.reconstruct_qqq_weights(value).tobytes(), expected.tobytes())

    def test_group_rounding_is_not_precombined_scale(self):
        _, tensors, parameters, codes, expected = next(
            fixture for fixture in fixtures() if fixture[0] == "k256-n64-g128-extremeTrue"
        )
        # The source fixture deliberately includes ties and saturation, so a plain
        # affine product cannot accidentally satisfy the reference comparison.
        self.assertIn(0, tensors["s_group"])
        self.assertIn(numpy.float16(65504), tensors["s_group"])
        self.assertEqual(set(codes.reshape(-1)), set(range(16)))
        inverse = numpy.argsort([i + 8 * j for i in range(8) for j in range(8)])
        groups = tensors["s_group"].reshape(-1, 64)[:, inverse].reshape(2, 64)
        expanded = numpy.repeat(groups.astype(numpy.float32), 128, axis=0).T
        delta = codes.astype(numpy.float32) - 8
        trap = (expanded == numpy.float32(0.2142333984375)) & (numpy.abs(delta) == 7)
        self.assertTrue(numpy.any(trap))
        products = delta[trap] * expanded[trap]
        self.assertTrue(
            numpy.all(numpy.rint(products.astype(numpy.float16)) != numpy.rint(products))
        )
        inverse_channel = numpy.argsort(
            [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
        )
        channel = tensors["s_channel"].reshape(-1, 32)[:, inverse_channel].reshape(64, 1)
        self.assertTrue(numpy.any(expected != delta * expanded * channel))
        encoded = qqq.import_qqq_weights(tensors, **parameters)
        self.assertEqual(qqq.reconstruct_qqq_weights(encoded).tobytes(), expected.tobytes())

    def test_rejects_unsupported_geometry_revision_and_limits(self):
        _, tensors, parameters, _, _ = next(fixtures())
        for key, value in (
            ("bits", 2),
            ("bits", 8),
            ("in_features", 0),
            ("in_features", -64),
            ("in_features", 65),
            ("out_features", 63),
            ("out_features", 80),
            ("group_size", 0),
            ("group_size", 64),
            ("group_size", 128),
            ("upstream_revision", "main"),
            ("max_bytes", 0),
            ("max_bytes", 64 * 64 * 4 - 1),
            ("in_features", 2**62),
            ("out_features", 2**62),
        ):
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                qqq.import_qqq_weights(tensors, **{**parameters, key: value})
        encoded = qqq.import_qqq_weights(tensors, **parameters)
        for consumer in (
            qqq.export_qqq_weights,
            qqq.unpack_qqq_codes,
            qqq.reconstruct_qqq_weights,
        ):
            with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                consumer(
                    encoded,
                    max_bytes=parameters["in_features"] * parameters["out_features"] * 4 - 1,
                )

    def test_rejects_arrays_and_side_tensors(self):
        _, tensors, parameters, codes, _ = next(fixtures())
        for name, value in (
            ("B", tensors["B"].astype(numpy.int64)),
            ("B", tensors["B"][:-1].copy()),
            ("B", tensors["B"].T),
            ("B", tensors["B"].astype(tensors["B"].dtype.newbyteorder("S"))),
            ("B", tensors["B"].tobytes()),
            ("s_channel", tensors["s_channel"].astype(numpy.float16)),
            ("s_channel", tensors["s_channel"].reshape(-1)),
            ("s_channel", numpy.full_like(tensors["s_channel"], numpy.inf)),
            ("s_channel", numpy.full_like(tensors["s_channel"], numpy.nan)),
            ("s_channel", -tensors["s_channel"]),
            ("s_group", numpy.empty((0, 64), dtype=numpy.float16)),
            ("s_group", numpy.empty(0, dtype=numpy.float32)),
            ("bias", numpy.zeros(64, dtype=numpy.float16)),
            ("qweight", tensors["B"]),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                qqq.import_qqq_weights({**tensors, name: value}, **parameters)
        for name in tensors:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                qqq.import_qqq_weights(
                    {key: value for key, value in tensors.items() if key != name}, **parameters
                )
        invalid_codes = codes.copy()
        invalid_codes[0, 0] = 16
        with self.assertRaises(ValueError):
            qqq.pack_qqq_weights(
                invalid_codes, tensors["s_channel"], tensors["s_group"], **parameters
            )
        with self.assertRaises(ValueError):
            qqq.pack_qqq_weights(
                codes.astype(numpy.int8), tensors["s_channel"], tensors["s_group"], **parameters
            )
        _, grouped, metadata, _, _ = next(
            fixture for fixture in fixtures() if fixture[0] == "k256-n64-g128-extremeFalse"
        )
        for value in (
            numpy.full_like(grouped["s_group"], numpy.inf),
            numpy.full_like(grouped["s_group"], numpy.nan),
            -grouped["s_group"],
            grouped["s_group"][:-1].copy(),
            numpy.empty(0, dtype=numpy.float16),
        ):
            with self.assertRaises(ValueError):
                qqq.import_qqq_weights({**grouped, "s_group": value}, **metadata)
        overflowing = {**tensors, "s_channel": numpy.full_like(tensors["s_channel"], 3e38)}
        with self.assertRaises(ValueError):
            qqq.import_qqq_weights(overflowing, **parameters)

    def test_rejects_payload_schema_and_logical_metadata(self):
        _, tensors, parameters, _, _ = next(fixtures())
        encoded = qqq.import_qqq_weights(tensors, **parameters)
        bad = []
        for payload in (encoded.raw_data[:-1], encoded.raw_data + b"\0"):
            altered = restored(encoded)
            altered.raw_data = payload
            bad.append(altered)
        altered = restored(encoded)
        altered.struct_type.name = "onnx_light.qqq_weights.v999"
        bad.append(altered)
        altered = restored(encoded)
        altered.logical_type.tensor_type.shape.dim[0].dim_value += 64
        bad.append(altered)
        altered = restored(encoded)
        altered.logical_type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
        bad.append(altered)
        altered = restored(encoded)
        altered.struct_type.structure.field[0].constant.int64_data.clear()
        altered.struct_type.structure.field[0].constant.int64_data.append(2**62)
        bad.append(altered)
        for altered in bad:
            for consumer in (
                qqq.export_qqq_weights,
                qqq.unpack_qqq_codes,
                qqq.reconstruct_qqq_weights,
            ):
                with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                    consumer(altered)
        # Portable profiles remain separate; no ordinary GPTQ schema is accepted.
        plan = quantization.make_quantization_plan(quantization.QuantizationFormat.GPTQ, 16, 16)
        portable = quantization.quantize_tensor_proto(
            onh.from_array(numpy.arange(16, dtype=numpy.float32)), plan
        )
        with self.assertRaises(ValueError):
            qqq.export_qqq_weights(portable)


if __name__ == "__main__":
    unittest.main()
