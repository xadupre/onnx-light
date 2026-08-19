# source: adapted from https://github.com/onnx/onnx/blob/main/onnx/test/numpy_helper_test.py
import os
import tempfile
import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_proto._numpy_helper import (
    _unpack_2bit,
    _unpack_4bit,
    _pack_4bitx2,
    _pack_2bitx4,
    to_float8e8m0,
    tobytes_little_endian,
)
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh


class TestNumpyHelper(ExtTestCase):
    def test_to_array_float(self) -> None:
        tensor = oh.make_tensor(
            "x", onnxl.TensorProto.FLOAT, [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(
            arr, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )

    def test_to_array_int32(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.INT32, [3], [1, 2, 3])
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, np.array([1, 2, 3], dtype=np.int32))

    def test_to_array_int64(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.INT64, [2], [10, 20])
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, np.array([10, 20], dtype=np.int64))

    def test_to_array_double(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.DOUBLE, [2], [1.5, 2.5])
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, np.array([1.5, 2.5], dtype=np.float64))

    def test_to_array_bool(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.BOOL, [3], [True, False, True])
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, np.array([True, False, True]))

    def test_to_array_float16(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        tensor = oh.make_tensor("x", onnxl.TensorProto.FLOAT16, [3], data)
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_to_array_string(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.STRING, [2], [b"hello", b"world"])
        arr = onh.to_array(tensor)
        self.assertEqual(list(arr), ["hello", "world"])

    def test_to_array_raw_data(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = oh.make_tensor("x", onnxl.TensorProto.FLOAT, [3], data, raw=True)
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_to_array_undefined_raises(self) -> None:
        tensor = onnxl.TensorProto()
        tensor.data_type = onnxl.TensorProto.UNDEFINED
        with self.assertRaises(TypeError):
            onh.to_array(tensor)

    def test_from_array_float(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = onh.from_array(data, name="x")
        self.assertEqual(tensor.name, "x")
        self.assertEqual(tensor.data_type, onnxl.TensorProto.FLOAT)
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_from_array_int64(self) -> None:
        data = np.array([[1, 2], [3, 4]], dtype=np.int64)
        tensor = onh.from_array(data)
        self.assertEqual(tensor.data_type, onnxl.TensorProto.INT64)
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_from_array_string(self) -> None:
        data = np.array(["hello", "world"], dtype=object)
        tensor = onh.from_array(data, name="s")
        self.assertEqual(tensor.data_type, onnxl.TensorProto.STRING)
        arr = onh.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_from_array_bytes(self) -> None:
        data = np.array([b"abc", b"def"], dtype=object)
        tensor = onh.from_array(data, name="b")
        self.assertEqual(tensor.data_type, onnxl.TensorProto.STRING)

    def test_roundtrip_float(self) -> None:
        data = np.random.rand(3, 4).astype(np.float32)
        tensor = onh.from_array(data, name="t")
        recovered = onh.to_array(tensor)
        np.testing.assert_array_almost_equal(recovered, data)

    def test_roundtrip_double(self) -> None:
        data = np.random.rand(5).astype(np.float64)
        tensor = onh.from_array(data)
        recovered = onh.to_array(tensor)
        np.testing.assert_array_almost_equal(recovered, data)

    def test_roundtrip_float16(self) -> None:
        data = np.random.rand(13, 37).astype(np.float16)
        tensor = onh.from_array(data, name="test")
        self.assertEqual(tensor.name, "test")
        recovered = onh.to_array(tensor)
        np.testing.assert_array_equal(recovered, data)

    def test_roundtrip_int8(self) -> None:
        data = np.array([-128, 0, 127], dtype=np.int8)
        tensor = onh.from_array(data)
        recovered = onh.to_array(tensor)
        np.testing.assert_array_equal(recovered, data)

    def test_roundtrip_uint8(self) -> None:
        data = np.array([0, 128, 255], dtype=np.uint8)
        tensor = onh.from_array(data)
        recovered = onh.to_array(tensor)
        np.testing.assert_array_equal(recovered, data)

    def test_roundtrip_bool(self) -> None:
        data = np.array([True, False, True, False])
        tensor = onh.from_array(data)
        recovered = onh.to_array(tensor)
        np.testing.assert_array_equal(recovered, data)

    def test_roundtrip_complex64(self) -> None:
        data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        tensor = onh.from_array(data)
        recovered = onh.to_array(tensor)
        np.testing.assert_array_almost_equal(recovered, data)

    def test_roundtrip_complex128(self) -> None:
        data = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        tensor = onh.from_array(data)
        recovered = onh.to_array(tensor)
        np.testing.assert_array_almost_equal(recovered, data)

    def test_tobytes_little_endian(self) -> None:
        data = np.array([1, 2, 3], dtype=np.float32)
        b = tobytes_little_endian(data)
        self.assertIsInstance(b, bytes)
        self.assertEqual(len(b), 12)

    def test_to_list_tensors(self) -> None:
        seq = onnxl.SequenceProto()
        seq.elem_type = onnxl.SequenceProto.TENSOR
        t1 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [2], [1.0, 2.0])
        t2 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [2], [3.0, 4.0])
        seq.tensor_values.extend([t1, t2])
        lst = onh.to_list(seq)
        self.assertEqual(len(lst), 2)
        np.testing.assert_array_equal(lst[0], np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(lst[1], np.array([3.0, 4.0], dtype=np.float32))

    def test_from_list_tensors(self) -> None:
        lst = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        seq = onh.from_list(lst, name="myseq")
        self.assertEqual(seq.name, "myseq")
        self.assertEqual(int(seq.elem_type), int(onnxl.SequenceProto.TENSOR))
        self.assertEqual(len(seq.tensor_values), 2)

    def test_from_list_empty_with_dtype(self) -> None:
        seq = onh.from_list([], dtype=onnxl.SequenceProto.TENSOR)
        self.assertEqual(int(seq.elem_type), int(onnxl.SequenceProto.TENSOR))

    def test_from_list_empty_no_dtype(self) -> None:
        seq = onh.from_list([])
        self.assertEqual(int(seq.elem_type), int(onnxl.SequenceProto.TENSOR))

    def test_from_list_mismatched_types_raises(self) -> None:
        lst = [np.array([1.0]), "not_a_tensor"]
        with self.assertRaises(TypeError):
            onh.from_list(lst)

    def test_to_dict_int_keys(self) -> None:
        map_proto = onnxl.MapProto()
        map_proto.key_type = onnxl.TensorProto.INT64
        map_proto.keys.extend([1, 2, 3])
        t1 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [10.0])
        t2 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [20.0])
        t3 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [30.0])
        map_proto.values.elem_type = onnxl.SequenceProto.TENSOR
        map_proto.values.tensor_values.extend([t1, t2, t3])
        d = onh.to_dict(map_proto)
        self.assertEqual(len(d), 3)

    def test_from_dict_int_keys(self) -> None:
        d = {
            np.int64(1): np.array([10.0], dtype=np.float32),
            np.int64(2): np.array([20.0], dtype=np.float32),
        }
        map_proto = onh.from_dict(d, name="mymap")
        self.assertEqual(map_proto.name, "mymap")
        self.assertEqual(int(map_proto.key_type), int(onnxl.TensorProto.INT64))

    def test_from_dict_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            onh.from_dict({})

    def test_to_optional_tensor(self) -> None:
        optional = onnxl.OptionalProto()
        optional.elem_type = onnxl.OptionalProto.TENSOR
        t = oh.make_tensor("", onnxl.TensorProto.FLOAT, [2], [1.0, 2.0])
        optional.tensor_value.CopyFrom(t)
        result = onh.to_optional(optional)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0], dtype=np.float32))

    def test_to_optional_undefined(self) -> None:
        optional = onnxl.OptionalProto()
        optional.elem_type = onnxl.OptionalProto.UNDEFINED
        result = onh.to_optional(optional)
        self.assertIsNone(result)

    def test_from_optional_tensor(self) -> None:
        data = np.array([1.0, 2.0], dtype=np.float32)
        opt_proto = onh.from_optional(data, name="opt")
        self.assertEqual(opt_proto.name, "opt")
        self.assertEqual(int(opt_proto.elem_type), int(onnxl.OptionalProto.TENSOR))

    def test_from_optional_none(self) -> None:
        opt_proto = onh.from_optional(None, dtype=onnxl.OptionalProto.UNDEFINED)
        self.assertEqual(int(opt_proto.elem_type), int(onnxl.OptionalProto.UNDEFINED))

    def test_from_optional_invalid_dtype_raises(self) -> None:
        with self.assertRaises(TypeError):
            onh.from_optional(None, dtype=999)

    def test_create_random_int(self) -> None:
        arr = onh.create_random_int((3, 4), np.int32)
        self.assertEqual(arr.shape, (3, 4))
        self.assertEqual(arr.dtype, np.int32)

    def test_create_random_int_unsupported_dtype_raises(self) -> None:
        with self.assertRaises(TypeError):
            onh.create_random_int((2, 2), np.float32)

    def test_external_data_roundtrip(self) -> None:
        data = np.random.rand(10).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write raw data to an external file
            ext_file = os.path.join(tmpdir, "tensor_data.bin")
            with open(ext_file, "wb") as f:
                f.write(data.tobytes())

            # Build a tensor that references the external file
            tensor = onnxl.TensorProto()
            tensor.data_type = onnxl.TensorProto.FLOAT
            tensor.dims.extend([10])
            tensor.name = "ext_tensor"
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            entry = tensor.external_data.add()
            entry.key = "location"
            entry.value = "tensor_data.bin"

            # Load and verify
            arr = onh.to_array(tensor, base_dir=tmpdir)
            np.testing.assert_array_almost_equal(arr, data)

    def test_load_external_data_method(self) -> None:
        data = np.arange(6, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_file = os.path.join(tmpdir, "weights.bin")
            with open(ext_file, "wb") as f:
                f.write(b"PAD")  # 3-byte offset
                f.write(data.tobytes())

            tensor = onnxl.TensorProto()
            tensor.name = "w"
            tensor.data_type = onnxl.TensorProto.FLOAT
            tensor.dims.extend([6])
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            for k, v in (
                ("location", "weights.bin"),
                ("offset", "3"),
                ("length", str(data.nbytes)),
            ):
                e = tensor.external_data.add()
                e.key = k
                e.value = v

            tensor.load_external_data(tmpdir)

            self.assertEqual(len(tensor.raw_data), data.nbytes)
            np.testing.assert_array_equal(
                np.frombuffer(bytes(tensor.raw_data), dtype=np.float32), data
            )
            # external_data and data_location must be preserved.
            self.assertEqual(int(tensor.data_location), int(onnxl.TensorProto.EXTERNAL))
            self.assertEqual(len(tensor.external_data), 3)
            self.assertEqual(str(tensor.external_data[0].key), "location")
            self.assertEqual(str(tensor.external_data[0].value), "weights.bin")

    def test_load_external_data_method_default(self) -> None:
        # No offset / length: full file is loaded.
        data = np.arange(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_file = os.path.join(tmpdir, "w.bin")
            with open(ext_file, "wb") as f:
                f.write(data.tobytes())

            tensor = onnxl.TensorProto()
            tensor.data_type = onnxl.TensorProto.FLOAT
            tensor.dims.extend([4])
            tensor.data_location = onnxl.TensorProto.EXTERNAL
            e = tensor.external_data.add()
            e.key = "location"
            e.value = "w.bin"

            tensor.load_external_data(tmpdir)
            np.testing.assert_array_equal(
                np.frombuffer(bytes(tensor.raw_data), dtype=np.float32), data
            )

    def test_load_external_data_method_requires_external(self) -> None:
        tensor = onnxl.TensorProto()
        tensor.data_type = onnxl.TensorProto.FLOAT
        with self.assertRaises(RuntimeError):
            tensor.load_external_data("")

    def test_unpack_4bit(self) -> None:
        # Pack two values into one byte: low nibble = 3, high nibble = 7
        packed = np.array([0x73], dtype=np.uint8)
        unpacked = _unpack_4bit(packed, [2])
        np.testing.assert_array_equal(unpacked, [3, 7])

    def test_pack_4bitx2(self) -> None:
        # Pack [3, 7] -> 0x73
        data = np.array([3, 7], dtype=np.uint8)
        packed = _pack_4bitx2(data)
        np.testing.assert_array_equal(packed, [0x73])

    def test_unpack_2bit(self) -> None:
        # Pack four 2-bit values: 0b11001001 = 0xC9
        # bits 0-1: 01 (1), bits 2-3: 10 (2), bits 4-5: 00 (0), bits 6-7: 11 (3)
        packed = np.array([0xC9], dtype=np.uint8)
        unpacked = _unpack_2bit(packed, [4])
        np.testing.assert_array_equal(unpacked, [1, 2, 0, 3])

    def test_pack_2bitx4(self) -> None:
        data = np.array([1, 2, 0, 3], dtype=np.uint8)
        packed = _pack_2bitx4(data)
        expected = np.array([0xC9], dtype=np.uint8)
        np.testing.assert_array_equal(packed, expected)

    def test_to_array_4bit_payload_too_small_raw_data(self) -> None:
        for name, data_type in [
            ("uint4", onnxl.TensorProto.UINT4),
            ("int4", onnxl.TensorProto.INT4),
            ("float4e2m1", onnxl.TensorProto.FLOAT4E2M1),
        ]:
            with self.subTest(name=name):
                tensor = onnxl.TensorProto()
                tensor.data_type = data_type
                tensor.dims.extend([1000])
                tensor.raw_data = b"\x00"  # encodes 2 elements, not 1000
                with self.assertRaises(ValueError):
                    onh.to_array(tensor)

    def test_to_array_4bit_payload_too_small_int32_data(self) -> None:
        for name, data_type in [
            ("uint4", onnxl.TensorProto.UINT4),
            ("int4", onnxl.TensorProto.INT4),
            ("float4e2m1", onnxl.TensorProto.FLOAT4E2M1),
        ]:
            with self.subTest(name=name):
                tensor = onnxl.TensorProto()
                tensor.data_type = data_type
                tensor.dims.extend([1000])
                tensor.int32_data.append(0)  # encodes 8 elements, not 1000
                with self.assertRaises(ValueError):
                    onh.to_array(tensor)

    def test_to_array_2bit_payload_too_small_raw_data(self) -> None:
        for name, data_type in [
            ("uint2", onnxl.TensorProto.UINT2),
            ("int2", onnxl.TensorProto.INT2),
        ]:
            with self.subTest(name=name):
                tensor = onnxl.TensorProto()
                tensor.data_type = data_type
                tensor.dims.extend([1000])
                tensor.raw_data = b"\x00"  # encodes 4 elements, not 1000
                with self.assertRaises(ValueError):
                    onh.to_array(tensor)

    def test_to_array_2bit_payload_too_small_int32_data(self) -> None:
        for name, data_type in [
            ("uint2", onnxl.TensorProto.UINT2),
            ("int2", onnxl.TensorProto.INT2),
        ]:
            with self.subTest(name=name):
                tensor = onnxl.TensorProto()
                tensor.data_type = data_type
                tensor.dims.extend([1000])
                tensor.int32_data.append(0)  # encodes 16 elements, not 1000
                with self.assertRaises(ValueError):
                    onh.to_array(tensor)


class TestHelperExtensions(ExtTestCase):
    def test_tensor_dtype_to_storage_tensor_dtype(self) -> None:
        # INT8 stored as INT32
        self.assertEqual(
            oh.tensor_dtype_to_storage_tensor_dtype(onnxl.TensorProto.INT8),
            onnxl.TensorProto.INT32,
        )
        # FLOAT stored as FLOAT
        self.assertEqual(
            oh.tensor_dtype_to_storage_tensor_dtype(onnxl.TensorProto.FLOAT),
            onnxl.TensorProto.FLOAT,
        )
        # INT64 stored as INT64
        self.assertEqual(
            oh.tensor_dtype_to_storage_tensor_dtype(onnxl.TensorProto.INT64),
            onnxl.TensorProto.INT64,
        )
        # BOOL stored as INT32
        self.assertEqual(
            oh.tensor_dtype_to_storage_tensor_dtype(onnxl.TensorProto.BOOL),
            onnxl.TensorProto.INT32,
        )

    def test_tensor_type_map_basic(self) -> None:
        import onnx_light.onnx_proto._helper as proto_helper

        self.assertIn(onnxl.TensorProto.FLOAT, proto_helper.TENSOR_TYPE_MAP)
        self.assertIn(onnxl.TensorProto.INT32, proto_helper.TENSOR_TYPE_MAP)
        self.assertIn(onnxl.TensorProto.INT64, proto_helper.TENSOR_TYPE_MAP)
        self.assertIn(onnxl.TensorProto.DOUBLE, proto_helper.TENSOR_TYPE_MAP)
        self.assertIn(onnxl.TensorProto.STRING, proto_helper.TENSOR_TYPE_MAP)

    def test_saturate_cast_float(self) -> None:
        from onnx_light.onnx_proto._numpy_helper import saturate_cast

        finfo = np.finfo(np.float16)
        x = np.array([1e30, -1e30, 1.0, 0.0], dtype=np.float32)
        result = saturate_cast(x, np.float16)
        self.assertEqual(result.dtype, np.float16)
        # Out-of-range values are clamped to the float16 representable range.
        self.assertEqual(float(result[0]), float(finfo.max))
        self.assertEqual(float(result[1]), float(finfo.min))
        self.assertEqual(float(result[2]), 1.0)
        self.assertEqual(float(result[3]), 0.0)

    def test_saturate_cast_integer(self) -> None:
        from onnx_light.onnx_proto._numpy_helper import saturate_cast

        iinfo = np.iinfo(np.int8)
        x = np.array([1000.0, -1000.0, 1.4, 1.6], dtype=np.float32)
        result = saturate_cast(x, np.int8)
        self.assertEqual(result.dtype, np.int8)
        # Out-of-range values are clamped, in-range values are rounded.
        self.assertEqual(int(result[0]), int(iinfo.max))
        self.assertEqual(int(result[1]), int(iinfo.min))
        self.assertEqual(int(result[2]), 1)
        self.assertEqual(int(result[3]), 2)

    def test_to_float8e8m0_powers_of_two(self) -> None:
        x = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        result = to_float8e8m0(x)
        self.assertEqual(str(result.dtype), "float8_e8m0fnu")
        np.testing.assert_array_equal(result.astype(np.float32), x)

    def test_to_float8e8m0_round_up(self) -> None:
        # 3.0 is not a power of two; "up" rounds to the next power of two.
        result = to_float8e8m0(np.array([3.0], dtype=np.float32), round_mode="up")
        np.testing.assert_array_equal(
            result.astype(np.float32), np.array([4.0], dtype=np.float32)
        )

    def test_to_float8e8m0_round_down(self) -> None:
        result = to_float8e8m0(np.array([3.0], dtype=np.float32), round_mode="down")
        np.testing.assert_array_equal(
            result.astype(np.float32), np.array([2.0], dtype=np.float32)
        )

    def test_to_float8e8m0_round_nearest(self) -> None:
        result = to_float8e8m0(np.array([3.0], dtype=np.float32), round_mode="nearest")
        np.testing.assert_array_equal(
            result.astype(np.float32), np.array([4.0], dtype=np.float32)
        )

    def test_to_float8e8m0_invalid_round_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            to_float8e8m0(np.array([1.0], dtype=np.float32), round_mode="invalid")

    def test_to_list_unsupported_raises(self) -> None:
        seq = onnxl.SequenceProto()
        seq.elem_type = onnxl.SequenceProto.UNDEFINED
        with self.assertRaises(TypeError):
            onh.to_list(seq)

    def test_from_list_to_list_roundtrip(self) -> None:
        lst = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        seq = onh.from_list(lst)
        out = onh.to_list(seq)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0], lst[0])
        np.testing.assert_array_equal(out[1], lst[1])

    def test_to_optional_sequence(self) -> None:
        optional = onnxl.OptionalProto()
        optional.elem_type = onnxl.OptionalProto.SEQUENCE
        optional.sequence_value.elem_type = onnxl.SequenceProto.TENSOR
        optional.sequence_value.tensor_values.extend(
            [oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [5.0])]
        )
        result = onh.to_optional(optional)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([5.0], dtype=np.float32))

    def test_to_optional_optional(self) -> None:
        inner = onnxl.OptionalProto()
        inner.elem_type = onnxl.OptionalProto.TENSOR
        inner.tensor_value.CopyFrom(oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [7.0]))
        optional = onnxl.OptionalProto()
        optional.elem_type = onnxl.OptionalProto.OPTIONAL
        optional.optional_value.CopyFrom(inner)
        result = onh.to_optional(optional)
        np.testing.assert_array_equal(result, np.array([7.0], dtype=np.float32))

    def test_to_optional_map(self) -> None:
        optional = onnxl.OptionalProto()
        optional.elem_type = onnxl.OptionalProto.MAP
        optional.map_value.key_type = onnxl.TensorProto.INT64
        optional.map_value.keys.extend([1])
        optional.map_value.values.elem_type = onnxl.SequenceProto.TENSOR
        optional.map_value.values.tensor_values.extend(
            [oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [1.0])]
        )
        result = onh.to_optional(optional)
        self.assertEqual(len(result), 1)

    def test_from_optional_sequence(self) -> None:
        opt_proto = onh.from_optional([np.array([1.0], dtype=np.float32)])
        self.assertEqual(int(opt_proto.elem_type), int(onnxl.OptionalProto.SEQUENCE))

    def test_from_optional_map(self) -> None:
        opt_proto = onh.from_optional({np.int64(1): np.array([1.0], dtype=np.float32)})
        self.assertEqual(int(opt_proto.elem_type), int(onnxl.OptionalProto.MAP))

    def test_from_optional_to_optional_roundtrip(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        opt_proto = onh.from_optional(data)
        result = onh.to_optional(opt_proto)
        np.testing.assert_array_equal(result, data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
