# source: adapted from https://github.com/onnx/onnx/blob/main/onnx/test/numpy_helper_test.py
import os
import tempfile
import unittest

import numpy as np

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as numpy_helper


class TestNumpyHelper(ExtTestCase):
    def test_to_array_float(self) -> None:
        tensor = oh.make_tensor(
            "x", onnxl.TensorProto.FLOAT, [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(
            arr, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )

    def test_to_array_int32(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.INT32, [3], [1, 2, 3])
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, np.array([1, 2, 3], dtype=np.int32))

    def test_to_array_int64(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.INT64, [2], [10, 20])
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, np.array([10, 20], dtype=np.int64))

    def test_to_array_double(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.DOUBLE, [2], [1.5, 2.5])
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, np.array([1.5, 2.5], dtype=np.float64))

    def test_to_array_bool(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.BOOL, [3], [True, False, True])
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, np.array([True, False, True]))

    def test_to_array_float16(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        tensor = oh.make_tensor("x", onnxl.TensorProto.FLOAT16, [3], data)
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_to_array_string(self) -> None:
        tensor = oh.make_tensor("x", onnxl.TensorProto.STRING, [2], [b"hello", b"world"])
        arr = numpy_helper.to_array(tensor)
        self.assertEqual(list(arr), ["hello", "world"])

    def test_to_array_raw_data(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = oh.make_tensor("x", onnxl.TensorProto.FLOAT, [3], data, raw=True)
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_to_array_undefined_raises(self) -> None:
        tensor = onnxl.TensorProto()
        tensor.data_type = onnxl.TensorProto.UNDEFINED
        with self.assertRaises(TypeError):
            numpy_helper.to_array(tensor)

    def test_from_array_float(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = numpy_helper.from_array(data, name="x")
        self.assertEqual(tensor.name, "x")
        self.assertEqual(tensor.data_type, onnxl.TensorProto.FLOAT)
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_from_array_int64(self) -> None:
        data = np.array([[1, 2], [3, 4]], dtype=np.int64)
        tensor = numpy_helper.from_array(data)
        self.assertEqual(tensor.data_type, onnxl.TensorProto.INT64)
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_from_array_string(self) -> None:
        data = np.array(["hello", "world"], dtype=object)
        tensor = numpy_helper.from_array(data, name="s")
        self.assertEqual(tensor.data_type, onnxl.TensorProto.STRING)
        arr = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(arr, data)

    def test_from_array_bytes(self) -> None:
        data = np.array([b"abc", b"def"], dtype=object)
        tensor = numpy_helper.from_array(data, name="b")
        self.assertEqual(tensor.data_type, onnxl.TensorProto.STRING)

    def test_roundtrip_float(self) -> None:
        data = np.random.rand(3, 4).astype(np.float32)
        tensor = numpy_helper.from_array(data, name="t")
        recovered = numpy_helper.to_array(tensor)
        np.testing.assert_array_almost_equal(recovered, data)

    def test_roundtrip_double(self) -> None:
        data = np.random.rand(5).astype(np.float64)
        tensor = numpy_helper.from_array(data)
        recovered = numpy_helper.to_array(tensor)
        np.testing.assert_array_almost_equal(recovered, data)

    def test_roundtrip_int8(self) -> None:
        data = np.array([-128, 0, 127], dtype=np.int8)
        tensor = numpy_helper.from_array(data)
        recovered = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(recovered, data)

    def test_roundtrip_uint8(self) -> None:
        data = np.array([0, 128, 255], dtype=np.uint8)
        tensor = numpy_helper.from_array(data)
        recovered = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(recovered, data)

    def test_roundtrip_bool(self) -> None:
        data = np.array([True, False, True, False])
        tensor = numpy_helper.from_array(data)
        recovered = numpy_helper.to_array(tensor)
        np.testing.assert_array_equal(recovered, data)

    def test_roundtrip_complex64(self) -> None:
        data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        tensor = numpy_helper.from_array(data)
        recovered = numpy_helper.to_array(tensor)
        np.testing.assert_array_almost_equal(recovered, data)

    def test_roundtrip_complex128(self) -> None:
        data = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        tensor = numpy_helper.from_array(data)
        recovered = numpy_helper.to_array(tensor)
        np.testing.assert_array_almost_equal(recovered, data)

    def test_tobytes_little_endian(self) -> None:
        data = np.array([1, 2, 3], dtype=np.float32)
        b = numpy_helper.tobytes_little_endian(data)
        self.assertIsInstance(b, bytes)
        self.assertEqual(len(b), 12)

    def test_to_list_tensors(self) -> None:
        seq = onnxl.SequenceProto()
        seq.elem_type = onnxl.SequenceProto.TENSOR
        t1 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [2], [1.0, 2.0])
        t2 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [2], [3.0, 4.0])
        seq.tensor_values.extend([t1, t2])
        lst = numpy_helper.to_list(seq)
        self.assertEqual(len(lst), 2)
        np.testing.assert_array_equal(lst[0], np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(lst[1], np.array([3.0, 4.0], dtype=np.float32))

    def test_from_list_tensors(self) -> None:
        lst = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        seq = numpy_helper.from_list(lst, name="myseq")
        self.assertEqual(seq.name, "myseq")
        self.assertEqual(int(seq.elem_type), int(onnxl.SequenceProto.TENSOR))
        self.assertEqual(len(seq.tensor_values), 2)

    def test_from_list_empty_with_dtype(self) -> None:
        seq = numpy_helper.from_list([], dtype=onnxl.SequenceProto.TENSOR)
        self.assertEqual(int(seq.elem_type), int(onnxl.SequenceProto.TENSOR))

    def test_from_list_empty_no_dtype(self) -> None:
        seq = numpy_helper.from_list([])
        self.assertEqual(int(seq.elem_type), int(onnxl.SequenceProto.TENSOR))

    def test_from_list_mismatched_types_raises(self) -> None:
        lst = [np.array([1.0]), "not_a_tensor"]
        with self.assertRaises(TypeError):
            numpy_helper.from_list(lst)

    def test_to_dict_int_keys(self) -> None:
        map_proto = onnxl.MapProto()
        map_proto.key_type = onnxl.TensorProto.INT64
        map_proto.keys.extend([1, 2, 3])
        t1 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [10.0])
        t2 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [20.0])
        t3 = oh.make_tensor("", onnxl.TensorProto.FLOAT, [1], [30.0])
        map_proto.values.elem_type = onnxl.SequenceProto.TENSOR
        map_proto.values.tensor_values.extend([t1, t2, t3])
        d = numpy_helper.to_dict(map_proto)
        self.assertEqual(len(d), 3)

    def test_from_dict_int_keys(self) -> None:
        d = {
            np.int64(1): np.array([10.0], dtype=np.float32),
            np.int64(2): np.array([20.0], dtype=np.float32),
        }
        map_proto = numpy_helper.from_dict(d, name="mymap")
        self.assertEqual(map_proto.name, "mymap")
        self.assertEqual(int(map_proto.key_type), int(onnxl.TensorProto.INT64))

    def test_from_dict_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            numpy_helper.from_dict({})

    def test_to_optional_tensor(self) -> None:
        optional = onnxl.OptionalProto()
        optional.elem_type = onnxl.OptionalProto.TENSOR
        t = oh.make_tensor("", onnxl.TensorProto.FLOAT, [2], [1.0, 2.0])
        optional.tensor_value.CopyFrom(t)
        result = numpy_helper.to_optional(optional)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0], dtype=np.float32))

    def test_to_optional_undefined(self) -> None:
        optional = onnxl.OptionalProto()
        optional.elem_type = onnxl.OptionalProto.UNDEFINED
        result = numpy_helper.to_optional(optional)
        self.assertIsNone(result)

    def test_from_optional_tensor(self) -> None:
        data = np.array([1.0, 2.0], dtype=np.float32)
        opt_proto = numpy_helper.from_optional(data, name="opt")
        self.assertEqual(opt_proto.name, "opt")
        self.assertEqual(int(opt_proto.elem_type), int(onnxl.OptionalProto.TENSOR))

    def test_from_optional_none(self) -> None:
        opt_proto = numpy_helper.from_optional(None, dtype=onnxl.OptionalProto.UNDEFINED)
        self.assertEqual(int(opt_proto.elem_type), int(onnxl.OptionalProto.UNDEFINED))

    def test_from_optional_invalid_dtype_raises(self) -> None:
        with self.assertRaises(TypeError):
            numpy_helper.from_optional(None, dtype=999)

    def test_create_random_int(self) -> None:
        arr = numpy_helper.create_random_int((3, 4), np.int32)
        self.assertEqual(arr.shape, (3, 4))
        self.assertEqual(arr.dtype, np.int32)

    def test_create_random_int_unsupported_dtype_raises(self) -> None:
        with self.assertRaises(TypeError):
            numpy_helper.create_random_int((2, 2), np.float32)

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
            arr = numpy_helper.to_array(tensor, base_dir=tmpdir)
            np.testing.assert_array_almost_equal(arr, data)

    def test_unpack_4bit(self) -> None:
        # Pack two values into one byte: low nibble = 3, high nibble = 7
        packed = np.array([0x73], dtype=np.uint8)
        unpacked = numpy_helper._unpack_4bit(packed, [2])
        np.testing.assert_array_equal(unpacked, [3, 7])

    def test_pack_4bitx2(self) -> None:
        # Pack [3, 7] -> 0x73
        data = np.array([3, 7], dtype=np.uint8)
        packed = numpy_helper._pack_4bitx2(data)
        np.testing.assert_array_equal(packed, [0x73])

    def test_unpack_2bit(self) -> None:
        # Pack four 2-bit values: 0b11001001 = 0xC9
        # bits 0-1: 01 (1), bits 2-3: 10 (2), bits 4-5: 00 (0), bits 6-7: 11 (3)
        packed = np.array([0xC9], dtype=np.uint8)
        unpacked = numpy_helper._unpack_2bit(packed, [4])
        np.testing.assert_array_equal(unpacked, [1, 2, 0, 3])

    def test_pack_2bitx4(self) -> None:
        data = np.array([1, 2, 0, 3], dtype=np.uint8)
        packed = numpy_helper._pack_2bitx4(data)
        expected = np.array([0xC9], dtype=np.uint8)
        np.testing.assert_array_equal(packed, expected)


class TestHelperExtensions(ExtTestCase):
    def test_tensor_dtype_to_storage_tensor_dtype(self) -> None:
        import onnx_light.onnx.helper as helper

        # INT8 stored as INT32
        self.assertEqual(
            helper.tensor_dtype_to_storage_tensor_dtype(onnxl.TensorProto.INT8),
            onnxl.TensorProto.INT32,
        )
        # FLOAT stored as FLOAT
        self.assertEqual(
            helper.tensor_dtype_to_storage_tensor_dtype(onnxl.TensorProto.FLOAT),
            onnxl.TensorProto.FLOAT,
        )
        # INT64 stored as INT64
        self.assertEqual(
            helper.tensor_dtype_to_storage_tensor_dtype(onnxl.TensorProto.INT64),
            onnxl.TensorProto.INT64,
        )
        # BOOL stored as INT32
        self.assertEqual(
            helper.tensor_dtype_to_storage_tensor_dtype(onnxl.TensorProto.BOOL),
            onnxl.TensorProto.INT32,
        )

    def test_tensor_type_map_basic(self) -> None:
        import onnx_light.onnx.helper as helper

        self.assertIn(onnxl.TensorProto.FLOAT, helper.TENSOR_TYPE_MAP)
        self.assertIn(onnxl.TensorProto.INT32, helper.TENSOR_TYPE_MAP)
        self.assertIn(onnxl.TensorProto.INT64, helper.TENSOR_TYPE_MAP)
        self.assertIn(onnxl.TensorProto.DOUBLE, helper.TENSOR_TYPE_MAP)
        self.assertIn(onnxl.TensorProto.STRING, helper.TENSOR_TYPE_MAP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
