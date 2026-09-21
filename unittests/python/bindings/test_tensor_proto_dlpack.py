import ctypes
import gc
import os
import sys
import tempfile
import unittest

import numpy

from onnx_light.onnx import FileLoadMode, GraphProto, ModelProto, ParseOptions, TensorProto
from onnx_light.onnx.numpy_helper import from_array


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


def capsule_tensor(capsule):
    """Returns the tensor at the start of a legacy DLManagedTensor capsule."""
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.cast(get_pointer(capsule, b"dltensor"), ctypes.POINTER(DLTensor)).contents


class CapsuleProducer:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self, **kwargs):
        return self.capsule

    def __dlpack_device__(self):
        return (1, 0)


class TestTensorProtoDLPack(unittest.TestCase):
    def test_release_owned_and_aligned(self):
        expected = numpy.arange(6, dtype=numpy.uint8).reshape(2, 3)
        for alignment in (0, 64):
            with self.subTest(alignment=alignment):
                options = ParseOptions()
                options.alignment = alignment
                tensor = TensorProto()
                tensor.ParseFromString(from_array(expected).SerializeToString(), options)
                tensor.name = "initializer"
                tensor.doc_string = "preserved"
                graph = GraphProto()
                graph.initializer.add()
                graph.initializer[0] = tensor
                tensor = graph.initializer[0]
                # Standard export remains non-destructive and snapshots metadata.
                before = tensor.SerializeToString()
                original = tensor.__dlpack__()
                self.assertEqual(tensor.SerializeToString(), before)
                address = capsule_tensor(original).data
                if alignment:
                    self.assertEqual(address % alignment, 0)
                del original
                references = sys.getrefcount(tensor)
                capsule = tensor.release_dlpack()
                self.assertEqual(sys.getrefcount(tensor), references)
                self.assertEqual(capsule_tensor(capsule).data, address)
                self.assertFalse(tensor.HasField("raw_data"))
                self.assertEqual(tensor.raw_data, b"")
                self.assertEqual(tensor.name, "initializer")
                self.assertEqual(tensor.doc_string, "preserved")
                self.assertEqual(list(tensor.dims), [2, 3])
                self.assertEqual(tensor.data_type, TensorProto.UINT8)
                tensor.dims.clear()
                tensor.dims.extend([6])
                tensor.data_type = TensorProto.INT8
                tensor.raw_data = b"xxxxxx"
                del tensor, graph
                gc.collect()
                producer = CapsuleProducer(capsule)
                view = numpy.from_dlpack(producer)
                with self.assertRaises(ValueError):
                    numpy.from_dlpack(producer)
                del capsule, producer
                gc.collect()
                self.assertEqual(view.ctypes.data, address)
                numpy.testing.assert_array_equal(view, expected)

    def test_release_shared_owner_cleanup(self):
        tensor, address, released = self.make_borrowed()
        graph = GraphProto()
        graph.initializer.add()
        graph.initializer[0] = tensor
        copied = graph.initializer[0]
        capsule = tensor.release_dlpack()
        second = copied.release_dlpack()
        self.assertEqual(capsule_tensor(capsule).data, address)
        self.assertEqual(capsule_tensor(second).data, address)
        self.assertFalse(tensor.HasField("raw_data"))
        self.assertFalse(copied.HasField("raw_data"))
        del tensor, copied, graph
        gc.collect()
        self.assertEqual(released, [])
        view = numpy.from_dlpack(CapsuleProducer(capsule))
        del capsule
        gc.collect()
        numpy.testing.assert_array_equal(view, numpy.arange(64, dtype=numpy.uint8))
        del view
        gc.collect()
        self.assertEqual(released, [])
        del second
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_release_rejects_active_owned_views(self):
        tensor = from_array(numpy.arange(6, dtype=numpy.uint8))
        view = numpy.from_dlpack(tensor)
        capsule = tensor.__dlpack__()
        before = tensor.SerializeToString()
        with self.assertRaisesRegex(ValueError, "active DLPack exports"):
            tensor.release_dlpack()
        self.assertEqual(tensor.SerializeToString(), before)
        del capsule
        with self.assertRaisesRegex(ValueError, "active DLPack exports"):
            tensor.release_dlpack()
        numpy.testing.assert_array_equal(view, numpy.arange(6, dtype=numpy.uint8))
        del view
        gc.collect()
        transferred = tensor.release_dlpack()
        self.assertFalse(tensor.HasField("raw_data"))
        numpy.testing.assert_array_equal(
            numpy.from_dlpack(CapsuleProducer(transferred)), numpy.arange(6, dtype=numpy.uint8)
        )

    def test_release_preserves_active_borrowed_view(self):
        tensor, _, released = self.make_borrowed()
        view = numpy.from_dlpack(tensor)
        capsule = tensor.release_dlpack()
        del capsule, tensor
        gc.collect()
        self.assertEqual(released, [])
        numpy.testing.assert_array_equal(view, numpy.arange(64, dtype=numpy.uint8))
        del view
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_release_owned_callback_and_unconsumed_capsule(self):
        released = []
        options = ParseOptions()
        options.raw_data_callback = lambda tensor, graph: lambda: released.append(True)
        tensor = TensorProto()
        tensor.ParseFromString(
            TensorProto(
                data_type=TensorProto.UINT8, dims=[1], raw_data=b"\x07"
            ).SerializeToString(),
            options,
        )
        capsule = tensor.release_dlpack()
        del tensor
        gc.collect()
        self.assertEqual(released, [])
        del capsule
        gc.collect()
        self.assertEqual(released, [True])

    def test_release_native_consumer_deleter_without_gil(self):
        tensor, _, released = self.make_borrowed()
        capsule = tensor.release_dlpack()
        del tensor
        gc.collect()
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        pointer = get_pointer(capsule, b"dltensor")
        managed = ctypes.cast(pointer, ctypes.POINTER(DLManagedTensor)).contents
        set_name = ctypes.pythonapi.PyCapsule_SetName
        set_name.restype = ctypes.c_int
        set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
        self.assertEqual(set_name(capsule, b"used_dltensor"), 0)
        self.assertEqual(released, [])
        # CFUNCTYPE releases the GIL, like a native consumer destroying its descriptor.
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(managed.deleter)(pointer)
        self.assertEqual(len(released), 1)
        del capsule
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_release_empty_scalar_and_absent(self):
        for shape in ((), (0,), (2, 0, 3)):
            with self.subTest(shape=shape):
                tensor = TensorProto(
                    data_type=TensorProto.UINT8, dims=shape, raw_data=b"" if shape else b"\x07"
                )
                view = numpy.from_dlpack(CapsuleProducer(tensor.release_dlpack()))
                self.assertFalse(tensor.HasField("raw_data"))
                self.assertEqual(view.shape, shape)
                self.assertEqual(view.size, 0 if shape else 1)
                with self.assertRaisesRegex(ValueError, "raw_data"):
                    tensor.release_dlpack()

    def test_release_failures_leave_source_intact(self):
        invalid = [
            TensorProto(data_type=TensorProto.STRING, dims=[1], raw_data=b"x"),
            TensorProto(data_type=TensorProto.UINT8, dims=[0]),
            TensorProto(data_type=TensorProto.FLOAT, dims=[1], float_data=[1.0]),
        ]
        for shape, raw in (
            ([-1], b""),
            ([0, -1], b""),
            ([2**62, 4], b""),
            ([1] * 129, b"\0"),
            ([2], b"\0"),
            ([0], b"\0"),
            ([], b""),
        ):
            invalid.append(TensorProto(data_type=TensorProto.UINT8, dims=shape, raw_data=raw))
        for tensor in invalid:
            before = tensor.SerializeToString()
            with self.assertRaises(ValueError):
                tensor.release_dlpack()
            self.assertEqual(tensor.SerializeToString(), before)
        unaligned, _, released = self.make_borrowed(TensorProto.FLOAT, 4, 1)
        before = unaligned.SerializeToString()
        with self.assertRaises(BufferError):
            unaligned.release_dlpack()
        self.assertEqual(unaligned.SerializeToString(), before)
        self.assertEqual(released, [])
        del unaligned
        gc.collect()
        self.assertEqual(len(released), 1)

    def make_borrowed(self, data_type=TensorProto.UINT8, alignment=1, remainder=0):
        """Returns a borrowed proto, its backing address, and a release counter."""
        payload = bytes(range(64))
        bits = 32 if data_type == TensorProto.FLOAT else 8
        tensor = TensorProto(data_type=data_type, dims=[64 * 8 // bits], raw_data=payload)
        get_pointer = ctypes.pythonapi.PyBytes_AsString
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object]
        for padding in range(1, alignment + 2):
            tensor.name = "x" * padding
            serialized = tensor.SerializeToString()
            address = get_pointer(serialized) + serialized.index(payload)
            if address % alignment == remainder:
                break
        self.assertEqual(address % alignment, remainder)
        released = []

        def callback(parsed, graph):
            def release(backing=serialized):
                released.append(len(backing))

            return release

        options = ParseOptions()
        options.no_copy = True
        options.raw_data_callback = callback
        parsed = TensorProto()
        parsed.ParseFromString(serialized, options)
        return parsed, address, released

    @unittest.skipUnless(sys.byteorder == "little", "Requires native little-endian storage.")
    def test_numpy_zero_copy_and_lifetime(self):
        for dtype in (
            numpy.float16,
            numpy.float32,
            numpy.float64,
            numpy.int8,
            numpy.uint8,
            numpy.int16,
            numpy.uint16,
            numpy.int32,
            numpy.uint32,
            numpy.int64,
            numpy.uint64,
            numpy.bool_,
            numpy.complex64,
            numpy.complex128,
        ):
            with self.subTest(dtype=dtype):
                expected = numpy.arange(6).astype(dtype).reshape(2, 3)
                tensor = from_array(expected)
                self.assertEqual(tensor.__dlpack_device__(), (1, 0))
                view = numpy.from_dlpack(tensor)
                other = numpy.from_dlpack(tensor)
                self.assertTrue(numpy.shares_memory(view, other))
                self.assertFalse(view.flags.writeable)
                self.assertEqual(view.dtype, expected.dtype)
                # A same-sized assignment reuses the proto's owned raw_data buffer.
                tensor.raw_data = numpy.zeros_like(expected).tobytes()
                numpy.testing.assert_array_equal(view, numpy.zeros_like(expected))
                del tensor, other
                gc.collect()
                numpy.testing.assert_array_equal(view, numpy.zeros_like(expected))

    def test_exact_descriptors(self):
        descriptors = {
            "FLOAT": (2, 32),
            "DOUBLE": (2, 64),
            "FLOAT16": (2, 16),
            "BFLOAT16": (4, 16),
            "INT8": (0, 8),
            "INT16": (0, 16),
            "INT32": (0, 32),
            "INT64": (0, 64),
            "UINT8": (1, 8),
            "UINT16": (1, 16),
            "UINT32": (1, 32),
            "UINT64": (1, 64),
            "BOOL": (6, 8),
            "COMPLEX64": (5, 64),
            "COMPLEX128": (5, 128),
            "FLOAT8E4M3FN": (10, 8),
            "FLOAT8E4M3FNUZ": (11, 8),
            "FLOAT8E5M2": (12, 8),
            "FLOAT8E5M2FNUZ": (13, 8),
            "FLOAT8E8M0": (14, 8),
        }
        for name, (code, bits) in descriptors.items():
            with self.subTest(name=name):
                tensor = TensorProto(
                    data_type=getattr(TensorProto, name), raw_data=bytes(bits // 8)
                )
                if sys.byteorder != "little" and bits > 8:
                    with self.assertRaisesRegex(BufferError, "endian"):
                        tensor.__dlpack__()
                    continue
                capsule = tensor.__dlpack__()
                exported = capsule_tensor(capsule)
                self.assertEqual((exported.dtype.code, exported.dtype.bits), (code, bits))
                self.assertEqual(exported.dtype.lanes, 1)
                self.assertEqual((exported.device.device_type, exported.device.device_id), (1, 0))
                self.assertEqual(exported.ndim, 0)
                self.assertEqual(exported.byte_offset, 0)
                address = exported.data
                del exported, capsule
                transferred = tensor.release_dlpack()
                released = capsule_tensor(transferred)
                self.assertEqual(
                    (released.dtype.code, released.dtype.bits, released.dtype.lanes),
                    (code, bits, 1),
                )
                self.assertEqual(released.data, address)
                self.assertFalse(tensor.HasField("raw_data"))

    def test_borrowed_pointer_copies_and_release(self):
        tensor, address, released = self.make_borrowed()
        # Indexed assignment uses the C++ copy constructor, preserving the token.
        # CopyFrom/append round-trip through serialization and own new bytes.
        graph = GraphProto()
        graph.initializer.add()
        graph.initializer[0] = tensor
        copied = graph.initializer[0]
        self.assertIsNot(copied, tensor)
        capsule = copied.__dlpack__()
        self.assertEqual(capsule_tensor(capsule).data, address)
        view = numpy.from_dlpack(tensor)
        self.assertEqual(view.ctypes.data, address)
        # The exported token must survive replacement of both borrowed spans.
        tensor.Clear()
        copied.Clear()
        del tensor, copied, graph
        gc.collect()
        self.assertEqual(released, [])
        numpy.testing.assert_array_equal(view, numpy.arange(64, dtype=numpy.uint8))
        del view
        gc.collect()
        self.assertEqual(released, [])
        del capsule
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_capsule_consumed_once(self):
        tensor, address, released = self.make_borrowed()
        producer = CapsuleProducer(tensor.__dlpack__())
        view = numpy.from_dlpack(producer)
        self.assertEqual(view.ctypes.data, address)
        with self.assertRaises(ValueError):
            numpy.from_dlpack(producer)
        del tensor, producer
        gc.collect()
        self.assertEqual(released, [])
        del view
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_unconsumed_capsule_release(self):
        tensor, _, released = self.make_borrowed()
        capsule = tensor.__dlpack__()
        del tensor
        gc.collect()
        self.assertEqual(released, [])
        del capsule
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_owned_source_retained_until_final_release(self):
        released = []
        options = ParseOptions()
        options.raw_data_callback = lambda tensor, graph: lambda: released.append(True)
        tensor = TensorProto()
        tensor.ParseFromString(
            TensorProto(
                data_type=TensorProto.UINT8, dims=[1], raw_data=b"\x07"
            ).SerializeToString(),
            options,
        )
        references = sys.getrefcount(tensor)
        view = numpy.from_dlpack(tensor)
        self.assertGreater(sys.getrefcount(tensor), references)
        del tensor
        gc.collect()
        self.assertEqual(released, [])
        self.assertEqual(view.tolist(), [7])
        del view
        gc.collect()
        self.assertEqual(released, [True])

    def test_scalar_and_empty(self):
        for shape in ((), (0,), (2, 0, 3)):
            with self.subTest(shape=shape):
                tensor = TensorProto(
                    data_type=TensorProto.UINT8, dims=shape, raw_data=b"" if shape else b"\x07"
                )
                view = numpy.from_dlpack(tensor)
                self.assertEqual(view.shape, shape)
                self.assertEqual(view.size, 0 if shape else 1)
        missing = TensorProto(data_type=TensorProto.FLOAT, dims=[0])
        with self.assertRaisesRegex(ValueError, "raw_data"):
            missing.__dlpack__()

    def test_unsupported_types(self):
        for name in (
            "UNDEFINED",
            "STRING",
            "UINT4",
            "INT4",
            "UINT2",
            "INT2",
            "FLOAT4E2M1",
            "FLOAT6E2M3",
            "FLOAT6E3M2",
        ):
            with self.subTest(name=name):
                tensor = TensorProto(
                    data_type=getattr(TensorProto, name), dims=[1], raw_data=b"\0"
                )
                with self.assertRaisesRegex(ValueError, "data type"):
                    tensor.__dlpack__()
                with self.assertRaisesRegex(ValueError, "data type"):
                    tensor.__dlpack_device__()
                with self.assertRaisesRegex(ValueError, "data type"):
                    tensor.release_dlpack()
                self.assertTrue(tensor.HasField("raw_data"))
        with self.assertRaises(ValueError):
            TensorProto(data_type=123456, raw_data=b"\0").__dlpack__()

    def test_malformed_payloads(self):
        for shape, raw in (
            ([-1], b""),
            ([0, -1], b""),
            ([2**62, 4], b""),
            ([1] * 129, b"\0"),
            ([2], b"\0"),
            ([0], b"\0"),
            ([], b""),
        ):
            with self.subTest(shape=shape, raw=raw):
                tensor = TensorProto(data_type=TensorProto.UINT8, dims=shape, raw_data=raw)
                with self.assertRaises(ValueError):
                    tensor.__dlpack__()
        typed = TensorProto(data_type=TensorProto.FLOAT, dims=[1], float_data=[1.0])
        with self.assertRaisesRegex(ValueError, "raw_data"):
            typed.__dlpack__()
        external = TensorProto(
            data_type=TensorProto.FLOAT, dims=[1], data_location=TensorProto.EXTERNAL
        )
        with self.assertRaisesRegex(ValueError, "external"):
            external.__dlpack__()
        segmented = TensorProto()
        segmented.ParseFromString(
            TensorProto(data_type=TensorProto.UINT8, dims=[1], raw_data=b"\0").SerializeToString()
            + b"\x1a\x04\x08\x00\x10\x01"
        )
        with self.assertRaisesRegex(ValueError, "segment"):
            segmented.__dlpack__()

    @unittest.skipUnless(sys.byteorder == "little", "Requires native little-endian storage.")
    def test_alignment(self):
        aligned, address, _ = self.make_borrowed(TensorProto.FLOAT, 4, 0)
        self.assertEqual(numpy.from_dlpack(aligned).ctypes.data, address)
        unaligned, _, _ = self.make_borrowed(TensorProto.FLOAT, 4, 1)
        with self.assertRaisesRegex(BufferError, "align"):
            unaligned.__dlpack__()

    @unittest.skipUnless(sys.byteorder == "little", "Requires native little-endian storage.")
    def test_aligned_owned_and_shape_snapshot(self):
        expected = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
        options = ParseOptions()
        options.alignment = 64
        tensor = TensorProto()
        tensor.ParseFromString(from_array(expected).SerializeToString(), options)
        view = numpy.from_dlpack(tensor)
        self.assertEqual(view.ctypes.data % 64, 0)
        tensor.dims.clear()
        tensor.dims.extend([6])
        del tensor
        gc.collect()
        self.assertEqual(view.shape, (2, 3))
        numpy.testing.assert_array_equal(view, expected)

    def test_protocol_arguments(self):
        tensor = TensorProto(data_type=TensorProto.UINT8, dims=[1], raw_data=b"\x07")
        capsule = tensor.__dlpack__(None, max_version=(1, 0), dl_device=(1, 0), copy=False)
        self.assertEqual(capsule_tensor(capsule).dtype.bits, 8)
        with self.assertRaisesRegex(BufferError, "copy"):
            tensor.__dlpack__(copy=True)
        with self.assertRaisesRegex(BufferError, "device"):
            tensor.__dlpack__(dl_device=(2, 0))
        with self.assertRaisesRegex(ValueError, "stream"):
            tensor.__dlpack__(stream=1)

    def test_loaded_mmap_payload(self):
        expected = numpy.arange(64, dtype=numpy.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.onnx")
            with open(path, "wb") as stream:
                stream.write(
                    ModelProto(
                        graph=GraphProto(initializer=[from_array(expected)])
                    ).SerializeToString()
                )
            options = ParseOptions()
            options.no_copy = True
            options.file_load_mode = FileLoadMode.MMAP
            model = ModelProto()
            model.ParseFromFile(path, options)
            graph = GraphProto()
            graph.initializer.add()
            graph.initializer[0] = model.graph.initializer[0]
            copied = graph.initializer[0]
            view = numpy.from_dlpack(copied)
            source_view = numpy.from_dlpack(model.graph.initializer[0])
            self.assertEqual(view.ctypes.data, source_view.ctypes.data)
            del source_view
            del model, graph, copied
            gc.collect()
            numpy.testing.assert_array_equal(view, expected)
            del view

    def test_loaded_external_payload(self):
        expected = numpy.arange(64, dtype=numpy.uint8)
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "weights.bin"), "wb") as stream:
                stream.write(expected.tobytes())
            tensor = TensorProto(
                data_type=TensorProto.UINT8, dims=[64], data_location=TensorProto.EXTERNAL
            )
            tensor.external_data.add(key="location", value="weights.bin")
            with self.assertRaisesRegex(ValueError, "external"):
                tensor.__dlpack__()
            tensor.load_external_data(directory)
            self.assertEqual(tensor.data_location, TensorProto.EXTERNAL)
            view = numpy.from_dlpack(tensor)
            del tensor
            gc.collect()
            numpy.testing.assert_array_equal(view, expected)


if __name__ == "__main__":
    unittest.main()
