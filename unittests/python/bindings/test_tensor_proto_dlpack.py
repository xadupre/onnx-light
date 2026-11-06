import copy
import ctypes
import gc
import importlib.util
import os
import sys
import tempfile
import unittest
import weakref

import numpy

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import (
    FileLoadMode,
    GraphProto,
    ModelProto,
    ParseOptions,
    SerializeOptions,
    TensorProto,
    load_model,
    save_model,
)
import onnx_light.onnx.numpy_helper as onh


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


class DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("version", DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", DLTensor),
    ]


def capsule_tensor(capsule):
    """Returns the tensor at the start of a legacy DLManagedTensor capsule."""
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    tensor = ctypes.cast(get_pointer(capsule, b"dltensor"), ctypes.POINTER(DLTensor)).contents
    tensor._capsule = capsule
    return tensor


class CapsuleProducer:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self, **kwargs):
        return self.capsule

    def __dlpack_device__(self):
        return (1, 0)


class TestTensorProtoDLPack(ExtTestCase):
    def test_external_save_shares_numpy_storage_and_isolates_callbacks(self):
        from onnx_light.onnx_core.graph_builder import GraphBuilder

        for construction in ("dlpack", "builder", "reimport"):
            for threads in (1, 2):
                with (
                    self.subTest(construction=construction, threads=threads),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    array = numpy.arange(1024, dtype=numpy.float32).reshape(32, 32)
                    address = array.ctypes.data
                    if construction == "builder":
                        builder = GraphBuilder()
                        builder.init(array, name="weight", copy=False)
                        model = builder.to_onnx()
                        del builder
                    else:
                        imported = TensorProto.from_dlpack(array, name="weight")
                        if construction == "reimport":
                            imported = TensorProto.from_dlpack(imported, name="weight")
                        model = ModelProto()
                        model.graph.initializer.append(imported)
                        del imported
                    model.graph.node.add(op_type="Identity", input=["weight"], output=["output"])
                    tensor = model.graph.initializer[0]
                    before = model.SerializeToString()
                    lifetime = weakref.ref(array)
                    expected = array.copy()
                    del array
                    seen = []

                    def callback(node, graph):
                        seen.append(capsule_tensor(graph.initializer[0].__dlpack__()).data)
                        self.assertIsNotNone(lifetime())
                        node.doc_string = "export only"

                    options = SerializeOptions()
                    options.num_threads = threads
                    options.min_parallel_block_size = 0
                    options.raw_data_threshold = 0
                    options.node_callback = callback
                    path = os.path.join(directory, "model.onnx")
                    weights = os.path.join(directory, "weights.bin")
                    model.SerializeToFile(path, options, weights)
                    self.assertEqual(seen, [address])
                    self.assertEqual(model.SerializeToString(), before)
                    self.assertEqual(capsule_tensor(tensor.__dlpack__()).data, address)
                    restored = load_model(path, load_external_data=True)
                    self.assertEqual(restored.graph.node[0].doc_string, "export only")
                    numpy.testing.assert_array_equal(
                        numpy.from_dlpack(restored.graph.initializer[0]), expected
                    )

                    def fail(node, graph):
                        self.assertEqual(
                            capsule_tensor(graph.initializer[0].__dlpack__()).data, address
                        )
                        raise ValueError("export callback failure")

                    options.node_callback = fail
                    with self.assertRaisesRegex(ValueError, "export callback failure"):
                        model.SerializeToFile(path, options, weights)
                    self.assertEqual(model.SerializeToString(), before)
                    self.assertEqual(capsule_tensor(tensor.__dlpack__()).data, address)
                    options.node_callback = None
                    options.max_serialized_size_bytes = 1
                    with self.assertRaisesRegex(RuntimeError, "max_serialized_size"):
                        model.SerializeToFile(path, options, weights)
                    self.assertIsNotNone(lifetime())
                    del tensor, model
                    gc.collect()
                    self.assertIsNone(lifetime())

    def test_external_save_retains_owner_after_callback_clears_source(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                array = numpy.arange(512 * 1024, dtype=numpy.float32)
                expected = array.copy()
                lifetime = weakref.ref(array)
                model = ModelProto()
                model.graph.initializer.append(TensorProto.from_dlpack(array, name="weight"))
                model.graph.node.add(op_type="Identity", input=["weight"], output=["output"])
                del array

                def callback(node, graph):
                    model.Clear()
                    gc.collect()
                    self.assertIsNotNone(lifetime())
                    if fail:
                        raise ValueError("export callback failure")

                options = SerializeOptions()
                options.num_threads = 2
                options.min_parallel_block_size = 0
                options.raw_data_threshold = 0
                options.node_callback = callback
                path = os.path.join(directory, "model.onnx")
                weights = os.path.join(directory, "weights.bin")
                if fail:
                    with self.assertRaisesRegex(ValueError, "export callback failure"):
                        model.SerializeToFile(path, options, weights)
                else:
                    model.SerializeToFile(path, options, weights)
                    restored = load_model(path, load_external_data=True)
                    numpy.testing.assert_array_equal(
                        numpy.from_dlpack(restored.graph.initializer[0]), expected
                    )
                gc.collect()
                self.assertIsNone(lifetime())

    def test_external_save_same_path_preserves_mapped_sources(self):
        for ownership in ("mapping", "callback", "callback_failure", "dlpack", "wrapped"):
            with self.subTest(ownership=ownership), tempfile.TemporaryDirectory() as directory:
                expected = numpy.arange(1024, dtype=numpy.float32)
                model = ModelProto()
                model.graph.initializer.append(TensorProto.from_dlpack(expected, name="weight"))
                path = os.path.join(directory, "model.onnx")
                save_model(
                    model,
                    path,
                    save_as_external_data=True,
                    location="weights.bin",
                    size_threshold=0,
                )
                released = []
                options = ParseOptions()
                options.no_copy = True
                options.num_threads = 1
                if ownership.startswith("callback"):

                    def callback(tensor, graph):
                        view = numpy.from_dlpack(tensor)

                        def cleanup():
                            released.append(view.size)

                        return cleanup

                    options.raw_data_callback = callback
                model = ModelProto()
                model.ParseFromFile(path, options, os.path.join(directory, "weights.bin"))
                if ownership in ("dlpack", "wrapped"):
                    tensor = model.graph.initializer[0]
                    producer = tensor if ownership == "dlpack" else numpy.from_dlpack(tensor)
                    imported = TensorProto.from_dlpack(producer, name="weight")
                    replacement = ModelProto()
                    replacement.graph.initializer.append(imported)
                    del tensor, producer, imported, model
                    model = replacement
                if ownership == "callback_failure":

                    def reject(tensor, graph, buffer, size_only):
                        raise ValueError("mapped export failure")

                    with self.assertRaisesRegex(ValueError, "mapped export failure"):
                        save_model(
                            model,
                            path,
                            save_as_external_data=True,
                            location="weights.bin",
                            size_threshold=0,
                            raw_data_callback=reject,
                        )
                    numpy.testing.assert_array_equal(
                        numpy.from_dlpack(model.graph.initializer[0]), expected
                    )
                save_model(
                    model,
                    path,
                    save_as_external_data=True,
                    location="weights.bin",
                    size_threshold=0,
                )
                numpy.testing.assert_array_equal(
                    numpy.from_dlpack(model.graph.initializer[0]), expected
                )
                restored = load_model(path, load_external_data=True)
                numpy.testing.assert_array_equal(
                    numpy.from_dlpack(restored.graph.initializer[0]), expected
                )
                if ownership.startswith("callback"):
                    self.assertEqual(released, [expected.size])
                del restored, model

    def test_external_save_numpy_mapping_views_are_not_classified_as_heap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "mapped.bin")
            mapping = numpy.memmap(path, mode="w+", dtype=numpy.float32, shape=(32,))
            try:
                mapping[:] = numpy.arange(32, dtype=numpy.float32)
                expected = mapping.copy()
                for producer in (mapping, numpy.asarray(mapping), numpy.asarray(mapping)[4:]):
                    with self.subTest(type=type(producer), shape=producer.shape):
                        model = ModelProto()
                        model.graph.initializer.append(
                            TensorProto.from_dlpack(producer, name="weight")
                        )
                        saved = os.path.join(directory, "model.onnx")
                        save_model(
                            model,
                            saved,
                            save_as_external_data=True,
                            location="weights.bin",
                            size_threshold=0,
                        )
                        self.assertNotEqual(
                            capsule_tensor(model.graph.initializer[0].__dlpack__()).data,
                            producer.ctypes.data,
                        )
                        numpy.testing.assert_array_equal(
                            numpy.from_dlpack(model.graph.initializer[0]),
                            expected[-producer.size :],
                        )
                        del model
                del producer
            finally:
                # NumPy has no public close API; explicitly release the backing mapping.
                mapping._mmap.close()

    @unittest.skipUnless(sys.byteorder == "little", "Requires native little-endian storage.")
    def test_import_numpy(self):
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
            numpy.complex64,
            numpy.complex128,
            numpy.bool_,
        ):
            for shape in ((), (2, 3), (0,), (2, 0, 3)):
                with self.subTest(dtype=dtype, shape=shape):
                    array = numpy.ones(shape, dtype=dtype)
                    expected = onh.from_array(array, name="weights").SerializeToString()
                    address = array.ctypes.data
                    source = weakref.ref(array)
                    tensor = TensorProto.from_dlpack(array, name="weights")
                    self.assertTrue(tensor.HasField("raw_data"))
                    if array.size:
                        self.assertEqual(capsule_tensor(tensor.__dlpack__()).data, address)
                        array[...] = 0
                        self.assertEqual(tensor.raw_data, array.tobytes())
                        array[...] = 1
                    del array
                    gc.collect()
                    self.assertIsNotNone(source())
                    self.assertEqual(tensor.SerializeToString(), expected)
                    del tensor
                    gc.collect()
                    self.assertIsNone(source())

    def test_import_legacy_producer_and_copies(self):
        original, address, released = self.make_borrowed()

        class LegacyProducer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __dlpack_device__(self):
                return (1, 0)

            def __dlpack__(self, stream=None):
                if stream is not None:
                    raise AssertionError("CPU stream must be None.")
                return self.tensor.release_dlpack()

        tensor = TensorProto.from_dlpack(LegacyProducer(original))
        copied = copy.copy(tensor)
        assigned = TensorProto()
        assigned.CopyFrom(tensor)
        assigned.CopyFrom(assigned)
        graph = GraphProto()
        graph.initializer.append(tensor)
        for item in (tensor, copied, assigned, graph.initializer[0]):
            self.assertEqual(capsule_tensor(item.__dlpack__()).data, address)
        del item, original, tensor, assigned, copied
        gc.collect()
        self.assertEqual(released, [])
        self.assertEqual(graph.initializer[0].raw_data, bytes(range(64)))
        del graph
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_import_readonly_singleton_and_metadata_snapshot(self):
        array = numpy.arange(6, dtype=numpy.float32).reshape(2, 1, 3)
        array.setflags(write=False)
        tensor = TensorProto.from_dlpack(array)
        self.assertEqual(capsule_tensor(tensor.__dlpack__()).data, array.ctypes.data)
        numpy.testing.assert_array_equal(numpy.from_dlpack(tensor), array)

        original, address, released = self.make_borrowed()
        capsule = original.release_dlpack()
        descriptor = capsule_tensor(capsule)
        shape = (ctypes.c_int64 * 3)(2, 1, 32)
        strides = (ctypes.c_int64 * 3)(32, 0, 1)
        descriptor.ndim = 3
        descriptor.shape = shape
        descriptor.strides = strides
        tensor = TensorProto.from_dlpack(CapsuleProducer(capsule))
        shape[0] = 99
        descriptor.dtype.bits = 32
        self.assertEqual(list(tensor.dims), [2, 1, 32])
        self.assertEqual(tensor.data_type, TensorProto.UINT8)
        self.assertEqual(capsule_tensor(tensor.__dlpack__()).data, address)
        del original, capsule, tensor
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_copy_from_owned_still_copies(self):
        tensor = onh.from_array(numpy.arange(8, dtype=numpy.uint8))
        copied = TensorProto()
        copied.CopyFrom(tensor)
        self.assertNotEqual(
            capsule_tensor(tensor.__dlpack__()).data, capsule_tensor(copied.__dlpack__()).data
        )
        self.assertEqual(tensor.SerializeToString(), copied.SerializeToString())
        tensor.Clear()
        self.assertEqual(copied.raw_data, bytes(range(8)))

    def test_copy_empty_and_ownerless(self):
        for payload in (b"", bytes(range(8))):
            for borrowed in (False, True):
                with self.subTest(payload=payload, borrowed=borrowed):
                    tensor = TensorProto(
                        data_type=TensorProto.UINT8, dims=[len(payload)], raw_data=payload
                    )
                    serialized = tensor.SerializeToString()
                    if borrowed:
                        options = ParseOptions()
                        options.no_copy = True
                        tensor = TensorProto()
                        tensor.ParseFromString(serialized, options)
                    copied = copy.copy(tensor)
                    assigned = TensorProto()
                    assigned.CopyFrom(tensor)
                    for item in (copied, assigned):
                        if payload:
                            self.assertNotEqual(
                                capsule_tensor(item.__dlpack__()).data,
                                capsule_tensor(tensor.__dlpack__()).data,
                            )
                        self.assertTrue(item.HasField("raw_data"))
                    del tensor, serialized
                    gc.collect()
                    self.assertEqual(copied.raw_data, payload)
                    self.assertEqual(assigned.raw_data, payload)

    def test_import_builder_model_lifetime(self):
        from onnx_light.onnx_core.graph_builder import GraphBuilder

        for move in (False, True):
            with self.subTest(move=move):
                original, address, released = self.make_borrowed()
                tensor = TensorProto.from_dlpack(original, name="weight")
                builder = GraphBuilder("dlpack")
                if move:
                    builder.make_initializer_move(tensor)
                    self.assertFalse(tensor.HasField("raw_data"))
                else:
                    builder.make_initializer(tensor)
                model = builder.to_onnx()
                self.assertEqual(
                    capsule_tensor(model.graph.initializer[0].__dlpack__()).data, address
                )
                del original, tensor, builder
                gc.collect()
                self.assertEqual(released, [])
                serialized = model.SerializeToString()
                restored = ModelProto()
                restored.ParseFromString(serialized)
                self.assertEqual(restored.graph.initializer[0].raw_data, bytes(range(64)))
                del model
                gc.collect()
                self.assertEqual(len(released), 1)

    def test_import_numpy_rejections(self):
        array = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
        for invalid in (
            array.T,
            array[:, ::2],
            array[::-1],
            numpy.broadcast_to(array, (2, 3, 4)),
        ):
            with (
                self.subTest(strides=invalid.strides),
                self.assertRaisesRegex(BufferError, "compact"),
            ):
                TensorProto.from_dlpack(invalid)
        for invalid in (array.astype(">f4"), numpy.array(["abc"]), numpy.array([object()])):
            with self.subTest(dtype=invalid.dtype), self.assertRaises((BufferError, TypeError)):
                TensorProto.from_dlpack(invalid)
        with self.assertRaises(TypeError):
            TensorProto.from_dlpack(object())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires PyTorch.")
    def test_import_torch(self):
        import torch

        array = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        address = array.data_ptr()
        tensor = TensorProto.from_dlpack(array)
        del array
        gc.collect()
        self.assertEqual(capsule_tensor(tensor.__dlpack__()).data, address)
        numpy.testing.assert_array_equal(
            numpy.from_dlpack(tensor), numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
        )

    def test_import_native_release_without_gil(self):
        original, _, released = self.make_borrowed()
        tensor = TensorProto.from_dlpack(CapsuleProducer(original.release_dlpack()))
        capsule = tensor.release_dlpack()
        del original, tensor
        self.assertEqual(released, [])
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        pointer = get_pointer(capsule, b"dltensor")
        descriptor = ctypes.cast(pointer, ctypes.POINTER(DLManagedTensor)).contents
        set_name = ctypes.pythonapi.PyCapsule_SetName
        set_name.restype = ctypes.c_int
        set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
        self.assertEqual(set_name(capsule, b"used_dltensor"), 0)
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.deleter)(pointer)
        self.assertEqual(len(released), 1)
        del capsule
        gc.collect()
        self.assertEqual(len(released), 1)

    def test_numpy_initializer_native_release_without_gil(self):
        from onnx_light.onnx_core.graph_builder import GraphBuilder

        array = numpy.ones(4, dtype=numpy.float32)
        references = sys.getrefcount(array)
        builder = GraphBuilder()
        builder.init(array, copy=False)
        model = builder.to_onnx()
        tensor = copy.copy(model.graph.initializer[0])
        capsule = tensor.release_dlpack()
        del builder, model, tensor
        gc.collect()
        self.assertEqual(sys.getrefcount(array), references + 1)
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        pointer = get_pointer(capsule, b"dltensor")
        descriptor = ctypes.cast(pointer, ctypes.POINTER(DLManagedTensor)).contents
        set_name = ctypes.pythonapi.PyCapsule_SetName
        set_name.restype = ctypes.c_int
        set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
        self.assertEqual(set_name(capsule, b"used_dltensor"), 0)
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.deleter)(pointer)
        self.assertEqual(sys.getrefcount(array), references)
        del capsule
        gc.collect()
        self.assertEqual(sys.getrefcount(array), references)

    def test_import_protocol_errors(self):
        class DeviceProducer:
            def __dlpack_device__(self):
                return (2, 0)

            def __dlpack__(self, **kwargs):
                raise AssertionError("Must reject device before exporting.")

        with self.assertRaisesRegex(BufferError, "CPU"):
            TensorProto.from_dlpack(DeviceProducer())

        class AsyncProducer(DeviceProducer):
            def __dlpack_device__(self):
                return (1, 0)

            def __dlpack__(self, *, stream, max_version, copy):
                if stream is None:
                    raise BufferError("requires asynchronous synchronization")
                raise AssertionError("Only stream=None is supported.")

        with self.assertRaisesRegex(BufferError, "asynchronous"):
            TensorProto.from_dlpack(AsyncProducer())

        class BrokenProducer(AsyncProducer):
            def __dlpack__(self, **kwargs):
                raise TypeError("producer failure")

        with self.assertRaisesRegex(TypeError, "producer failure"):
            TensorProto.from_dlpack(BrokenProducer())
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            TensorProto.from_dlpack(CapsuleProducer(None))

    def test_import_capsule_validation_and_cleanup(self):
        cases = [
            ("device", lambda t: setattr(t.device, "device_type", 2), BufferError),
            ("device_id", lambda t: setattr(t.device, "device_id", 1), BufferError),
            ("dtype", lambda t: setattr(t.dtype, "code", 3), ValueError),
            ("lanes", lambda t: setattr(t.dtype, "lanes", 2), ValueError),
            ("subbyte", lambda t: setattr(t.dtype, "bits", 4), ValueError),
            ("rank", lambda t: setattr(t, "ndim", -1), ValueError),
            ("rank_limit", lambda t: setattr(t, "ndim", 129), ValueError),
            ("shape", lambda t: setattr(t, "shape", None), ValueError),
            ("negative_dim", lambda t: t.shape.__setitem__(0, -1), ValueError),
            ("overflow", lambda t: t.shape.__setitem__(0, 2**63 - 1), ValueError),
            ("null_data", lambda t: setattr(t, "data", None), BufferError),
            ("offset", lambda t: setattr(t, "byte_offset", 1), BufferError),
            ("offset_overflow", lambda t: setattr(t, "byte_offset", 2**64 - 1), BufferError),
            ("alignment", lambda t: setattr(t, "data", t.data + 1), BufferError),
        ]
        for label, mutate, exception in cases:
            with self.subTest(label=label):
                original, _, released = self.make_borrowed(TensorProto.FLOAT, alignment=4)
                capsule = original.release_dlpack()
                mutate(capsule_tensor(capsule))
                with self.assertRaises(exception):
                    TensorProto.from_dlpack(CapsuleProducer(capsule))
                self.assertEqual(len(released), 1)
                with self.assertRaisesRegex(ValueError, "unconsumed"):
                    TensorProto.from_dlpack(CapsuleProducer(capsule))
                del capsule, original
                gc.collect()
                self.assertEqual(len(released), 1)

    def test_import_byte_offset_and_empty_null(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                original, address, released = self.make_borrowed()
                capsule = original.release_dlpack()
                descriptor = capsule_tensor(capsule)
                descriptor.shape[0] = 0 if empty else 60
                descriptor.byte_offset = 0 if empty else 4
                if empty:
                    descriptor.data = None
                tensor = TensorProto.from_dlpack(CapsuleProducer(capsule))
                self.assertTrue(tensor.HasField("raw_data"))
                self.assertEqual(tensor.raw_data, b"" if empty else bytes(range(4, 64)))
                if not empty:
                    self.assertEqual(capsule_tensor(tensor.__dlpack__()).data, address + 4)
                del capsule, original
                self.assertEqual(released, [])
                del tensor
                gc.collect()
                self.assertEqual(len(released), 1)

    @unittest.skipUnless(sys.byteorder == "little", "Requires native little-endian storage.")
    def test_bool_uint8_metadata_roundtrip(self):
        for dtype, onnx_type, dlpack_code in (
            (numpy.float32, TensorProto.FLOAT, 2),
            (numpy.bool_, TensorProto.BOOL, 6),
            (numpy.uint8, TensorProto.UINT8, 1),
        ):
            with self.subTest(dtype=dtype):
                array = numpy.array([0, 1, 0, 1], dtype=dtype)
                if dtype == numpy.uint8:
                    array[2:] = [2, 255]
                tensor = TensorProto.from_dlpack(array)
                capsule = tensor.__dlpack__()
                descriptor = capsule_tensor(capsule)
                self.assertEqual(descriptor.dtype.code, dlpack_code)
                self.assertEqual(descriptor.dtype.bits, array.itemsize * 8)
                self.assertEqual(descriptor.dtype.lanes, 1)
                descriptor.shape[0] = 3
                strides = (ctypes.c_int64 * 1)(1)
                descriptor.strides = strides
                descriptor.byte_offset = array.itemsize
                imported = TensorProto.from_dlpack(CapsuleProducer(capsule))
                view = numpy.from_dlpack(imported)
                self.assertEqual(imported.data_type, onnx_type)
                self.assertEqual(view.dtype, array.dtype)
                numpy.testing.assert_array_equal(view, array[1:])
                self.assertEqual(view.ctypes.data, array.ctypes.data + array.itemsize)
                self.assertTrue(numpy.shares_memory(view, array))
                array[1] = 0
                numpy.testing.assert_array_equal(view, array[1:])
                del tensor, imported, capsule, descriptor
                gc.collect()
                numpy.testing.assert_array_equal(view, array[1:])

    def test_import_null_deleter(self):
        array = numpy.array([0, 1, 2, 255], dtype=numpy.uint8)
        shape = (ctypes.c_int64 * 1)(array.size)
        managed = DLManagedTensor()
        managed.dl_tensor = DLTensor(
            array.ctypes.data, DLDevice(1, 0), 1, DLDataType(1, 8, 1), shape, None, 0
        )
        new_capsule = ctypes.pythonapi.PyCapsule_New
        new_capsule.restype = ctypes.py_object
        new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        capsule = new_capsule(ctypes.addressof(managed), b"dltensor", None)
        tensor = TensorProto.from_dlpack(CapsuleProducer(capsule))
        view = numpy.from_dlpack(tensor)
        self.assertEqual(tensor.data_type, TensorProto.UINT8)
        self.assertEqual(view.dtype, array.dtype)
        self.assertTrue(numpy.shares_memory(view, array))
        del tensor, view, capsule
        gc.collect()
        numpy.testing.assert_array_equal(array, [0, 1, 2, 255])

    def ort_dlpack_runtime(self):
        """Returns the optional runtime and its native bool-override contract."""
        if not importlib.util.find_spec("onnxruntime"):
            self.skipTest("Requires ONNX Runtime.")
        import onnxruntime
        from onnxruntime.capi import _pybind_state

        if not hasattr(onnxruntime.OrtValue, "from_dlpack"):
            self.skipTest("Requires the public ONNX Runtime DLPack protocol API.")
        # The version string does not identify builds containing the DLPack 1.3 change.
        legacy = "is_bool_tensor" in _pybind_state.OrtValue.from_dlpack.__doc__
        return onnxruntime, legacy

    @unittest.skipUnless(sys.byteorder == "little", "Requires native little-endian storage.")
    def test_ort_roundtrip(self):
        ort, _ = self.ort_dlpack_runtime()
        for dtype in (numpy.float32, numpy.uint8):
            for shape in ((), (2, 3)):
                with self.subTest(dtype=dtype, shape=shape):
                    array = (
                        numpy.array(255, dtype=dtype)
                        if not shape
                        else numpy.array([[0, 1, 2], [3, 127, 255]], dtype=dtype)
                    )
                    tensor = TensorProto.from_dlpack(array)
                    for producer in (array, tensor):
                        value = ort.OrtValue.from_dlpack(producer)
                        view = numpy.from_dlpack(value)
                        imported = TensorProto.from_dlpack(value)
                        other = numpy.from_dlpack(imported)
                        self.assertEqual(value.__dlpack_device__(), (1, 0))
                        self.assertEqual(imported.data_type, tensor.data_type)
                        for result in (view, other):
                            self.assertEqual(result.dtype, array.dtype)
                            numpy.testing.assert_array_equal(result, array)
                            self.assertTrue(numpy.shares_memory(result, array))
                        array[...] = 1
                        numpy.testing.assert_array_equal(other, array)

    def test_ort_bool_contract(self):
        ort, legacy = self.ort_dlpack_runtime()
        array = numpy.array([True, False, True, False], dtype=numpy.bool_)
        tensor = TensorProto.from_dlpack(array)
        for producer in (array, tensor):
            if legacy:
                with self.assertRaisesRegex(RuntimeError, "Unsupported code 6"):
                    ort.OrtValue.from_dlpack(producer)
            else:
                value = ort.OrtValue.from_dlpack(producer)
                imported = TensorProto.from_dlpack(value)
                view = numpy.from_dlpack(imported)
                self.assertEqual(imported.data_type, TensorProto.BOOL)
                self.assertEqual(view.dtype, array.dtype)
                numpy.testing.assert_array_equal(view, array)
                self.assertTrue(numpy.shares_memory(view, array))

        value = ort.OrtValue.ortvalue_from_numpy(array)
        imported = TensorProto.from_dlpack(value)
        view = numpy.from_dlpack(imported)
        # Old exporters label bool storage as uint8; consumers must honor that metadata.
        self.assertEqual(imported.data_type, TensorProto.UINT8 if legacy else TensorProto.BOOL)
        self.assertEqual(view.dtype, numpy.dtype(numpy.uint8 if legacy else numpy.bool_))
        numpy.testing.assert_array_equal(view, array)
        self.assertTrue(numpy.shares_memory(view, array))

    def test_ort_capsule_lifetime(self):
        ort, _ = self.ort_dlpack_runtime()
        for consume in (False, True):
            with self.subTest(consume=consume):
                original, address, released = self.make_borrowed()
                capsule = original.__dlpack__()
                if consume:
                    value = ort.OrtValue.from_dlpack(capsule)
                    tensor = TensorProto.from_dlpack(value)
                    view = numpy.from_dlpack(tensor)
                    self.assertEqual(view.ctypes.data, address)
                    del value, tensor
                del original
                gc.collect()
                self.assertEqual(released, [])
                del capsule
                if consume:
                    gc.collect()
                    self.assertEqual(released, [])
                    numpy.testing.assert_array_equal(view, numpy.arange(64, dtype=numpy.uint8))
                    del view
                gc.collect()
                self.assertEqual(len(released), 1)

    @unittest.skipUnless(sys.byteorder == "little", "Requires native little-endian storage.")
    def test_ort_offset_strides_and_null_deleter(self):
        ort, legacy = self.ort_dlpack_runtime()
        if legacy:
            self.skipTest("Offset/stride/null-deleter import requires the DLPack 1.3 contract.")
        noncontiguous = numpy.arange(8, dtype=numpy.float32).reshape(2, 4)[:, ::2]
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            ort.OrtValue.from_dlpack(noncontiguous)
        for dtype in (numpy.float32, numpy.bool_, numpy.uint8):
            for null_deleter in (False, True):
                with self.subTest(dtype=dtype, null_deleter=null_deleter):
                    array = numpy.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=dtype)
                    tensor = TensorProto.from_dlpack(array)
                    capsule = tensor.__dlpack__()
                    descriptor = capsule_tensor(capsule)
                    shape = (ctypes.c_int64 * 2)(2, 3)
                    strides = (ctypes.c_int64 * 2)(3, 1)
                    descriptor.ndim = 2
                    descriptor.shape = shape
                    descriptor.strides = strides
                    descriptor.byte_offset = array.itemsize
                    if null_deleter:
                        # A separate caller-owned descriptor avoids leaking an export owner.
                        managed = DLManagedTensor()
                        managed.dl_tensor = descriptor
                        del descriptor, capsule
                        new_capsule = ctypes.pythonapi.PyCapsule_New
                        new_capsule.restype = ctypes.py_object
                        new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
                        capsule = new_capsule(ctypes.addressof(managed), b"dltensor", None)
                    value = ort.OrtValue.from_dlpack(capsule)
                    view = numpy.from_dlpack(value)
                    self.assertEqual(view.dtype, array.dtype)
                    self.assertEqual(view.ctypes.data, array.ctypes.data + array.itemsize)
                    numpy.testing.assert_array_equal(view, array[1:7].reshape(2, 3))
                    del view, value, capsule
                    if null_deleter:
                        del managed
                    else:
                        del descriptor
                    gc.collect()

    def test_import_versioned_capsules(self):
        for major, flags in ((1, 0), (1, 1), (2, 0), (1, 2), (1, 4), (1, 8)):
            with self.subTest(major=major, flags=flags):
                original, address, released = self.make_borrowed()
                legacy = original.release_dlpack()
                descriptor = DLManagedTensorVersioned()
                descriptor.version = DLPackVersion(major, 0)
                descriptor.flags = flags
                descriptor.dl_tensor = capsule_tensor(legacy)
                calls = []

                @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
                def release(pointer):
                    calls.append(pointer)
                    # The fixture keeps the descriptor alive; the native owner holds the payload.
                    legacy_owner.clear()

                legacy_owner = [legacy]
                descriptor.deleter = ctypes.cast(release, ctypes.c_void_p)
                new_capsule = ctypes.pythonapi.PyCapsule_New
                new_capsule.restype = ctypes.py_object
                new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
                capsule = new_capsule(ctypes.addressof(descriptor), b"dltensor_versioned", None)
                del legacy, original
                if major == 1 and flags in (0, 1):
                    tensor = TensorProto.from_dlpack(CapsuleProducer(capsule))
                    self.assertEqual(capsule_tensor(tensor.__dlpack__()).data, address)
                    self.assertEqual(calls, [])
                    del tensor
                else:
                    with self.assertRaises(BufferError):
                        TensorProto.from_dlpack(CapsuleProducer(capsule))
                self.assertEqual(calls, [ctypes.addressof(descriptor)])
                self.assertEqual(len(released), 1)
                with self.assertRaisesRegex(ValueError, "unconsumed"):
                    TensorProto.from_dlpack(CapsuleProducer(capsule))
                del capsule
                gc.collect()
                self.assertEqual(len(calls), 1)

    def test_release_owned_and_aligned(self):
        expected = numpy.arange(6, dtype=numpy.uint8).reshape(2, 3)
        for alignment in (0, 64):
            with self.subTest(alignment=alignment):
                options = ParseOptions()
                options.alignment = alignment
                tensor = TensorProto()
                tensor.ParseFromString(onh.from_array(expected).SerializeToString(), options)
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
        tensor = onh.from_array(numpy.arange(6, dtype=numpy.uint8))
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
                tensor = onh.from_array(expected)
                self.assertEqual(tensor.__dlpack_device__(), (1, 0))
                view = numpy.from_dlpack(tensor)
                other = numpy.from_dlpack(tensor)
                self.assertTrue(numpy.shares_memory(view, other))
                self.assertFalse(view.flags.writeable)
                self.assertEqual(view.dtype, expected.dtype)
                with self.assertRaisesRegex(ValueError, "active exported views"):
                    tensor.raw_data = numpy.zeros_like(expected).tobytes()
                self.assertEqual(tensor.raw_data, expected.tobytes())
                numpy.testing.assert_array_equal(view, expected)
                del other
                gc.collect()
                with self.assertRaisesRegex(ValueError, "active exported views"):
                    tensor.raw_data = numpy.zeros_like(expected).tobytes()
                del tensor
                gc.collect()
                numpy.testing.assert_array_equal(view, expected)

    def test_owned_assignment_after_export_release(self):
        tensor = onh.from_array(numpy.arange(6, dtype=numpy.uint8))
        view = numpy.from_dlpack(tensor)
        del view
        gc.collect()
        tensor.raw_data = bytes(6)
        self.assertEqual(tensor.raw_data, bytes(6))

    def test_exports_survive_clear_and_storage_replacement(self):
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                array = numpy.arange(6, dtype=numpy.uint8)
                tensor = TensorProto.from_dlpack(array) if borrowed else onh.from_array(array)
                first = numpy.from_dlpack(tensor)
                tensor.Clear()
                tensor.data_type = TensorProto.UINT8
                tensor.dims.extend([6])
                tensor.raw_data = b"second"
                second = numpy.from_dlpack(tensor)
                tensor.Clear()
                del tensor, array
                gc.collect()
                numpy.testing.assert_array_equal(first, numpy.arange(6, dtype=numpy.uint8))
                numpy.testing.assert_array_equal(
                    second, numpy.frombuffer(b"second", dtype=numpy.uint8)
                )

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
        # TensorProto.CopyFrom and append also preserve borrowed owner tokens.
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
        tensor.ParseFromString(onh.from_array(expected).SerializeToString(), options)
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
                        graph=GraphProto(initializer=[onh.from_array(expected)])
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
