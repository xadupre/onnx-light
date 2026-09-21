# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests zero-copy NumPy initializer ownership."""

import copy
import gc
import subprocess
import sys
import textwrap
import tracemalloc
import unittest
import weakref

import ml_dtypes
import numpy

from onnx_light.ext_test_case import import_or_skip
from onnx_light.onnx import ModelProto, TensorProto, checker, numpy_helper
from onnx_light.onnx_core.graph_builder import GraphBuilder


@unittest.skipUnless(sys.byteorder == "little", "Requires native little-endian storage.")
class TestGraphBuilderNumpy(unittest.TestCase):
    def assertPointer(self, tensor, array):
        view = numpy.from_dlpack(tensor)
        self.assertEqual(view.ctypes.data, array.ctypes.data)
        numpy.testing.assert_array_equal(view, array)

    def test_dtypes_shapes_and_readonly(self):
        for dtype in (
            numpy.bool_,
            numpy.int8,
            numpy.int16,
            numpy.int32,
            numpy.int64,
            numpy.uint8,
            numpy.uint16,
            numpy.uint32,
            numpy.uint64,
            numpy.float16,
            numpy.float32,
            numpy.float64,
            numpy.complex64,
            numpy.complex128,
        ):
            for shape in ((), (2, 3), (0,), (2, 0, 3)):
                for readonly in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, readonly=readonly):
                        array = numpy.ones(shape, dtype=dtype)
                        array.setflags(write=not readonly)
                        references = sys.getrefcount(array)
                        builder = GraphBuilder("numpy")
                        self.assertEqual(builder.init(array, copy=False), "init")
                        self.assertEqual(sys.getrefcount(array), references + 1)
                        model = builder.to_onnx()
                        tensor = model.graph.initializer[0]
                        self.assertTrue(tensor.HasField("raw_data"))
                        self.assertEqual(list(tensor.dims), list(shape))
                        if array.size:
                            self.assertPointer(tensor, array)
                        expected = numpy_helper.from_array(array, name="init")
                        expected.metadata_props.extend(tensor.metadata_props)
                        self.assertEqual(tensor.SerializeToString(), expected.SerializeToString())
                        del tensor, model, builder
                        gc.collect()
                        self.assertEqual(sys.getrefcount(array), references)

    def test_lifetime_copies_and_execution(self):
        ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        array = numpy.arange(6, dtype=numpy.float32)
        expected = array + 1
        source = weakref.ref(array)
        released = []
        weakref.finalize(array, released.append, "released")
        references = sys.getrefcount(array)
        builder = GraphBuilder("numpy")
        builder.set_opset_version("", 18)
        x = builder.inp("x", TensorProto.FLOAT, [6])
        weight = builder.init(array, name="weight", copy=False)
        builder.out(builder.op.Add(x, weight))
        first = builder.to_onnx()
        second = builder.to_onnx()
        rebuilt = GraphBuilder(first).to_onnx()
        shallow = copy.copy(first.graph.initializer[0])
        assigned = TensorProto()
        assigned.CopyFrom(shallow)
        assigned.CopyFrom(assigned)
        for tensor in (
            first.graph.initializer[0],
            second.graph.initializer[0],
            rebuilt.graph.initializer[0],
            shallow,
            assigned,
        ):
            self.assertPointer(tensor, array)
        del tensor
        self.assertEqual(sys.getrefcount(array), references + 1)
        independent = ModelProto()
        independent.CopyFrom(first)
        self.assertNotEqual(
            numpy.from_dlpack(independent.graph.initializer[0]).ctypes.data, array.ctypes.data
        )
        del array
        gc.collect()
        self.assertIsNotNone(source())
        del builder, first, second, shallow
        gc.collect()
        self.assertEqual(released, [])
        checker.check_model(rebuilt)
        serialized = rebuilt.SerializeToString()
        restored = ModelProto()
        restored.ParseFromString(serialized)
        for model in (rebuilt, restored, independent):
            evaluator = ReferenceEvaluator(model)
            numpy.testing.assert_array_equal(
                evaluator.run(None, {"x": numpy.ones(6, dtype=numpy.float32)})[0], expected
            )
        del model, evaluator, rebuilt
        gc.collect()
        self.assertEqual(released, [])
        assigned.Clear()
        gc.collect()
        self.assertIsNone(source())
        self.assertEqual(released, ["released"])
        del assigned
        gc.collect()
        self.assertEqual(released, ["released"])

    def test_replacing_payload_releases_last_reference(self):
        for replacement in ("raw_data", "Clear", "CopyFrom", "ParseFromString"):
            with self.subTest(replacement=replacement):
                array = numpy.ones(4, dtype=numpy.float32)
                references = sys.getrefcount(array)
                builder = GraphBuilder()
                builder.init(array, copy=False)
                model = builder.to_onnx()
                tensor = model.graph.initializer[0]
                del builder
                gc.collect()
                self.assertEqual(sys.getrefcount(array), references + 1)
                if replacement == "raw_data":
                    tensor.raw_data = bytes(array.nbytes)
                elif replacement == "Clear":
                    tensor.Clear()
                elif replacement == "CopyFrom":
                    tensor.CopyFrom(TensorProto())
                else:
                    tensor.ParseFromString(
                        numpy_helper.from_array(
                            numpy.zeros(4, dtype=numpy.float32)
                        ).SerializeToString()
                    )
                gc.collect()
                self.assertEqual(sys.getrefcount(array), references)
                del tensor, model
                gc.collect()
                self.assertEqual(sys.getrefcount(array), references)

    def test_model_import_preserves_metadata(self):
        builder = GraphBuilder()
        array = numpy.ones(2, dtype=numpy.float32)
        builder.init(array, copy=False)
        model = builder.to_onnx()
        metadata = {
            "ir_version": 10,
            "producer_name": "numpy",
            "producer_version": "1",
            "domain": "example",
            "model_version": 42,
            "doc_string": "Borrowed weights.",
        }
        for name, value in metadata.items():
            setattr(model, name, value)
        model.metadata_props.add(key="custom", value="retained")
        configuration = model.configuration.add()
        configuration.name = "cpu"
        configuration.num_devices = 1
        configuration.device.append("cpu")
        rebuilt = GraphBuilder(model).to_onnx()
        for name, value in metadata.items():
            self.assertEqual(getattr(rebuilt, name), value)
        self.assertIn(("custom", "retained"), [(p.key, p.value) for p in rebuilt.metadata_props])
        self.assertEqual(
            rebuilt.configuration[0].SerializeToString(), configuration.SerializeToString()
        )
        self.assertPointer(rebuilt.graph.initializer[0], array)

    def test_contiguous_view_and_visible_mutations(self):
        base = numpy.arange(12, dtype=numpy.float32)
        array = base[3:9].reshape(2, 3)
        base_ref = weakref.ref(base)
        array_ref = weakref.ref(array)
        builder = GraphBuilder()
        builder.init(array, copy=False)
        model = builder.to_onnx()
        self.assertPointer(model.graph.initializer[0], array)
        self.assertTrue(array.flags.writeable)
        base[3:9] = 7
        numpy.testing.assert_array_equal(
            numpy.from_dlpack(model.graph.initializer[0]), numpy.full((2, 3), 7)
        )
        del array, base, builder
        gc.collect()
        self.assertIsNotNone(array_ref())
        self.assertIsNotNone(base_ref())
        del model
        gc.collect()
        self.assertIsNone(array_ref())
        self.assertIsNone(base_ref())

    def test_default_still_copies(self):
        for options in ({}, {"copy": True}):
            with self.subTest(options=options):
                array = numpy.arange(6, dtype=numpy.float32)
                expected = array.copy()
                references = sys.getrefcount(array)
                builder = GraphBuilder()
                builder.init(array, **options)
                model = builder.to_onnx()
                self.assertEqual(sys.getrefcount(array), references)
                self.assertNotEqual(
                    numpy.from_dlpack(model.graph.initializer[0]).ctypes.data, array.ctypes.data
                )
                array[:] = -1
                numpy.testing.assert_array_equal(
                    numpy.from_dlpack(model.graph.initializer[0]), expected
                )

    def test_borrowed_values_are_not_cached_by_shape_analysis(self):
        for dtype in (numpy.int64, numpy.float32):
            for do_copy in (False, True):
                with self.subTest(dtype=dtype, copy=do_copy):
                    array = numpy.array([2, 3], dtype=dtype)
                    builder = GraphBuilder()
                    name = builder.init(array, copy=do_copy)
                    descriptor = builder.shapes.get(name)
                    self.assertEqual(list(descriptor.shape), [2])
                    self.assertEqual(descriptor.has_min(), do_copy)
                    self.assertEqual(descriptor.has_max(), do_copy)
                    self.assertEqual(
                        descriptor.has_value_as_shape(), do_copy and dtype == numpy.int64
                    )

    def test_rejections_do_not_retain_or_insert(self):
        matrix = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
        invalid_arrays = (
            matrix.T,
            matrix[:, ::2],
            matrix[::-1],
            numpy.broadcast_to(matrix, (2, 3, 4)),
            matrix.astype(">f4"),
            numpy.array(["abc"]),
            numpy.array([b"abc"]),
            numpy.array([object()]),
            numpy.zeros(2, dtype=[("value", "f4")]),
            numpy.zeros(2, dtype="datetime64[ns]"),
            numpy.zeros(2, dtype="V4"),
            numpy.zeros(2, dtype=ml_dtypes.bfloat16),
        )
        if numpy.dtype(numpy.longdouble).itemsize > 8:
            invalid_arrays += (numpy.zeros(2, dtype=numpy.longdouble),)
        for array in invalid_arrays:
            with self.subTest(dtype=array.dtype, strides=array.strides):
                references = sys.getrefcount(array)
                builder = GraphBuilder()
                with self.assertRaisesRegex((ValueError, TypeError), "copy=False"):
                    builder.init(array, name="invalid", copy=False)
                gc.collect()
                self.assertEqual(sys.getrefcount(array), references)
                self.assertFalse(builder.has_name("invalid"))
                self.assertEqual(len(builder.to_onnx().graph.initializer), 0)
        with self.assertRaises(TypeError):
            GraphBuilder().init([1, 2], copy=False)

    def test_buffer_acquisition_failure_preserves_tensor(self):
        tensor = numpy_helper.from_array(numpy.ones(2, dtype=numpy.float32))
        expected = tensor.SerializeToString()
        array = numpy.arange(6, dtype=numpy.float32)[::2]
        references = sys.getrefcount(array)
        with self.assertRaises((ValueError, BufferError)):
            tensor._set_raw_data_from_buffer(array)
        gc.collect()
        self.assertEqual(sys.getrefcount(array), references)
        self.assertEqual(tensor.SerializeToString(), expected)

    def test_misaligned_contiguous_arrays_require_copy(self):
        for dtype in (numpy.int16, numpy.int32, numpy.int64, numpy.float32, numpy.float64):
            with self.subTest(dtype=dtype):
                array = numpy.ndarray(
                    (3,),
                    dtype=dtype,
                    buffer=bytearray(3 * numpy.dtype(dtype).itemsize + 1),
                    offset=1,
                )
                array[:] = [1, 2, 3]
                self.assertTrue(array.flags.c_contiguous)
                self.assertFalse(array.flags.aligned)
                references = sys.getrefcount(array)
                builder = GraphBuilder()
                with self.assertRaisesRegex(ValueError, "dtype-aligned"):
                    builder.init(array, copy=False)
                gc.collect()
                self.assertEqual(sys.getrefcount(array), references)
                self.assertEqual(len(builder.to_onnx().graph.initializer), 0)
                self.assertEqual(builder.init(array, copy=True), "init")
                tensor = builder.to_onnx().graph.initializer[0]
                numpy.testing.assert_array_equal(numpy_helper.to_array(tensor), array)

    def test_name_collision_releases_buffer(self):
        builder = GraphBuilder()
        builder.init(numpy.ones(2, dtype=numpy.float32), name="weight")
        array = numpy.zeros(3, dtype=numpy.float32)
        references = sys.getrefcount(array)
        with self.assertRaisesRegex(ValueError, "already"):
            builder.init(array, name="weight", copy=False)
        gc.collect()
        self.assertEqual(sys.getrefcount(array), references)
        self.assertEqual(builder.init(array, copy=False), "init")
        self.assertEqual(builder.init(array, copy=False), "init_1")

    def test_no_payload_materialization(self):
        class NoBytesArray(numpy.ndarray):
            def tobytes(self, *args, **kwargs):
                raise AssertionError("The zero-copy path must not materialize bytes.")

        array = numpy.ones(2**22, dtype=numpy.float32).view(NoBytesArray)
        builder = GraphBuilder()
        tracemalloc.start()
        try:
            builder.init(array, copy=False)
            model = builder.to_onnx()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, array.nbytes // 4)
        self.assertPointer(model.graph.initializer[0], array)

    @unittest.skipUnless(sys.platform == "linux", "Requires Linux address-space accounting.")
    def test_no_native_payload_sized_allocation(self):
        code = textwrap.dedent("""
            import os
            import resource
            import numpy
            from onnx_light.onnx_core.graph_builder import GraphBuilder

            arrays = [
                numpy.ones(2**24, dtype=numpy.float32),
                numpy.ones(2**23, dtype=numpy.int64),
            ]
            builder = GraphBuilder()
            with open("/proc/self/statm") as stream:
                virtual_size = int(stream.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            # Leave room for metadata, but not for a single 64 MiB payload copy.
            resource.setrlimit(resource.RLIMIT_AS, (virtual_size + 16 * 2**20, hard))
            for array in arrays:
                builder.init(array, copy=False)
            model = builder.to_onnx()
            rebuilt = GraphBuilder(model).to_onnx()
            for tensor, array in zip(rebuilt.graph.initializer, arrays):
                assert numpy.from_dlpack(tensor).ctypes.data == array.ctypes.data
            """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
