# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests portable quantization through actual native bindings."""

import gc
from pathlib import Path
import tempfile
import unittest

import numpy

from onnx_light import onnx
from onnx_light.ext_test_case import import_or_skip
from onnx_light.onnx import numpy_helper

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
QuantizationFormat = runtime.QuantizationFormat


class TestQuantizedValues(unittest.TestCase):
    def roundtrip(self, values, plan):
        """Returns reconstructed values after serialization and source release."""
        source = numpy_helper.from_array(values, name="weight")
        original = source.SerializeToString()
        encoded = runtime.quantize_tensor_proto(source, plan)
        self.assertEqual(source.SerializeToString(), original)
        wire = encoded.SerializeToString()
        del source, encoded
        gc.collect()
        restored = onnx.EncodedValueProto()
        restored.ParseFromString(wire)
        result = runtime.dequantize_tensor_proto(restored)
        self.assertEqual(result.name, "weight")
        return numpy_helper.to_array(result)

    def test_public_module_and_block_mutation(self):
        from onnx_light.onnx_core.quantization import make_quantization_plan, quantization_formats

        formats = quantization_formats()
        self.assertIsInstance(formats, list)
        self.assertEqual(len(formats), 40)
        self.assertTrue(all(isinstance(value, QuantizationFormat) for value in formats))
        self.assertEqual(formats[0], QuantizationFormat.INT8)
        self.assertEqual(formats[-1], QuantizationFormat.COLUMN_MAJOR)
        for value in formats:
            self.assertEqual(
                runtime.parse_quantization_format(runtime.quantization_format_name(value)), value
            )
        plan = make_quantization_plan(QuantizationFormat.INT4, 6, 3)
        first = plan.run(0)
        second = plan.run(0)
        block = first.block(0)
        block.scale = 0.5
        first.blocks = [block]
        block = second.block(1)
        block.scale = 2
        second.blocks = [block]
        second.layout.bits = 5
        plan.runs = [first, second]
        source = numpy.array([[-4, 0, 3.5], [-32, 0, 30]], dtype=numpy.float32)
        numpy.testing.assert_array_equal(self.roundtrip(source, plan), source)
        with self.assertRaisesRegex(ValueError, "index"):
            plan.run(2)

    def test_all_catalogue_profiles(self):
        source = numpy.array([-1, -1, 0, 0, 1, 1, 1, 1], dtype=numpy.float32)
        for name in runtime.quantization_formats():
            with self.subTest(format=name):
                plan = runtime.make_quantization_plan(name, source.size, 4)
                run = plan.run(0)
                blocks = run.blocks
                for block in blocks:
                    if (
                        run.layout.method == runtime.QuantizationMethod.CODEBOOK
                        and not block.codebook
                    ):
                        run.layout.bits = 2
                        run.layout.entries = 4
                        run.layout.vector_size = 2
                        block.codebook = [
                            float(entry - 1) if book == 0 else 0.0
                            for book in range(run.layout.books)
                            for entry in range(4)
                            for _ in range(2)
                        ]
                    if name == QuantizationFormat.IQ4_NL:
                        block.scale = 0.01
                run.blocks = blocks
                plan.set_run(0, run)
                if name in {
                    QuantizationFormat.QUAROT,
                    QuantizationFormat.QUIP_SHARP,
                    QuantizationFormat.SMOOTHQUANT,
                }:
                    plan.transform_size = 2
                    plan.forward = [1, 0, 0, 1]
                    plan.inverse = plan.forward
                tolerance = (
                    1
                    if name == QuantizationFormat.BINARY
                    else 0.14 if name == QuantizationFormat.IQ4_NL else 1e-6
                )
                numpy.testing.assert_allclose(
                    self.roundtrip(source, plan), source, rtol=0, atol=tolerance
                )

    def test_proto_dtypes_scalar_and_empty(self):
        for dtype in (numpy.float16, numpy.float32, numpy.float64):
            for shape in ((), (0,), (2, 0), (2, 3)):
                with self.subTest(dtype=dtype, shape=shape):
                    source = numpy.full(shape, 2, dtype=dtype)
                    plan = runtime.make_quantization_plan(QuantizationFormat.INT8, source.size)
                    result = self.roundtrip(source, plan)
                    self.assertEqual(result.dtype, source.dtype)
                    self.assertEqual(result.shape, source.shape)
                    numpy.testing.assert_array_equal(result, source)

    def test_tensor_runtime_value_bridge(self):
        source = numpy.array([1, 2, 3], dtype=numpy.float32)
        tensor = runtime.tensor_from_numpy(
            "x", onnx.TensorProto.FLOAT, list(source.shape), source.view(numpy.uint8)
        )
        plan = runtime.make_quantization_plan(QuantizationFormat.INT8, source.size)
        encoded = runtime.quantize_tensor(tensor, plan)
        self.assertIsInstance(encoded, onnx.EncodedValueProto)
        decoded = runtime.dequantize_tensor(encoded)
        del tensor, encoded
        gc.collect()
        numpy.testing.assert_array_equal(numpy.from_dlpack(decoded), source)

    def test_rotation_permutation_sparse_and_serialization(self):
        source = numpy.array([[1, 1000], [2, 3]], dtype=numpy.float32)
        plan = runtime.make_quantization_plan(QuantizationFormat.QUAROT, source.size)
        plan.transform_size = 2
        plan.forward = [1, 1, 1, -1]
        plan.inverse = [0.5, 0.5, 0.5, -0.5]
        plan.permutation = [2, 3, 0, 1]
        plan.outliers = [1]
        numpy.testing.assert_array_equal(self.roundtrip(source, plan), source)

    def test_exact_affine_bytes(self):
        source = numpy.array([-9, -7.5, -0.5, 0.5, 1.5, 6.5, 8], dtype=numpy.float32)
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, source.size)
        encoded = runtime.quantize_tensor_proto(numpy_helper.from_array(source), plan)
        self.assertEqual(bytes(encoded.raw_data)[-4:], b"\x88\x00\x62\x07")
        numpy.testing.assert_array_equal(
            numpy_helper.to_array(runtime.dequantize_tensor_proto(encoded)),
            [-8, -8, 0, 0, 2, 6, 7],
        )

    def test_missing_parameters_are_not_invented(self):
        source = numpy_helper.from_array(numpy.ones(8, dtype=numpy.float32))
        for format_name, message in [
            (QuantizationFormat.AQLM, "codebook"),
            (QuantizationFormat.STQ1_0, "codebook"),
            (QuantizationFormat.IQ1_S, "codebook"),
            (QuantizationFormat.SQUEEZELLM, "codebook"),
            (QuantizationFormat.QUAROT, "transforms"),
        ]:
            with self.subTest(format=format_name):
                plan = runtime.make_quantization_plan(format_name, 8)
                with self.assertRaisesRegex(ValueError, message):
                    runtime.quantize_tensor_proto(source, plan)

    def test_invalid_inputs_and_payload(self):
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, 1)
        for value in (numpy.nan, numpy.inf, -numpy.inf):
            source = numpy_helper.from_array(numpy.array([value], dtype=numpy.float32))
            with self.assertRaisesRegex(ValueError, "finite"):
                runtime.quantize_tensor_proto(source, plan)
        source = numpy_helper.from_array(numpy.array([1], dtype=numpy.int32))
        with self.assertRaisesRegex(ValueError, "FLOAT"):
            runtime.quantize_tensor_proto(source, plan)
        source = numpy_helper.from_array(numpy.array([1], dtype=numpy.float32))
        encoded = runtime.quantize_tensor_proto(source, plan)
        encoded.raw_data = bytes(encoded.raw_data)[:-1]
        with self.assertRaises(ValueError):
            runtime.dequantize_tensor_proto(encoded)

    def test_mutated_profile_names(self):
        source = numpy_helper.from_array(numpy.array([1, 2], dtype=numpy.float32))
        valid = runtime.quantize_tensor_proto(
            source, runtime.make_quantization_plan(QuantizationFormat.INT4, 2)
        )
        for name in ("", "unknown", "QUAROT"):
            with self.subTest(format=name):
                plan = runtime.make_quantization_plan(QuantizationFormat.QUAROT, 2)
                with self.assertRaises(TypeError):
                    plan.format = name
                with self.assertRaises(TypeError):
                    runtime.make_quantization_plan(name, 2)
                with self.assertRaisesRegex(ValueError, "Unknown quantization format"):
                    runtime.parse_quantization_format(name)
                encoded = onnx.EncodedValueProto()
                encoded.ParseFromString(valid.SerializeToString())
                encoded.struct_type.name = f"onnx_light.quantization.v1/{name}"
                with self.assertRaisesRegex(ValueError, "Unknown quantization format"):
                    runtime.dequantize_tensor_proto(encoded)
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, 2)
        for value in (0, 999, "int4"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    plan.format = value
                with self.assertRaises(TypeError):
                    runtime.make_quantization_plan(value, 2)

    def test_run_geometry_and_shared_layout(self):
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, 10, 4)
        runs = plan.runs
        self.assertEqual([(run.layout.count, len(run.blocks)) for run in runs], [(4, 2), (2, 1)])
        runs[0].layout.bits = 5
        blocks = runs[0].blocks
        blocks[0].scale = 0.5
        blocks[1].scale = 2
        runs[0].blocks = blocks
        self.assertEqual(plan.run(0).layout.bits, 4)
        plan.runs = runs
        source = numpy.array([-8, 0, 7.5, 1, -32, 0, 30, 2, 0, 1], dtype=numpy.float32)
        numpy.testing.assert_array_equal(self.roundtrip(source, plan), source)
        self.assertEqual(runtime.make_quantization_plan(QuantizationFormat.INT4, 0).runs, [])

        runs[1].blocks = []
        plan.runs = runs
        with self.assertRaisesRegex(ValueError, "nonempty"):
            runtime.quantize_tensor_proto(numpy_helper.from_array(source), plan)

    def test_loaded_external_tensor(self):
        values = numpy.array([1, 2], dtype=numpy.float32)
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, values.size)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "weights.bin").write_bytes(values.astype("<f4").tobytes())
            source = onnx.TensorProto()
            source.data_type = onnx.TensorProto.FLOAT
            source.dims.append(values.size)
            source.data_location = onnx.TensorProto.EXTERNAL
            location = source.external_data.add()
            location.key = "location"
            location.value = "weights.bin"
            with self.assertRaisesRegex(ValueError, "Load external"):
                runtime.quantize_tensor_proto(source, plan)
            source.load_external_data(directory)
            self.assertEqual(source.data_location, onnx.TensorProto.EXTERNAL)
            original = source.SerializeToString()
            encoded = runtime.quantize_tensor_proto(source, plan)
            self.assertEqual(source.SerializeToString(), original)
        restored = numpy_helper.to_array(runtime.dequantize_tensor_proto(encoded))
        numpy.testing.assert_array_equal(restored, values)

    def test_catalogue_reference(self):
        source = numpy.array([1, 2, 3], dtype=numpy.float32)
        plan = runtime.make_quantization_plan(QuantizationFormat.INT8, source.size)
        encoded = runtime.quantize_tensor_proto(numpy_helper.from_array(source), plan)
        model = onnx.ModelProto()
        declaration = model.struct_types.add()
        declaration.CopyFrom(encoded.struct_type)
        declaration.type_id = 91
        encoded.struct_type = onnx.StructTypeProto(type_ref=91)
        result = runtime.dequantize_tensor_proto(encoded, model=model)
        numpy.testing.assert_array_equal(numpy_helper.to_array(result), source)
        with self.assertRaises(ValueError):
            runtime.dequantize_tensor_proto(encoded)

    def test_block_copies_and_replacement(self):
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, 4)
        run = plan.run(0)
        block = run.block(0)
        block.scale = 2
        run.layout.bits = 5
        self.assertEqual(plan.run(0).block(0).scale, 1)
        self.assertEqual(plan.run(0).layout.bits, 4)
        run.set_block(0, block)
        plan.set_run(0, run)
        self.assertEqual(plan.run(0).block(0).scale, 2)
        self.assertEqual(plan.run(0).layout.bits, 5)
        plan.runs = []
        block.scale = 3
        self.assertEqual(block.scale, 3)
        with self.assertRaisesRegex(ValueError, "index"):
            plan.set_run(0, run)
        run.blocks = []
        with self.assertRaisesRegex(ValueError, "index"):
            run.set_block(0, block)
        with self.assertRaisesRegex(ValueError, "index"):
            run.block(0)

    def test_codebook_levels_against_numpy_reference(self):
        source = numpy.linspace(-2, 2, 129, dtype=numpy.float64)
        for name in (
            QuantizationFormat.NF4,
            QuantizationFormat.IQ4_NL,
            QuantizationFormat.LOG,
            QuantizationFormat.MXFP4,
            QuantizationFormat.MXFP6,
            QuantizationFormat.FP8_E4M3,
        ):
            with self.subTest(format=name):
                plan = runtime.make_quantization_plan(name, source.size, source.size)
                run = plan.run(0)
                block = run.block(0)
                block.scale = 0.02 if name == QuantizationFormat.IQ4_NL else 2
                run.set_block(0, block)
                plan.set_run(0, run)
                table = numpy.array(block.codebook) * block.scale
                indices = numpy.abs(source[:, None] - table[None, :]).argmin(axis=1)
                expected = table[indices]
                numpy.testing.assert_allclose(
                    self.roundtrip(source, plan), expected, rtol=0, atol=1e-14
                )

    def test_explicit_column_major_order(self):
        source = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
        plan = runtime.make_quantization_plan(QuantizationFormat.COLUMN_MAJOR, source.size)
        plan.permutation = [0, 3, 1, 4, 2, 5]
        encoded = runtime.quantize_tensor_proto(numpy_helper.from_array(source), plan)
        self.assertEqual(
            bytes(encoded.raw_data)[-source.nbytes :], source.T.copy().astype("<f4").tobytes()
        )
        numpy.testing.assert_array_equal(self.roundtrip(source, plan), source)

    def test_affine_against_numpy_reference(self):
        source = numpy.linspace(-2, 8, 201, dtype=numpy.float64)
        plan = runtime.make_quantization_plan(QuantizationFormat.GPTQ, source.size, source.size)
        run = plan.run(0)
        block = run.block(0)
        block.scale = 0.125
        block.zero_point = 3
        block.offset = 3.141
        run.set_block(0, block)
        plan.set_run(0, run)
        codes = numpy.clip(
            numpy.rint((source - block.offset) / block.scale + block.zero_point), 0, 15
        )
        expected = (codes - block.zero_point) * block.scale + block.offset
        numpy.testing.assert_allclose(self.roundtrip(source, plan), expected, rtol=0, atol=1e-15)

    def test_cast_precision_against_numpy_reference(self):
        source = numpy.linspace(-3, 3, 201, dtype=numpy.float32)
        plan = runtime.make_quantization_plan(
            QuantizationFormat.TILED_FLOAT, source.size, source.size
        )
        run = plan.run(0)
        run.layout.cast_type = onnx.TensorProto.FLOAT16
        plan.set_run(0, run)
        expected = source.astype(numpy.float16).astype(numpy.float32)
        numpy.testing.assert_array_equal(self.roundtrip(source, plan), expected)


if __name__ == "__main__":
    unittest.main()
