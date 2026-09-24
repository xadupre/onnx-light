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
import onnx_light.onnx.checker as checker
from onnx_light.ext_test_case import import_or_skip
from onnx_light.onnx import helper, numpy_helper

runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels", "runtime")
QuantizationFormat = runtime.QuantizationFormat


class TestQuantizedValues(unittest.TestCase):
    def test_graph_quantize_dequantize(self):
        values = numpy.array([-8, -4, 0, 7, -16, 0, 8, 14], dtype=numpy.float32)
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, values.size, 4)
        destination = onnx.TypeProto()
        destination.struct_type.CopyFrom(runtime.make_quantization_type(plan))
        encode = helper.make_node("Quantize", ["X"], ["Q"], domain="ai.rt", type=destination)
        self.assertEqual(
            helper.get_attribute_value(encode.attribute[0]).SerializeToString(),
            destination.SerializeToString(),
        )
        with self.assertRaises(TypeError):
            helper.make_attribute("type", destination, attr_type=onnx.AttributeProto.INT)
        decode = helper.make_node(
            "Dequantize", ["Q"], ["Y"], domain="ai.rt", dtype=onnx.TensorProto.DOUBLE
        )
        graph = helper.make_graph(
            [encode, decode],
            "codecs",
            [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [8])],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, [8])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("ai.rt", 1)]
        )
        context = runtime.RuntimeContext()
        context.set(
            "X",
            runtime.tensor_from_numpy("X", onnx.TensorProto.FLOAT, [8], values.view(numpy.uint8)),
        )
        runtime.RuntimeSession(model).run(context)
        actual = numpy.from_dlpack(context.get("Y"))
        self.assertEqual(actual.dtype, numpy.float64)
        numpy.testing.assert_array_equal(actual, values)
        model.graph.node[0].input.append("scales")
        model.graph.input.append(
            helper.make_tensor_value_info("scales", onnx.TensorProto.DOUBLE, [2])
        )
        scales = numpy.array([2, 4], dtype=numpy.float64)
        context.set(
            "scales",
            runtime.tensor_from_numpy(
                "scales", onnx.TensorProto.DOUBLE, [2], scales.view(numpy.uint8)
            ),
        )
        runtime.RuntimeSession(model).run(context)
        numpy.testing.assert_array_equal(
            numpy.from_dlpack(context.get("Y")), [-8, -4, 0, 8, -16, 0, 8, 16]
        )

    def make_encoded_initializer_model(self):
        """Returns a model with a referenced encoded initializer and its source values."""
        values = numpy.array([-1, 0, 1], dtype=numpy.float32)
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, 3)
        encoded = runtime.quantize_tensor_proto(numpy_helper.from_array(values, name="Q"), plan)
        declaration = onnx.StructTypeProto()
        declaration.CopyFrom(encoded.struct_type)
        declaration.type_id = 1
        encoded.struct_type = onnx.StructTypeProto(type_ref=1)
        decode = helper.make_node(
            "Dequantize", ["Q"], ["Y"], domain="ai.rt", dtype=onnx.TensorProto.FLOAT
        )
        graph = helper.make_graph(
            [decode],
            "encoded_initializer",
            [],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3])],
        )
        graph.encoded_initializer.append(encoded)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("ai.rt", 1)])
        model.struct_types.append(declaration)
        return model, values

    def test_graph_encoded_initializer_and_catalogue(self):
        model, values = self.make_encoded_initializer_model()
        checker.check_model(model)
        context = runtime.RuntimeContext()
        runtime.RuntimeSession(model).run(context)
        numpy.testing.assert_array_equal(numpy.from_dlpack(context.get("Y")), values)

    def test_checker_encoded_initializer_failures(self):
        for failure in (
            "empty_name",
            "duplicate",
            "tensor_collision",
            "unknown_reference",
            "duplicate_type_id",
            "truncated_payload",
            "node_output_collision",
        ):
            with self.subTest(failure=failure):
                model, values = self.make_encoded_initializer_model()
                encoded = model.graph.encoded_initializer[0]
                if failure == "empty_name":
                    encoded.name = ""
                elif failure == "duplicate":
                    model.graph.encoded_initializer.append(encoded)
                elif failure == "tensor_collision":
                    model.graph.initializer.append(numpy_helper.from_array(values, name="Q"))
                elif failure == "unknown_reference":
                    encoded.struct_type = onnx.StructTypeProto(type_ref=99)
                elif failure == "duplicate_type_id":
                    model.struct_types.append(model.struct_types[0])
                elif failure == "truncated_payload":
                    encoded.raw_data = bytes(encoded.raw_data)[:-1]
                elif failure == "node_output_collision":
                    model.graph.node[0].output.clear()
                    model.graph.node[0].output.append("Q")
                with self.assertRaises(checker.ValidationError):
                    checker.check_model(model)

    def test_checker_structured_graph_output(self):
        model, _ = self.make_encoded_initializer_model()
        output_type = onnx.TypeProto()
        output_type.struct_type = onnx.StructTypeProto(type_ref=1)
        model.graph.output.append(helper.make_value_info("Q", output_type))
        checker.check_model(model)
        model.graph.output[-1].type.struct_type = onnx.StructTypeProto(type_ref=99)
        with self.assertRaises(checker.ValidationError):
            checker.check_model(model)

    def test_checker_nested_encoded_initializer(self):
        model, _ = self.make_encoded_initializer_model()
        conditional = helper.make_node(
            "If", ["condition"], ["Y"], then_branch=model.graph, else_branch=model.graph
        )
        graph = helper.make_graph(
            [conditional],
            "nested_encoded",
            [],
            list(model.graph.output),
            initializer=[helper.make_tensor("condition", onnx.TensorProto.BOOL, [], [True])],
        )
        model.graph = graph
        model.opset_import.append(helper.make_opsetid("", 21))
        checker.check_model(model)
        branch = model.graph.node[0].attribute[0].g
        branch.encoded_initializer[0].struct_type = onnx.StructTypeProto(type_ref=99)
        with self.assertRaises(checker.ValidationError):
            checker.check_model(model)

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
        self.assertEqual(len(formats), 43)
        self.assertTrue(all(isinstance(value, QuantizationFormat) for value in formats))
        self.assertEqual(formats[0], QuantizationFormat.INT8)
        self.assertEqual(formats[-1], QuantizationFormat.ORT_MATMULNBITS_INT8)
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
        source = numpy.array([-1, -1, 0, 0, 1, 1, 1, 1], dtype=numpy.float32).reshape(4, 2)
        for name in runtime.quantization_formats():
            with self.subTest(format=name):
                plan = (
                    runtime.make_matmul_nbits_plan(name, 4, 2, 16)
                    if name
                    in {
                        QuantizationFormat.ORT_MATMULNBITS_INT2,
                        QuantizationFormat.ORT_MATMULNBITS_INT4,
                        QuantizationFormat.ORT_MATMULNBITS_INT8,
                    }
                    else runtime.make_quantization_plan(name, source.size, 4)
                )
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

    def test_encoding_rejects_affine_reconstruction_overflow(self):
        for dtype, exponent in ((numpy.float16, 15), (numpy.float32, 127), (numpy.float64, 1023)):
            scale = float(2**exponent)
            formats = [QuantizationFormat.INT4]
            if dtype != numpy.float64:
                formats += [
                    QuantizationFormat.ORT_MATMULNBITS_INT2,
                    QuantizationFormat.ORT_MATMULNBITS_INT4,
                    QuantizationFormat.ORT_MATMULNBITS_INT8,
                ]
            for format_value in formats:
                with self.subTest(dtype=dtype, format=format_value):
                    values = numpy.array([[1.5 * scale]], dtype=dtype)
                    source = numpy_helper.from_array(values)
                    original = source.SerializeToString()
                    plan = (
                        runtime.make_quantization_plan(format_value, 1)
                        if format_value == QuantizationFormat.INT4
                        else runtime.make_matmul_nbits_plan(format_value, 1, 1, 16)
                    )
                    run = plan.run(0)
                    block = run.block(0)
                    block.scale = scale
                    block.zero_point = 0
                    run.set_block(0, block)
                    plan.set_run(0, run)
                    with self.assertRaisesRegex(ValueError, "overflow|nonfinite"):
                        runtime.quantize_tensor_proto(source, plan)
                    self.assertEqual(source.SerializeToString(), original)
                    block.scale = scale / 2
                    run.set_block(0, block)
                    plan.set_run(0, run)
                    encoded = runtime.quantize_tensor_proto(source, plan)
                    decoded = runtime.dequantize_tensor_proto(encoded)
                    numpy.testing.assert_array_equal(numpy_helper.to_array(decoded), values)

    def test_proto_metadata_presence(self):
        values = numpy.array([[1], [2]], dtype=numpy.float32)
        plans = [
            runtime.make_quantization_plan(QuantizationFormat.INT4, values.size),
            runtime.make_matmul_nbits_plan(QuantizationFormat.ORT_MATMULNBITS_INT4, 2, 1, 16),
        ]
        for plan in plans:
            for name in (None, "", "weight"):
                for doc in (None, "", "weight documentation"):
                    with self.subTest(format=plan.format, name=name, doc=doc):
                        source = numpy_helper.from_array(values)
                        source.ClearField("name")
                        if name is not None:
                            source.name = name
                        if doc is not None:
                            source.doc_string = doc
                        original = source.SerializeToString()
                        encoded = runtime.quantize_tensor_proto(source, plan)
                        self.assertEqual(source.SerializeToString(), original)
                        self.assertEqual(encoded.has_name(), name is not None)
                        self.assertEqual(encoded.has_doc_string(), doc is not None)
                        loaded = onnx.EncodedValueProto()
                        loaded.ParseFromString(encoded.SerializeToString())
                        decoded = runtime.dequantize_tensor_proto(loaded)
                        self.assertEqual(decoded.has_name(), name is not None)
                        self.assertEqual(decoded.has_doc_string(), doc is not None)
                        self.assertEqual(decoded.name, name if name is not None else "")
                        self.assertEqual(decoded.doc_string, doc if doc is not None else "")
                        numpy.testing.assert_array_equal(numpy_helper.to_array(decoded), values)

    def test_explicit_none_model(self):
        values = numpy.array([[1], [2]], dtype=numpy.float32)
        plans = [
            runtime.make_quantization_plan(QuantizationFormat.INT4, values.size),
            runtime.make_matmul_nbits_plan(QuantizationFormat.ORT_MATMULNBITS_INT4, 2, 1, 16),
        ]
        for plan in plans:
            with self.subTest(format=plan.format):
                encoded = runtime.quantize_tensor_proto(numpy_helper.from_array(values), plan)
                numpy.testing.assert_array_equal(
                    numpy_helper.to_array(runtime.dequantize_tensor_proto(encoded, model=None)),
                    values,
                )
                numpy.testing.assert_array_equal(
                    numpy.from_dlpack(runtime.dequantize_tensor(encoded, model=None)), values
                )
                if plan.format == QuantizationFormat.ORT_MATMULNBITS_INT4:
                    inputs = runtime.export_matmul_nbits_inputs(encoded, model=None)
                    self.assertEqual(
                        inputs.weights.SerializeToString(),
                        runtime.export_matmul_nbits_inputs(encoded).weights.SerializeToString(),
                    )
                encoded.struct_type = onnx.StructTypeProto(type_ref=123)
                with self.assertRaises(ValueError):
                    runtime.dequantize_tensor_proto(encoded, model=None)
                with self.assertRaises(ValueError):
                    runtime.dequantize_tensor(encoded, model=None)
                if plan.format == QuantizationFormat.ORT_MATMULNBITS_INT4:
                    with self.assertRaises(ValueError):
                        runtime.export_matmul_nbits_inputs(encoded, model=None)

    def test_loaded_empty_external_tensor(self):
        plan = runtime.make_quantization_plan(QuantizationFormat.INT4, 0)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "empty.bin").write_bytes(b"")
            source = onnx.TensorProto()
            source.data_type = onnx.TensorProto.FLOAT
            source.dims.append(0)
            source.data_location = onnx.TensorProto.EXTERNAL
            location = source.external_data.add()
            location.key = "location"
            location.value = "empty.bin"
            with self.assertRaisesRegex(ValueError, "Load external"):
                runtime.quantize_tensor_proto(source, plan)
            source.load_external_data(directory)
            self.assertTrue(source.HasField("raw_data"))
            self.assertEqual(bytes(source.raw_data), b"")
            original = source.SerializeToString()
            encoded = runtime.quantize_tensor_proto(source, plan)
            self.assertEqual(source.SerializeToString(), original)
        decoded = runtime.dequantize_tensor_proto(encoded)
        numpy.testing.assert_array_equal(
            numpy_helper.to_array(decoded), numpy.empty((0,), dtype=numpy.float32)
        )

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

    def test_cast_type_is_validated_for_every_method(self):
        values = numpy.array([-1, 0, 1], dtype=numpy.float32)
        source = numpy_helper.from_array(values)
        for format_value in (
            QuantizationFormat.INT4,
            QuantizationFormat.NF4,
            QuantizationFormat.TILED_FLOAT,
        ):
            plan = runtime.make_quantization_plan(format_value, values.size)
            for cast_type in (-1, 0, onnx.TensorProto.INT8, onnx.TensorProto.STRING, 2**31 - 1):
                with self.subTest(format=format_value, cast_type=cast_type):
                    run = plan.run(0)
                    run.layout.cast_type = cast_type
                    plan.set_run(0, run)
                    with self.assertRaisesRegex(ValueError, "cast_type"):
                        runtime.quantize_tensor_proto(source, plan)
            for cast_type in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.DOUBLE,
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.BFLOAT16,
            ):
                with self.subTest(format=format_value, cast_type=cast_type):
                    run = plan.run(0)
                    run.layout.cast_type = cast_type
                    plan.set_run(0, run)
                    numpy.testing.assert_array_equal(self.roundtrip(values, plan), values)

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

    def make_ort_inputs(self, bits, mode, dtype, k=35, n=3, block_size=16):
        """Returns weights and encoded ORT inputs with independent column/block parameters."""
        from onnx_light.onnx_core.quantization import (
            export_matmul_nbits_inputs,
            make_matmul_nbits_plan,
            parse_quantization_format,
            quantize_tensor_proto,
        )

        format_value = parse_quantization_format(f"ort_matmulnbits_int{bits}")
        plan = make_matmul_nbits_plan(format_value, k, n, block_size)
        groups = (k + block_size - 1) // block_size
        run = plan.run(0)
        blocks = run.blocks
        for index, block in enumerate(blocks):
            block.scale = ((index % groups) + 1) * (0.5 if index // groups != 1 else -0.5)
            if mode != "implicit":
                block.zero_point = index % (1 << bits) + (0.25 if mode == "floating" else 0)
        run.blocks = blocks
        plan.set_run(0, run)
        source = numpy.empty((k, n), dtype=dtype)
        for row in range(k):
            for column in range(n):
                block = blocks[column * groups + row // block_size]
                code = (row + column) % (1 << bits)
                source[row, column] = (code - block.zero_point) * block.scale
        encoded = quantize_tensor_proto(numpy_helper.from_array(source), plan)
        return source, encoded, export_matmul_nbits_inputs(encoded)

    def test_ort_input_shapes_modes_serialization_and_ownership(self):
        for bits in (2, 4, 8):
            for mode in ("implicit", "packed", "floating"):
                for dtype in (numpy.float32, numpy.float16):
                    with self.subTest(bits=bits, mode=mode, dtype=dtype):
                        source, encoded, inputs = self.make_ort_inputs(bits, mode, dtype)
                        self.assertEqual(
                            (inputs.k, inputs.n, inputs.bits, inputs.block_size),
                            (35, 3, bits, 16),
                        )
                        self.assertEqual(tuple(inputs.weights.dims), (3, 3, 16 * bits // 8))
                        self.assertEqual(tuple(inputs.scales.dims), (3, 3))
                        self.assertEqual(numpy_helper.to_array(inputs.scales).dtype, dtype)
                        if mode == "implicit":
                            self.assertIsNone(inputs.zero_points)
                        else:
                            shape = (3, (3 * bits + 7) // 8) if mode == "packed" else (3, 3)
                            self.assertEqual(tuple(inputs.zero_points.dims), shape)
                            self.assertEqual(
                                numpy_helper.to_array(inputs.zero_points).dtype,
                                numpy.uint8 if mode == "packed" else dtype,
                            )
                        loaded = onnx.EncodedValueProto()
                        loaded.ParseFromString(encoded.SerializeToString())
                        numpy.testing.assert_array_equal(
                            numpy_helper.to_array(runtime.dequantize_tensor_proto(loaded)), source
                        )
                        original = bytes(inputs.weights.raw_data)
                        model = onnx.ModelProto()
                        model.struct_types.add().CopyFrom(loaded.struct_type)
                        model.struct_types[0].type_id = 101
                        loaded.struct_type = onnx.StructTypeProto(type_ref=101)
                        exported = runtime.export_matmul_nbits_inputs(loaded, model=model)
                        self.assertEqual(bytes(exported.weights.raw_data), original)
                        del encoded, loaded
                        gc.collect()
                        self.assertEqual(bytes(inputs.weights.raw_data), original)

    def test_matrix_shape_is_a_mutable_native_shape(self):
        from onnx_light.onnx_core.quantization import Shape
        from onnx_light.onnx_core import shape_inference

        self.assertIs(Shape, shape_inference.Shape)
        plan = runtime.make_matmul_nbits_plan(QuantizationFormat.ORT_MATMULNBITS_INT4, 2, 1, 16)
        self.assertIsInstance(plan.matrix_shape, Shape)
        self.assertEqual(list(plan.matrix_shape), [2, 1])
        source = numpy_helper.from_array(numpy.array([[1], [2]], dtype=numpy.float32))
        for dims in ([2, 1], (2, 1), Shape([2, 1])):
            plan.matrix_shape = dims
            self.assertEqual(plan.matrix_shape, Shape([2, 1]))
            runtime.quantize_tensor_proto(source, plan)
        assigned = Shape([2, 1])
        plan.matrix_shape = assigned
        assigned[0] = 9
        self.assertEqual(list(plan.matrix_shape), [2, 1])
        for invalid in (["K", 1], [2.5, 1], None):
            with self.assertRaises(TypeError):
                plan.matrix_shape = invalid
            self.assertEqual(list(plan.matrix_shape), [2, 1])
        with self.assertRaises(ValueError):
            plan.matrix_shape = [1] * 17
        self.assertEqual(list(plan.matrix_shape), [2, 1])
        view = plan.matrix_shape
        view[0] = 3
        with self.assertRaisesRegex(ValueError, "shape"):
            runtime.quantize_tensor_proto(source, plan)
        del plan
        gc.collect()
        self.assertEqual(list(view), [3, 1])
        view[0] = 2
        self.assertEqual(list(view), [2, 1])

    def test_ort_rejects_nonzero_padding(self):
        for bits in (2, 4, 8):
            for mode in ("implicit", "packed", "floating"):
                source, encoded, inputs = self.make_ort_inputs(bits, mode, numpy.float32)
                weights_per_column = len(inputs.weights.raw_data) // inputs.n
                zero_offset = len(inputs.weights.raw_data) + len(inputs.scales.raw_data)
                for column in range(inputs.n):
                    corruptions = [
                        (column * weights_per_column * 8 + inputs.k * bits, "weight padding"),
                        ((column + 1) * weights_per_column * 8 - 1, "weight padding"),
                    ]
                    if mode == "packed" and bits != 8:
                        zeros_per_column = len(inputs.zero_points.raw_data) // inputs.n
                        corruptions.append(
                            (
                                (zero_offset + (column + 1) * zeros_per_column) * 8 - 1,
                                "zero-point padding",
                            )
                        )
                    for bit, message in corruptions:
                        with self.subTest(bits=bits, mode=mode, column=column, bit=bit):
                            corrupt = onnx.EncodedValueProto()
                            corrupt.CopyFrom(encoded)
                            raw = bytearray(encoded.raw_data)
                            self.assertEqual(raw[bit // 8] & (1 << (bit % 8)), 0)
                            raw[bit // 8] |= 1 << (bit % 8)
                            corrupt.raw_data = bytes(raw)
                            for decode in (
                                runtime.dequantize_tensor,
                                runtime.dequantize_tensor_proto,
                                runtime.export_matmul_nbits_inputs,
                            ):
                                with self.assertRaisesRegex(ValueError, message):
                                    decode(corrupt)
                numpy.testing.assert_array_equal(
                    numpy_helper.to_array(runtime.dequantize_tensor_proto(encoded)), source
                )

    def test_ort_matmul_nbits_interoperability(self):
        onnxruntime = import_or_skip("onnxruntime")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        for bits in (2, 4, 8):
            # ORT CPU has no 8-bit unpacked-compute path for floating zero points.
            modes = ("implicit", "packed") if bits == 8 else ("implicit", "packed", "floating")
            for mode in modes:
                for dtype in (numpy.float32, numpy.float16):
                    for k, block_size in ((35, 16), (65, 32), (64, 64)):
                        with self.subTest(
                            bits=bits, mode=mode, dtype=dtype, k=k, block_size=block_size
                        ):
                            source, _, inputs = self.make_ort_inputs(
                                bits, mode, dtype, k=k, block_size=block_size
                            )
                            initializers = [inputs.weights, inputs.scales]
                            names = ["A", "B", "scales"]
                            if inputs.zero_points is not None:
                                initializers.append(inputs.zero_points)
                                names.append("zero_points")
                            node = helper.make_node(
                                "MatMulNBits",
                                names,
                                ["Y"],
                                domain="com.microsoft",
                                K=inputs.k,
                                N=inputs.n,
                                bits=inputs.bits,
                                block_size=inputs.block_size,
                                accuracy_level=1,
                            )
                            elem_type = inputs.scales.data_type
                            graph = helper.make_graph(
                                [node],
                                "ort_input_packing",
                                [helper.make_tensor_value_info("A", elem_type, [2, k])],
                                [helper.make_tensor_value_info("Y", elem_type, [2, inputs.n])],
                                initializers,
                            )
                            model = helper.make_model(
                                graph,
                                opset_imports=[
                                    helper.make_opsetid("", 21),
                                    helper.make_opsetid("com.microsoft", 1),
                                ],
                            )
                            model.ir_version = 10
                            session = onnxruntime.InferenceSession(
                                model.SerializeToString(),
                                options,
                                providers=["CPUExecutionProvider"],
                            )
                            a = ((numpy.arange(2 * k).reshape(2, k) % 7 - 3) / 4).astype(dtype)
                            expected = (
                                a.astype(numpy.float32) @ source.astype(numpy.float32)
                            ).astype(dtype)
                            actual = session.run(None, {"A": a})[0]
                            numpy.testing.assert_allclose(
                                actual,
                                expected,
                                rtol=1e-3 if dtype == numpy.float16 else 1e-5,
                                atol=1e-3 if dtype == numpy.float16 else 1e-5,
                            )


if __name__ == "__main__":
    unittest.main()
