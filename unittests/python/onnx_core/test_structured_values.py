# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Regressions for structured types and encoded graph initializers."""

import gc
import unittest

from onnx_light import onnx
from onnx_light.onnx import helper
from onnx_light.onnx.onnx_pb import AffineLayoutProto, EncodedValueProto, StructTypeProto
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_core.shape_inference import ShapesContext


def tensor_type(elem_type=onnx.TensorProto.FLOAT, shape=(2,)):
    """Returns a concrete tensor type."""
    return helper.make_tensor_type_proto(elem_type, list(shape))


def declaration(type_id=1):
    """Returns a one-byte structured record declaration."""
    return StructTypeProto(
        type_id=type_id,
        name="record",
        bit_packing=StructTypeProto.BitPacking(
            dimension=1,
            component=[StructTypeProto.BitPacking.Component(name="code", bit_width=8)],
        ),
    )


def encoded(name="weights", logical=True):
    """Returns two encoded records referencing the model catalogue."""
    value = EncodedValueProto(
        name=name, struct_type=StructTypeProto(type_ref=1), raw_data=b"\x00\xff"
    )
    if logical:
        value.logical_type = tensor_type()
    return value


class TestStructuredValues(unittest.TestCase):
    def test_nested_proto_roundtrip(self):
        record = StructTypeProto(
            type_id=2**40,
            name="nested",
            structure=StructTypeProto.Structure(
                field=[
                    StructTypeProto.Structure.Field(
                        name="typed", type=tensor_type(onnx.TensorProto.UINT8)
                    ),
                    StructTypeProto.Structure.Field(
                        name="constant",
                        constant=helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [0.5]),
                    ),
                    StructTypeProto.Structure.Field(
                        name="array",
                        type=onnx.TypeProto(
                            struct_type=StructTypeProto(
                                array=StructTypeProto.Array(
                                    dimension=3,
                                    element_type=onnx.TypeProto(
                                        struct_type=StructTypeProto(type_ref=1)
                                    ),
                                )
                            )
                        ),
                    ),
                ]
            ),
            metadata_props=[onnx.StringStringEntryProto(key="format", value="example")],
            decoder=onnx.FunctionProto(name="decode", domain="example"),
            encoder=onnx.FunctionProto(name="encode", domain="example"),
            doc_string="Nested typed and constant fields.",
        )
        model = onnx.ModelProto(
            graph=onnx.GraphProto(name="g", encoded_initializer=[encoded()]),
            struct_types=[declaration(), record],
        )
        restored = onnx.ModelProto()
        restored.ParseFromString(model.SerializeToString())
        self.assertEqual(restored.SerializeToString(), model.SerializeToString())
        self.assertTrue(restored.HasField("struct_types"))
        self.assertTrue(restored.graph.HasField("encoded_initializer"))
        self.assertEqual(restored.struct_types[1].type_id, 2**40)
        fields = restored.struct_types[1].structure.field
        self.assertEqual(fields[0].WhichOneof("content"), "type")
        self.assertEqual(fields[1].WhichOneof("content"), "constant")
        self.assertEqual(list(fields[1].constant.float_data), [0.5])
        self.assertEqual(fields[2].type.WhichOneof("value"), "struct_type")
        self.assertTrue(fields[2].type.HasField("struct_type"))
        self.assertEqual(fields[2].type.struct_type.array.dimension, 3)
        self.assertEqual(restored.struct_types[1].decoder.name, "decode")
        self.assertEqual(restored.struct_types[1].encoder.name, "encode")
        self.assertEqual(restored.struct_types[1].metadata_props[0].value, "example")
        self.assertEqual(restored.graph.encoded_initializer[0].raw_data, b"\x00\xff")

    def test_presence_and_oneof_replacement(self):
        value = StructTypeProto()
        self.assertIsNone(value.WhichOneof("kind"))
        self.assertFalse(value.HasField("type_ref"))
        value.type_ref = 2**40
        self.assertEqual(value.type_ref, 2**40)
        self.assertEqual(value.WhichOneof("kind"), "type_ref")
        value.array = StructTypeProto.Array(dimension=2, element_type=tensor_type())
        self.assertFalse(value.HasField("type_ref"))
        self.assertEqual(value.WhichOneof("kind"), "array")
        value.type_ref = 1
        self.assertFalse(value.HasField("array"))
        value.type_ref = None
        self.assertIsNone(value.WhichOneof("kind"))
        field = StructTypeProto.Structure.Field(name="x", type=tensor_type())
        field.constant = helper.make_tensor("c", onnx.TensorProto.INT64, [], [1])
        self.assertFalse(field.HasField("type"))
        self.assertTrue(field.HasField("constant"))
        with self.assertRaises(ValueError):
            field.WhichOneof("missing")
        with self.assertRaises(AttributeError):
            value.HasField("missing")

    def test_affine_and_external_proto_roundtrip(self):
        layout = AffineLayoutProto(
            storage_type=onnx.TensorProto.UINT4,
            scale=helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [0.5]),
            zero_point=helper.make_tensor("zero", onnx.TensorProto.UINT4, [], [0]),
            axis=-1,
            block_size=2**40,
        )
        self.assertTrue(layout.HasField("axis"))
        self.assertEqual(layout.axis, -1)
        self.assertEqual(layout.block_size, 2**40)
        layout.axis = None
        self.assertFalse(layout.HasField("axis"))
        value = EncodedValueProto(
            name="external",
            affine=layout,
            logical_type=tensor_type(),
            data_location=onnx.TensorProto.EXTERNAL,
            external_data=[
                onnx.StringStringEntryProto(key="location", value="weights.bin"),
                onnx.StringStringEntryProto(key="offset", value="0"),
                onnx.StringStringEntryProto(key="length", value="1"),
            ],
        )
        restored = EncodedValueProto()
        restored.ParseFromString(value.SerializeToString())
        self.assertEqual(restored.WhichOneof("layout"), "affine")
        self.assertTrue(restored.HasField("external_data"))
        self.assertEqual(restored.external_data[2].value, "1")
        self.assertEqual(restored.data_location, onnx.TensorProto.DataLocation.EXTERNAL)
        restored.struct_type = StructTypeProto(type_ref=1)
        self.assertFalse(restored.HasField("affine"))
        self.assertEqual(restored.WhichOneof("layout"), "struct_type")

    def test_context_type_and_layout(self):
        context = ShapesContext()
        context.set_struct_types([declaration()])
        self.assertEqual(context.struct_types()[0].type_id, 1)
        self.assertEqual(context.resolve_struct_type(StructTypeProto(type_ref=1)).name, "record")
        context.set_encoded_value("weights", encoded())
        self.assertTrue(context.has_encoded_value("weights"))
        self.assertTrue(context.has_type("weights"))
        self.assertEqual(context.get_encoded_value("weights").raw_data, b"\x00\xff")
        copied = context.get_encoded_value("weights")
        copied.raw_data = b"\x01\x02"
        self.assertEqual(context.get_encoded_value("weights").raw_data, b"\x00\xff")
        self.assertEqual(context.get_type("weights").WhichOneof("value"), "struct_type")
        self.assertTrue(context.has("weights"))
        self.assertEqual(
            context.get_encoded_value("weights").logical_type.tensor_type.elem_type,
            onnx.TensorProto.DataType.FLOAT,
        )
        layout = context.get_encoded_layout("weights")
        self.assertEqual(
            (layout.element_bits, layout.payload_bytes, layout.record_count), (8, 2, 2)
        )
        self.assertEqual(
            context.resolve_struct_type(context.get_type("weights").struct_type).type_id, 1
        )
        self.assertFalse(layout.external)
        self.assertTrue(layout.content_verified)
        context.set_type("record", onnx.TypeProto(struct_type=StructTypeProto(type_ref=1)))
        self.assertEqual(context.get_type("record").struct_type.type_ref, 1)
        with self.assertRaises(IndexError):
            context.get_type("missing")
        context.clear()
        self.assertFalse(context.has_type("record"))
        self.assertFalse(context.has_encoded_value("weights"))
        self.assertEqual(len(context.struct_types()), 1)
        context.set_struct_types([])
        self.assertEqual(len(context.struct_types()), 0)

    def test_context_rejects_invalid_encoding(self):
        context = ShapesContext()
        context.set_struct_types([declaration()])
        unknown = encoded()
        unknown.struct_type.type_ref = 999
        with self.assertRaises(ValueError):
            context.set_encoded_value("unknown", unknown)
        self.assertFalse(context.has_encoded_value("unknown"))
        with self.assertRaises(ValueError):
            context.set_struct_types([declaration(), declaration()])
        self.assertEqual(len(context.struct_types()), 1)

    def test_context_result_lifetimes(self):
        context = ShapesContext()
        resolved = context.resolve_struct_type(
            StructTypeProto(
                array=StructTypeProto.Array(
                    dimension=1, element_type=tensor_type(onnx.TensorProto.UINT8, [])
                )
            )
        )
        context.set_struct_types([declaration()])
        context.set_encoded_value("weights", encoded())
        layout = context.get_encoded_layout("weights")
        root = context.resolve_struct_type(context.get_type("weights").struct_type)
        owned = context.get_encoded_value("weights")
        self.assertFalse(hasattr(layout, "root"))
        self.assertFalse(hasattr(layout, "affine"))
        context.clear()
        context.set_struct_types([declaration(2)])
        replacement = encoded(logical=False)
        replacement.struct_type.type_ref = 2
        replacement.raw_data = b"\x01\x02\x03\x04"
        context.set_encoded_value("weights", replacement)
        replacement.raw_data = b"\x05"
        context.set_encoded_value("weights", replacement)
        del context
        gc.collect()
        self.assertEqual(resolved.array.dimension, 1)
        self.assertEqual(root.type_id, 1)
        self.assertEqual(owned.raw_data, b"\x00\xff")
        self.assertEqual(
            (layout.element_bits, layout.payload_bytes, layout.record_count), (8, 2, 2)
        )

    def test_builder_roundtrip(self):
        builder = GraphBuilder("encoded")
        builder.make_struct_type(declaration())
        name = builder.make_encoded_initializer(encoded())
        self.assertEqual(name, "weights")
        self.assertEqual(builder.encoded_initializers()[0].name, name)
        layout = builder.shapes.get_encoded_layout(name)
        self.assertEqual((layout.payload_bytes, layout.record_count), (2, 2))
        output = builder.op.Identity(name)
        builder.out(output)
        model = builder.to_model()
        self.assertEqual(len(model.struct_types), 1)
        self.assertEqual(model.graph.encoded_initializer[0].raw_data, b"\x00\xff")
        self.assertEqual(model.graph.output[0].type.struct_type.type_ref, 1)
        self.assertEqual(
            model.graph.encoded_initializer[0].logical_type.tensor_type.elem_type,
            onnx.TensorProto.DataType.FLOAT,
        )
        restored = onnx.ModelProto()
        restored.ParseFromString(model.SerializeToString())
        rebuilt = GraphBuilder(restored).to_model()
        self.assertEqual(
            rebuilt.struct_types[0].SerializeToString(), model.struct_types[0].SerializeToString()
        )
        self.assertEqual(
            rebuilt.graph.encoded_initializer[0].SerializeToString(),
            model.graph.encoded_initializer[0].SerializeToString(),
        )

    def test_structured_identity_without_logical_tensor(self):
        builder = GraphBuilder("structured")
        builder.make_struct_type(declaration())
        name = builder.make_encoded_initializer(encoded(logical=False))
        output = builder.op.Identity(name)
        builder.out(output)
        model = builder.to_model()
        self.assertEqual(model.graph.output[0].type.WhichOneof("value"), "struct_type")
        self.assertEqual(model.graph.output[0].type.struct_type.type_ref, 1)

    def test_standard_export_rejects_structured_values(self):
        catalogue = GraphBuilder("catalogue")
        catalogue.make_struct_type(declaration())
        with self.assertRaisesRegex(ValueError, "standard ONNX|structured"):
            catalogue.to_standard_model()
        affine = GraphBuilder("affine")
        affine.make_encoded_initializer(
            EncodedValueProto(
                name="weights",
                affine=AffineLayoutProto(
                    storage_type=onnx.TensorProto.UINT8,
                    scale=helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [1.0]),
                ),
                logical_type=tensor_type(),
                raw_data=b"\x01\x02",
            )
        )
        with self.assertRaisesRegex(ValueError, "standard ONNX|encoded"):
            affine.to_standard_model()
        structured = GraphBuilder("types")
        structured.make_input(
            onnx.ValueInfoProto(name="x", type=onnx.TypeProto(struct_type=StructTypeProto()))
        )
        with self.assertRaisesRegex(ValueError, "standard ONNX|structured"):
            structured.to_standard_model()

    def test_standard_tensor_regression(self):
        builder = GraphBuilder("ordinary")
        value = builder.inp("x", onnx.TensorProto.FLOAT, [2])
        output = builder.op.Identity(value)
        builder.out(output)
        model = builder.to_standard_model()
        self.assertEqual(len(model.struct_types), 0)
        self.assertEqual(len(model.graph.encoded_initializer), 0)
        self.assertEqual(model.graph.output[0].type.WhichOneof("value"), "tensor_type")
        from onnx_light.onnx.checker import check_model

        check_model(model)

    def test_nested_structured_if_roundtrip(self):
        value_type = onnx.TypeProto(
            sequence_type=onnx.TypeProto.Sequence(
                elem_type=onnx.TypeProto(struct_type=StructTypeProto(type_ref=1))
            )
        )
        branch = helper.make_graph(
            [helper.make_node("Identity", ["records"], ["forwarded"])],
            "branch",
            [],
            [onnx.ValueInfoProto(name="forwarded", type=value_type)],
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "If", ["condition"], ["output"], then_branch=branch, else_branch=branch
                    )
                ],
                "nested",
                [
                    helper.make_tensor_value_info("condition", onnx.TensorProto.BOOL, []),
                    onnx.ValueInfoProto(name="records", type=value_type),
                ],
                [onnx.ValueInfoProto(name="output", type=value_type)],
            )
        )
        model.struct_types = [declaration()]
        context = ShapesContext()
        context.compute_shape_model(model)
        self.assertEqual(
            context.get_type("output").SerializeToString(), value_type.SerializeToString()
        )
        exported = GraphBuilder(model).to_model()
        self.assertEqual(
            exported.graph.output[0].type.SerializeToString(), value_type.SerializeToString()
        )
        restored = onnx.ModelProto()
        restored.ParseFromString(exported.SerializeToString())
        context.compute_shape_model(restored)
        self.assertEqual(
            context.get_type("output").SerializeToString(), value_type.SerializeToString()
        )

    def test_encoded_default_preserves_public_input(self):
        builder = GraphBuilder("default")
        builder.make_struct_type(declaration())
        builder.inp("weights", onnx.TensorProto.FLOAT, ["batch"])
        builder.make_encoded_initializer(encoded())
        builder.out(builder.op.Abs("weights"))
        model = builder.to_model()
        context = ShapesContext()
        context.compute_shape_model(model)
        self.assertFalse(context.has_encoded_value("weights"))
        self.assertFalse(context.has_encoded_value(model.graph.output[0].name))
        context.apply_inferred_shapes_to_model(model)
        self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
        self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[0].dim_param, "batch")
        self.assertEqual(model.graph.encoded_initializer[0].raw_data, b"\x00\xff")


if __name__ == "__main__":
    unittest.main()
