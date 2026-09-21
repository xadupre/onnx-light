"""Checks graph-scoped persistence declarations without storing runtime state."""

import unittest

from onnx_light import onnx
from onnx_light.onnx import helper
from onnx_light.onnx.onnx_pb import PersistentBindingProto
from onnx_light.onnx_core.graph_builder import GraphBuilder
from onnx_light.onnx_lib import PersistentBindingProto as LibraryPersistentBindingProto
from onnx_light.onnx_proto import verify


def tensor_type(dtype=onnx.TensorProto.FLOAT, shape=(2,)):
    """Returns a concrete tensor type."""
    return helper.make_tensor_type_proto(dtype, list(shape))


def structure_type(fields):
    """Returns a structure with literal field names."""
    return onnx.TypeProto(
        struct_type=onnx.StructTypeProto(
            structure=onnx.StructTypeProto.Structure(
                field=[
                    onnx.StructTypeProto.Structure.Field(name=name, type=value)
                    for name, value in fields
                ]
            )
        )
    )


def model_with_binding(input_type=None, output_type=None, **kwargs):
    """Returns a typed root graph with one persistent declaration."""
    input_type = tensor_type() if input_type is None else input_type
    output_type = input_type if output_type is None else output_type
    graph = onnx.GraphProto(
        name="persistent",
        input=[onnx.ValueInfoProto(name="state.in", type=input_type)],
        output=[onnx.ValueInfoProto(name="state.out", type=output_type)],
        node=[helper.make_node("Identity", ["state.in"], ["state.out"])],
        persistent_bindings=[
            PersistentBindingProto(input_name="state.in", output_name="state.out", **kwargs)
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


class TestPersistentBindings(unittest.TestCase):
    def test_repeated_binding_container_api(self):
        graph = onnx.GraphProto()
        self.assertEqual(len(graph.persistent_bindings), 0)
        binding = graph.persistent_bindings.add(input_name="x", output_name="y")
        binding.input_field_path.extend(["a.b", "cache"])
        binding.output_field_path.extend(["a", "b.cache"])
        self.assertEqual(len(graph.persistent_bindings), 1)
        self.assertEqual(graph.persistent_bindings[0].input_name, "x")
        self.assertEqual(list(graph.persistent_bindings[0].input_field_path), ["a.b", "cache"])
        graph.persistent_bindings.append(PersistentBindingProto(input_name="u", output_name="v"))
        graph.persistent_bindings.extend(
            [PersistentBindingProto(input_name="m", output_name="n")]
        )
        self.assertEqual(
            [value.input_name for value in graph.persistent_bindings], ["x", "u", "m"]
        )
        graph.persistent_bindings[1] = PersistentBindingProto(input_name="p", output_name="q")
        self.assertEqual(graph.persistent_bindings[1].output_name, "q")
        del graph.persistent_bindings[:]
        self.assertEqual(len(graph.persistent_bindings), 0)
        graph.persistent_bindings.append(binding)
        self.assertEqual(len(graph.persistent_bindings), 1)
        graph.persistent_bindings.clear()
        self.assertEqual(len(graph.persistent_bindings), 0)
        self.assertEqual(binding.input_name, "x")
        self.assertEqual(list(binding.output_field_path), ["a", "b.cache"])

    def test_exports_and_wire_roundtrip(self):
        self.assertIs(PersistentBindingProto, onnx.PersistentBindingProto)
        self.assertIs(PersistentBindingProto, LibraryPersistentBindingProto)
        binding = PersistentBindingProto(
            input_name="in",
            output_name="out",
            input_field_path=["a.b", "c"],
            output_field_path=["a", "b.c"],
        )
        # Exact field numbers and repeated strings, not dotted-path encoding.
        wire = b"\x0a\x02in\x12\x03out\x1a\x03a.b\x1a\x01c\x22\x01a\x22\x03b.c"
        self.assertEqual(binding.SerializeToString(), wire)
        parsed = PersistentBindingProto()
        parsed.ParseFromString(wire)
        self.assertEqual(list(parsed.input_field_path), ["a.b", "c"])
        self.assertEqual(list(parsed.output_field_path), ["a", "b.c"])
        graph = onnx.GraphProto(persistent_bindings=[binding])
        self.assertEqual(graph.SerializeToString(), b"\xca\x3e" + bytes([len(wire)]) + wire)
        restored = onnx.GraphProto()
        restored.ParseFromString(graph.SerializeToString())
        self.assertEqual(restored.persistent_bindings[0].input_name, "in")

    def test_whole_tensor_and_model_roundtrip(self):
        model = model_with_binding()
        verify.verify_model(model)
        restored = onnx.ModelProto()
        restored.ParseFromString(model.SerializeToString())
        verify.verify_model(restored)
        self.assertEqual(restored.graph.persistent_bindings[0].output_name, "state.out")
        self.assertEqual(len(restored.graph.persistent_bindings[0].input_field_path), 0)

    def test_whole_structure_and_catalogue(self):
        value_type = structure_type([("cache", tensor_type())])
        verify.verify_model(model_with_binding(value_type))
        declaration = value_type.struct_type
        declaration.type_id = 7
        reference = onnx.TypeProto(struct_type=onnx.StructTypeProto(type_ref=7))
        model = model_with_binding(reference)
        model.struct_types.append(declaration)
        verify.verify_model(model)
        model.graph.persistent_bindings[0].input_field_path.append("cache")
        model.graph.persistent_bindings[0].output_field_path.append("cache")
        verify.verify_model(model)

    def test_dotted_fields_are_unambiguous(self):
        value_type = structure_type(
            [("a.b", tensor_type()), ("a", structure_type([("b", tensor_type())]))]
        )
        model = model_with_binding(
            value_type, input_field_path=["a.b"], output_field_path=["a", "b"]
        )
        model.graph.persistent_bindings.append(
            PersistentBindingProto(
                input_name="state.in",
                output_name="state.out",
                input_field_path=["a", "b"],
                output_field_path=["a.b"],
            )
        )
        verify.verify_model(model)

    def test_duplicate_and_overlapping_destinations(self):
        value_type = structure_type([("cache", tensor_type())])
        for first_path, second_path in [([], []), ([], ["cache"]), (["cache"], [])]:
            with self.subTest(first=first_path, second=second_path):
                model = model_with_binding(
                    value_type, input_field_path=first_path, output_field_path=first_path
                )
                model.graph.persistent_bindings.append(
                    PersistentBindingProto(
                        input_name="state.in",
                        output_name="state.out",
                        input_field_path=second_path,
                        output_field_path=second_path,
                    )
                )
                with self.assertRaisesRegex(ValueError, "overlapping"):
                    verify.verify_model(model)

    def test_invalid_names_and_paths(self):
        for field, value in [
            ("input_name", ""),
            ("output_name", ""),
            ("input_name", "missing"),
            ("output_name", "missing"),
        ]:
            with self.subTest(field=field, value=value):
                model = model_with_binding()
                setattr(model.graph.persistent_bindings[0], field, value)
                with self.assertRaisesRegex(ValueError, "input/output"):
                    verify.verify_model(model)
        value_type = structure_type([("cache", tensor_type())])
        for path in [[""], ["missing"], ["cache", "extra"]]:
            with self.subTest(path=path):
                model = model_with_binding(value_type, input_field_path=path)
                with self.assertRaisesRegex(ValueError, "path"):
                    verify.verify_model(model)

    def test_constants_cannot_be_selected(self):
        value_type = structure_type([("cache", tensor_type())])
        value_type.struct_type.structure.field.append(
            onnx.StructTypeProto.Structure.Field(
                name="constant",
                constant=helper.make_tensor("constant", onnx.TensorProto.FLOAT, [], [1.0]),
            )
        )
        model = model_with_binding(
            value_type, input_field_path=["constant"], output_field_path=["constant"]
        )
        with self.assertRaisesRegex(ValueError, "constant"):
            verify.verify_model(model)

    def test_invalid_types_and_shapes(self):
        for output_type in [
            tensor_type(onnx.TensorProto.INT32),
            tensor_type(shape=(1, 2)),
            tensor_type(shape=(3,)),
            tensor_type(shape=(-1,)),
            onnx.TypeProto(),
            structure_type([("cache", tensor_type())]),
        ]:
            with self.subTest(output_type=str(output_type)), self.assertRaises(ValueError):
                verify.verify_model(model_with_binding(output_type=output_type))

    def test_partial_and_symbolic_tensor_declarations(self):
        unknown_rank = onnx.TypeProto(
            tensor_type=onnx.TypeProto.Tensor(elem_type=onnx.TensorProto.FLOAT)
        )
        for input_type, output_type in [
            (tensor_type(shape=("N",)), tensor_type(shape=("M",))),
            (tensor_type(shape=("N",)), tensor_type(shape=(3,))),
            (tensor_type(shape=(None, 2)), tensor_type(shape=(7, 2))),
            (unknown_rank, tensor_type(shape=(3, 2))),
            (tensor_type(shape=(3, 2)), unknown_rank),
            (unknown_rank, unknown_rank),
        ]:
            with self.subTest(input_type=str(input_type), output_type=str(output_type)):
                if input_type.tensor_type.has_shape() and output_type.tensor_type.has_shape():
                    verify.verify_model(model_with_binding(input_type, output_type))
                verify.verify_model(
                    model_with_binding(
                        structure_type([("cache", input_type)]),
                        structure_type([("cache", output_type)]),
                    )
                )
        with self.assertRaisesRegex(ValueError, "declared dimensions"):
            verify.verify_model(
                model_with_binding(tensor_type(shape=("N", 2)), tensor_type(shape=("M", 3)))
            )

    def test_subgraphs_reject_bindings(self):
        model = model_with_binding()
        with self.assertRaisesRegex(ValueError, "root graph"):
            verify.verify_graph(model.graph, is_main_graph=False)
        child = model.graph
        parent = helper.make_model(
            helper.make_graph(
                [helper.make_node("Custom", [], ["out"], body=child)],
                "parent",
                [],
                [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [2])],
            )
        )
        with self.assertRaisesRegex(ValueError, "root graph"):
            verify.verify_model(parent)

    def test_builder_preserves_declarations_and_io_rewrites(self):
        model = model_with_binding()
        builder = GraphBuilder(model)
        self.assertEqual(len(builder.persistent_bindings()), 1)
        builder.remove_identity_nodes()
        builder.remove_unused_nodes()
        builder.remove_duplicate_nodes()
        rebuilt = builder.to_onnx()
        verify.verify_model(rebuilt)
        self.assertEqual(rebuilt.graph.persistent_bindings[0].input_name, "state.in")
        self.assertEqual(rebuilt.graph.persistent_bindings[0].output_name, "state.out")
        self.assertEqual(builder.build_graph().persistent_bindings[0].output_name, "state.out")
        again = GraphBuilder(rebuilt).to_onnx()
        self.assertEqual(
            again.graph.persistent_bindings[0].SerializeToString(),
            model.graph.persistent_bindings[0].SerializeToString(),
        )

    def test_builder_authoring_and_rejected_exports(self):
        builder = GraphBuilder("persistent")
        builder.make_input("x", onnx.TensorProto.FLOAT, [2])
        builder.make_node("Identity", ["x"], ["y"])
        builder.make_output("y")
        builder.make_persistent_binding(PersistentBindingProto(input_name="x", output_name="y"))
        self.assertEqual(len(builder.to_onnx().graph.persistent_bindings), 1)
        with self.assertRaisesRegex(Exception, "standard ONNX"):
            builder.to_standard_model()
        with self.assertRaisesRegex(Exception, "FunctionProto"):
            builder.to_onnx("function")
        builder.make_persistent_binding(PersistentBindingProto(input_name="x", output_name="y"))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            builder.to_onnx()

    def test_renamed_or_removed_io_cannot_silently_break_bindings(self):
        model = model_with_binding()
        model.graph.input[0].name = "renamed"
        model.graph.node[0].input.clear()
        model.graph.node[0].input.extend(["renamed"])
        with self.assertRaisesRegex(ValueError, "input/output"):
            GraphBuilder(model).to_onnx()
        model.graph.persistent_bindings[0].input_name = "renamed"
        verify.verify_model(GraphBuilder(model).to_onnx())
        del model.graph.output[:]
        with self.assertRaisesRegex(ValueError, "input/output"):
            GraphBuilder(model).to_onnx()


if __name__ == "__main__":
    unittest.main()
