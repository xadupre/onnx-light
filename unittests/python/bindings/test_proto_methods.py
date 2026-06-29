# Tests for the convenience methods added to proto classes
# (see onnx_light/onnx_proto/_proto_methods.py).
import unittest

import numpy as np

import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_lib import (
    AttributeProto,
    DeviceConfigurationProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    OptionalProto,
    SparseTensorProto,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
)


class TestProtoMethods(ExtTestCase):
    def test_node_set_attribute_scalar(self):
        node = NodeProto()
        node.op_type = "Conv"
        node.set_attribute("axis", 1)
        node.set_attribute("alpha", 0.5)
        node.set_attribute("mode", "constant")
        node.set_attribute("pads", [1, 1, 2, 2])

        self.assertEqual(len(node.attribute), 4)
        by_name = {a.name: a for a in node.attribute}
        self.assertEqual(by_name["axis"].type, AttributeProto.INT)
        self.assertEqual(by_name["axis"].i, 1)
        self.assertEqual(by_name["alpha"].type, AttributeProto.FLOAT)
        self.assertEqual(by_name["alpha"].f, 0.5)
        self.assertEqual(by_name["mode"].type, AttributeProto.STRING)
        self.assertEqual(by_name["mode"].s, b"constant")
        self.assertEqual(by_name["pads"].type, AttributeProto.INTS)
        self.assertEqual(list(by_name["pads"].ints), [1, 1, 2, 2])

    def test_node_set_attribute_replace(self):
        node = NodeProto()
        node.set_attribute("axis", 1)
        node.set_attribute("axis", 2)
        self.assertEqual(len(node.attribute), 1)
        self.assertEqual(node.attribute[0].i, 2)

    def test_graph_add_input_output(self):
        g = GraphProto()
        vi = g.add_input("X", TensorProto.FLOAT, [None, 3])
        self.assertIsInstance(vi, ValueInfoProto)
        self.assertEqual(vi.name, "X")
        g.add_output("Y", TensorProto.FLOAT, [None, 3])
        self.assertEqual([v.name for v in g.input], ["X"])
        self.assertEqual([v.name for v in g.output], ["Y"])

    def test_graph_add_input_accepts_value_info(self):
        g = GraphProto()
        vi = oh.make_tensor_value_info("a", TensorProto.FLOAT, [1])
        g.add_input(vi)
        self.assertEqual(g.input[0].name, "a")

    def test_graph_add_initializer_from_array(self):
        g = GraphProto()
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = g.add_initializer("W", arr)
        self.assertEqual(t.name, "W")
        self.assertEqual(list(t.dims), [2, 2])
        self.assertEqual(t.data_type, int(TensorProto.FLOAT))

    def test_graph_add_initializer_from_tensor(self):
        g = GraphProto()
        t = oh.make_tensor("c", TensorProto.INT64, [2], [3, 4])
        g.add_initializer(t)
        self.assertEqual(g.initializer[0].name, "c")
        self.assertEqual(list(g.initializer[0].dims), [2])

    def test_graph_add_node_with_attributes(self):
        g = GraphProto()
        node = g.add_node("Concat", ["a", "b"], ["c"], name="concat0", axis=0)
        self.assertIsInstance(node, NodeProto)
        self.assertEqual(node.op_type, "Concat")
        self.assertEqual(list(node.input), ["a", "b"])
        self.assertEqual(list(node.output), ["c"])
        self.assertEqual(node.name, "concat0")
        self.assertEqual(len(node.attribute), 1)
        self.assertEqual(node.attribute[0].name, "axis")
        self.assertEqual(node.attribute[0].i, 0)

    def test_function_add_helpers(self):
        f = FunctionProto()
        f.name = "MyFunc"
        f.add_input("x")
        f.add_input("a")
        f.add_output("y")
        f.add_node("Relu", ["x"], ["y"])
        f.add_opset("", 18)
        self.assertEqual(list(f.input), ["x", "a"])
        self.assertEqual(list(f.output), ["y"])
        self.assertEqual(f.node[0].op_type, "Relu")
        self.assertEqual(f.opset_import[0].domain, "")
        self.assertEqual(f.opset_import[0].version, 18)

    def test_model_add_function_opset_metadata(self):
        m = ModelProto()
        m.add_opset("", 18)
        m.add_opset("ai.onnx.ml", 4)
        m.add_metadata("author", "alice")
        # Replace existing metadata entry.
        m.add_metadata("author", "bob")
        m.add_metadata("version", "1")

        f = FunctionProto()
        f.name = "F"
        m.add_function(f)

        self.assertEqual(
            [(o.domain, o.version) for o in m.opset_import], [("", 18), ("ai.onnx.ml", 4)]
        )
        self.assertEqual(
            [(p.key, p.value) for p in m.metadata_props], [("author", "bob"), ("version", "1")]
        )
        self.assertEqual([fn.name for fn in m.functions], ["F"])

    def test_model_add_function_type_check(self):
        m = ModelProto()
        with self.assertRaises(TypeError):
            m.add_function("not-a-function")

    def test_end_to_end_round_trip(self):
        g = GraphProto()
        g.name = "main"
        g.add_input("X", TensorProto.FLOAT, [None, 3])
        g.add_output("Y", TensorProto.FLOAT, [None, 3])
        g.add_initializer("W", np.eye(3, dtype=np.float32))
        g.add_node("MatMul", ["X", "W"], ["Y"])

        m = ModelProto()
        m.graph.CopyFrom(g)
        m.add_opset("", 18)
        m.ir_version = 10
        m.add_metadata("producer", "demo")

        buf = m.SerializeToString()
        m2 = ModelProto()
        m2.ParseFromString(buf)
        self.assertEqual([vi.name for vi in m2.graph.input], ["X"])
        self.assertEqual([vi.name for vi in m2.graph.output], ["Y"])
        self.assertEqual(m2.graph.initializer[0].name, "W")
        self.assertEqual(m2.graph.node[0].op_type, "MatMul")
        self.assertEqual(m2.opset_import[0].version, 18)
        self.assertEqual(m2.metadata_props[0].key, "producer")

    def test_clear_field_repeated(self):
        tensor = TensorProto()
        tensor.dims.extend([1, 2, 3])
        tensor.ClearField("dims")
        self.assertEqual(list(tensor.dims), [])

    def test_clear_field_string(self):
        tensor = TensorProto()
        tensor.name = "abc"
        tensor.ClearField("name")
        self.assertEqual(tensor.name, "")
        self.assertFalse(tensor.has_name())

    def test_clear_field_optional_message(self):
        type_proto = TypeProto()
        type_proto.tensor_type.elem_type = TensorProto.FLOAT
        type_proto.tensor_type.shape.dim.add().dim_value = 3
        self.assertTrue(type_proto.tensor_type.has_shape())
        type_proto.tensor_type.ClearField("shape")
        self.assertFalse(type_proto.tensor_type.has_shape())
        # Other fields are left untouched.
        self.assertEqual(type_proto.tensor_type.elem_type, TensorProto.FLOAT)

    def test_clear_field_optional_scalar(self):
        dim = TensorShapeProto.Dimension()
        dim.dim_value = 5
        self.assertTrue(dim.has_dim_value())
        dim.ClearField("dim_value")
        self.assertFalse(dim.has_dim_value())

    def test_clear_field_required_message(self):
        sparse = SparseTensorProto()
        sparse.values.name = "v"
        sparse.values.dims.extend([2])
        sparse.ClearField("values")
        self.assertEqual(sparse.values.name, "")
        self.assertEqual(list(sparse.values.dims), [])

    def test_clear_field_absent_is_noop(self):
        # Clearing an unset field must not raise.
        TensorProto().ClearField("name")

    def test_set_optional_field_to_none(self):
        type_proto = TypeProto()
        type_proto.tensor_type.shape.dim.add()
        type_proto.tensor_type.shape = None
        self.assertFalse(type_proto.tensor_type.has_shape())

        dim = TensorShapeProto.Dimension()
        dim.dim_value = 7
        dim.dim_value = None
        self.assertFalse(dim.has_dim_value())

    def test_which_oneof_type_proto(self):
        type_proto = TypeProto()
        self.assertIsNone(type_proto.WhichOneof("value"))
        type_proto.tensor_type.elem_type = TensorProto.FLOAT
        self.assertEqual(type_proto.WhichOneof("value"), "tensor_type")

        seq_proto = TypeProto()
        seq_proto.sequence_type.elem_type.tensor_type.elem_type = TensorProto.FLOAT
        self.assertEqual(seq_proto.WhichOneof("value"), "sequence_type")

    def test_which_oneof_type_proto_invalid_name(self):
        type_proto = TypeProto()
        with self.assertRaises(ValueError):
            type_proto.WhichOneof("not_a_oneof")

    def test_which_oneof_optional_proto(self):
        optional_proto = OptionalProto()
        self.assertIsNone(optional_proto.WhichOneof("value"))
        optional_proto.tensor_value.name = "t"
        self.assertEqual(optional_proto.WhichOneof("value"), "tensor_value")

    def test_which_oneof_optional_proto_invalid_name(self):
        optional_proto = OptionalProto()
        with self.assertRaises(ValueError):
            optional_proto.WhichOneof("not_a_oneof")

    def test_repeated_proto_add_with_kwargs(self):
        model = ModelProto()
        opset = model.opset_import.add(domain="", version=18)
        self.assertEqual(opset.domain, "")
        self.assertEqual(opset.version, 18)
        self.assertEqual(len(model.opset_import), 1)
        self.assertEqual(model.opset_import[0].version, 18)

    def test_repeated_proto_add_without_kwargs(self):
        model = ModelProto()
        opset = model.opset_import.add()
        self.assertEqual(opset.domain, "")
        self.assertEqual(opset.version, 0)
        self.assertEqual(len(model.opset_import), 1)

    def test_proto_constructor_without_kwargs(self):
        tensor = TensorProto()
        self.assertEqual(list(tensor.dims), [])
        self.assertEqual(int(tensor.data_type), 0)

    def test_proto_constructor_with_kwargs(self):
        source = TensorProto()
        source.dims.extend([1, 9])
        tensor = TensorProto(dims=source.dims, data_type=TensorProto.FLOAT, raw_data=b"abcd")
        self.assertEqual(list(tensor.dims), [1, 9])
        self.assertEqual(int(tensor.data_type), int(TensorProto.FLOAT))
        self.assertEqual(tensor.raw_data, b"abcd")

    def test_proto_constructor_with_repeated_string_kwargs(self):
        config = DeviceConfigurationProto(name="conf0", num_devices=2, device=["CPU", "CUDA:0"])
        self.assertEqual(config.name, "conf0")
        self.assertEqual(config.num_devices, 2)
        self.assertEqual(list(config.device), ["CPU", "CUDA:0"])

    @unittest.skip("RepeatedField cannot be initialized with a list")
    def test_repeated_proto_add_with_repeated_string_kwargs(self):
        model = ModelProto()
        config = model.configuration.add(name="c", num_devices=1, device=["CPU"])
        self.assertEqual(config.name, "c")
        self.assertEqual(list(config.device), ["CPU"])

    def test_repeated_field_is_sequence(self):
        import collections.abc

        node = oh.make_node("MatMul", ["a", "b"], ["c"])
        # RepeatedField of strings.
        self.assertIsInstance(node.input, collections.abc.Sequence)
        # RepeatedProtoField of messages.
        model = ModelProto()
        model.opset_import.add(domain="", version=18)
        self.assertIsInstance(model.opset_import, collections.abc.Sequence)

    def test_repeated_field_eq_list_str(self):
        node = oh.make_node("MatMul", ["a", "b"], ["c"])
        # RepeatedField of strings compared to a list of str.
        self.assertTrue(node.input == ["a", "b"])
        self.assertFalse(node.input != ["a", "b"])
        self.assertFalse(node.input == ["a"])
        self.assertFalse(node.input == ["a", "x"])
        # str elements also compare equal to their bytes counterparts.
        self.assertTrue(node.input == [b"a", b"b"])

    def test_repeated_field_eq_list_numbers(self):
        ints = oh.make_attribute("ints", [1, 2, 3]).ints
        self.assertTrue(ints == [1, 2, 3])
        self.assertFalse(ints == [1, 2])
        self.assertFalse(ints == [1, 2, 4])
        floats = oh.make_attribute("floats", [1.0, 2.0]).floats
        self.assertTrue(floats == [1.0, 2.0])
        self.assertFalse(floats == [1.0, 3.0])

    def test_repeated_field_extend_iterable_str(self):
        # ``extend`` must accept any iterable (e.g. a generator), not just
        # sequences such as lists or tuples.
        node = oh.make_node("MatMul", ["a", "b"], ["c"])
        node.input.extend(name for name in ["d", "e"])
        self.assertEqual(list(node.input), ["a", "b", "d", "e"])
        node.input.extend(map(str, ["f", "g"]))
        self.assertEqual(list(node.input), ["a", "b", "d", "e", "f", "g"])

    def test_repeated_field_extend_iterable_numbers(self):
        # Same behavior for numeric repeated fields.
        tensor = TensorProto()
        tensor.dims.extend(d for d in [2, 3])
        self.assertEqual(list(tensor.dims), [2, 3])
        tensor.dims.extend(map(int, ["4", "5"]))
        self.assertEqual(list(tensor.dims), [2, 3, 4, 5])
        # Plain sequences keep working.
        tensor.dims.extend([6, 7])
        self.assertEqual(list(tensor.dims), [2, 3, 4, 5, 6, 7])

    def test_init_kwargs_tensor_proto(self):
        # The use case from the issue: build a TensorProto from another
        # tensor's repeated ``dims`` field, its ``data_type`` and raw bytes.
        raw = np.array([1, 2, 3, 4], dtype=np.float32).tobytes()
        source = TensorProto()
        source.dims.extend([2, 2])
        source.data_type = TensorProto.FLOAT
        tensor = TensorProto(dims=source.dims, data_type=source.data_type, raw_data=raw)
        self.assertEqual(list(tensor.dims), [2, 2])
        self.assertEqual(tensor.data_type, TensorProto.FLOAT)
        self.assertEqual(tensor.raw_data, raw)

    def test_init_kwargs_repeated_from_list(self):
        tensor = TensorProto(dims=[3, 4], data_type=TensorProto.INT64, name="x")
        self.assertEqual(list(tensor.dims), [3, 4])
        self.assertEqual(tensor.data_type, TensorProto.INT64)
        self.assertEqual(tensor.name, "x")

    def test_init_kwargs_empty_matches_default(self):
        self.assertEqual(list(TensorProto().dims), [])
        self.assertEqual(TensorProto().SerializeToString(), TensorProto().SerializeToString())

    def test_init_kwargs_node_proto(self):
        node = NodeProto(op_type="Add", input=["a", "b"], output=["c"], name="n1")
        self.assertEqual(node.op_type, "Add")
        self.assertEqual(list(node.input), ["a", "b"])
        self.assertEqual(list(node.output), ["c"])
        self.assertEqual(node.name, "n1")

    def test_init_kwargs_repeated_message_from_list(self):
        node = NodeProto(op_type="Add")
        graph = GraphProto(name="g", node=[node])
        self.assertEqual(graph.name, "g")
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].op_type, "Add")

    def test_init_kwargs_subproto(self):
        tensor_type = TypeProto.Tensor(elem_type=TensorProto.FLOAT)
        self.assertEqual(tensor_type.elem_type, TensorProto.FLOAT)

    def test_init_kwargs_unknown_field_raises(self):
        with self.assertRaises(AttributeError):
            TensorProto(not_a_field=1)


class TestProtoStrToMultiline(ExtTestCase):
    """Tests for the proto_str_to_multiline utility."""

    def _import(self):
        from onnx_light.onnx_proto._helper import proto_str_to_multiline

        return proto_str_to_multiline

    def test_flat_node_becomes_multiline(self):
        """Flat proto string for a NodeProto is formatted with indentation."""
        fn = self._import()
        node = oh.make_node("Relu", ["X"], ["Y"], name="relu1")
        flat = str(node)
        result = fn(flat)
        self.assertNotIn("\n" + " " * 0 + "{", result[1:])  # inner braces are indented
        self.assertIn('name: "relu1"', result)
        self.assertIn('op_type: "Relu"', result)
        self.assertIn("\n", result)
        # No trailing trailing space; each field on its own line.
        for line in result.splitlines():
            self.assertFalse(line.endswith(" "), f"trailing space in: {line!r}")

    def test_output_starts_with_open_brace(self):
        fn = self._import()
        node = oh.make_node("Add", ["a", "b"], ["c"])
        result = fn(str(node))
        self.assertTrue(result.startswith("{\n"))

    def test_output_ends_with_close_brace(self):
        fn = self._import()
        node = oh.make_node("Add", ["a", "b"], ["c"])
        result = fn(str(node))
        self.assertTrue(result.rstrip("\n").endswith("}"))

    def test_nested_message_indented(self):
        """Nested messages (e.g. model.graph) are indented one level deeper."""
        fn = self._import()
        node = oh.make_node("Relu", ["X"], ["Y"])
        graph = oh.make_graph([node], "g", [], [])
        model = oh.make_model(graph)
        result = fn(str(model))
        lines = result.splitlines()
        graph_line = next((line for line in lines if "graph:" in line), None)
        self.assertIsNotNone(graph_line, "graph field not found")
        # The graph field must be indented (not at column 0).
        self.assertTrue(graph_line.startswith("  "), f"graph line not indented: {graph_line!r}")

    def test_array_value_kept_on_one_line(self):
        """Array fields like dims are kept on a single line."""
        fn = self._import()
        tensor = TensorProto()
        tensor.name = "t"
        tensor.dims.extend([2, 3, 4])
        result = fn(str(tensor))
        # dims: [2, 3, 4] must appear on one line.
        self.assertIn("dims: [2, 3, 4]", result)

    def test_custom_indent(self):
        """The indent parameter controls the indentation string."""
        fn = self._import()
        node = oh.make_node("Relu", ["X"], ["Y"])
        result_4 = fn(str(node), indent="    ")
        # Fields at depth 1 should use four spaces.
        has_four_space_line = any(
            line.startswith("    ") and not line.startswith("        ")
            for line in result_4.splitlines()
            if line.strip()
        )
        self.assertTrue(has_four_space_line)

    def test_importable_from_onnx_module(self):
        """proto_str_to_multiline is accessible from onnx_light.onnx."""
        import onnx_light.onnx as onnxl

        self.assertTrue(hasattr(onnxl, "proto_str_to_multiline"))
        node = oh.make_node("Sigmoid", ["x"], ["y"])
        result = onnxl.proto_str_to_multiline(str(node))
        self.assertIn('op_type: "Sigmoid"', result)


if __name__ == "__main__":
    unittest.main()
