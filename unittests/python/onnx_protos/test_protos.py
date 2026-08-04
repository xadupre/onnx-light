import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx.helper as oh
from onnx_light.onnx_py import _onnxpyprotoop as m

MAX_SHORT_REPR_LENGTH = 60


class TestStringBinding(ExtTestCase):
    """Tests for `_onnxpy.String` behavior."""

    def test_string_add_str(self):
        """Tests String + str returns a Python string."""
        value = m.String("ab") + "cd"
        self.assertEqual(value, "abcd")
        self.assertIsInstance(value, str)

    def test_str_add_string(self):
        """Tests str + String returns a Python string."""
        value = "ab" + m.String("cd")
        self.assertEqual(value, "abcd")
        self.assertIsInstance(value, str)

    def test_string_str(self):
        """Tests str(String) returns a Python string."""
        value = str(m.String("abc"))
        self.assertEqual(value, "abc")
        self.assertIsInstance(value, str)

    def test_string_eq_string(self):
        """Tests String == String."""
        self.assertTrue(m.String("abc") == m.String("abc"))
        self.assertFalse(m.String("abc") == m.String("xyz"))

    def test_string_eq_str(self):
        """Tests String == str on both sides."""
        self.assertTrue(m.String("abc") == "abc")
        self.assertTrue("abc" == m.String("abc"))
        self.assertFalse(m.String("abc") == "xyz")
        self.assertFalse("xyz" == m.String("abc"))

    def test_string_eq_bytes(self):
        """Tests String == bytes on both sides."""
        self.assertTrue(m.String("abc") == b"abc")
        self.assertTrue(b"abc" == m.String("abc"))
        self.assertFalse(m.String("abc") == b"xyz")
        self.assertFalse(b"xyz" == m.String("abc"))

    def test_string_eq_none(self):
        """Tests String == None returns False."""
        self.assertFalse(m.String("abc") == None)  # noqa: E711
        self.assertFalse(None == m.String("abc"))  # noqa: E711
        self.assertTrue(m.String("abc") != None)  # noqa: E711
        self.assertTrue(None != m.String("abc"))  # noqa: E711

    def test_string_eq_int(self):
        """Tests String == int returns False."""
        self.assertFalse(m.String("abc") == 3)
        self.assertFalse(3 == m.String("abc"))
        self.assertTrue(m.String("abc") != 3)
        self.assertTrue(3 != m.String("abc"))

    def test_string_eq_other(self):
        """Tests String compared to unrelated objects returns False."""
        self.assertFalse(m.String("abc") == 1.5)
        self.assertFalse(1.5 == m.String("abc"))
        self.assertFalse(m.String("abc") == [1, 2])
        self.assertFalse([1, 2] == m.String("abc"))
        self.assertTrue(m.String("abc") != 1.5)
        self.assertTrue(1.5 != m.String("abc"))
        self.assertTrue(m.String("abc") != [1, 2])
        self.assertTrue([1, 2] != m.String("abc"))

    def test_string_ne_str(self):
        """Tests String != str."""
        self.assertTrue(m.String("abc") != "xyz")
        self.assertFalse(m.String("abc") != "abc")

    def test_string_ne_bytes(self):
        """Tests String != bytes."""
        self.assertTrue(m.String("abc") != b"xyz")
        self.assertFalse(m.String("abc") != b"abc")

    def test_string_ne_string(self):
        """Tests String != String."""
        self.assertTrue(m.String("abc") != m.String("xyz"))
        self.assertFalse(m.String("abc") != m.String("abc"))

    def test_string_lt_str(self):
        """Tests String < str."""
        self.assertTrue(m.String("abc") < "xyz")
        self.assertFalse(m.String("xyz") < "abc")
        self.assertFalse(m.String("abc") < "abc")

    def test_string_lt_bytes(self):
        """Tests String < bytes."""
        self.assertTrue(m.String("abc") < b"xyz")
        self.assertFalse(m.String("xyz") < b"abc")

    def test_string_lt_string(self):
        """Tests String < String."""
        self.assertTrue(m.String("abc") < m.String("xyz"))
        self.assertFalse(m.String("xyz") < m.String("abc"))
        self.assertFalse(m.String("abc") < m.String("abc"))

    def test_string_gt_str(self):
        """Tests String > str."""
        self.assertTrue(m.String("xyz") > "abc")
        self.assertFalse(m.String("abc") > "xyz")
        self.assertFalse(m.String("abc") > "abc")

    def test_string_gt_bytes(self):
        """Tests String > bytes."""
        self.assertTrue(m.String("xyz") > b"abc")
        self.assertFalse(m.String("abc") > b"xyz")

    def test_string_gt_string(self):
        """Tests String > String."""
        self.assertTrue(m.String("xyz") > m.String("abc"))
        self.assertFalse(m.String("abc") > m.String("xyz"))
        self.assertFalse(m.String("abc") > m.String("abc"))

    def test_string_decode_default(self):
        """Tests String.decode mimics bytes.decode with default utf-8."""
        value = m.String("hello").decode()
        self.assertEqual(value, "hello")
        self.assertIsInstance(value, str)

    def test_string_decode_encoding(self):
        """Tests String.decode accepts an explicit encoding argument."""
        self.assertEqual(m.String("hello").decode("utf-8"), "hello")
        self.assertEqual(m.String("").decode("utf-8"), "")

    def test_string_decode_errors(self):
        """Tests String.decode forwards the errors argument."""
        # "é" is stored as its UTF-8 encoding (0xC3 0xA9), invalid as ASCII.
        self.assertEqual(m.String("é").decode("ascii", "ignore"), "")
        with self.assertRaises(UnicodeDecodeError):
            m.String("é").decode("ascii")

    def test_string_decode_attribute_strings(self):
        """Tests decoding STRINGS attribute values returned as String objects."""
        from onnx_light.onnx_proto._helper import get_attribute_value

        attr = oh.make_attribute("x", ["hello", "world"])
        values = get_attribute_value(attr)
        self.assertEqual([v.decode("utf-8") for v in values], ["hello", "world"])


class TestIterator(ExtTestCase):
    def test_node_iterator(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "test_graph",
                [oh.make_tensor_value_info("X", m.TensorProto.FLOAT, [3])],
                [oh.make_tensor_value_info("Y", m.TensorProto.FLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        for node in model.graph.node:
            self.assertEqual(list(node.input), ["X"])
            node.input.clear()
            node.input.extend(["XX"])
            self.assertEqual(list(node.input), ["XX"])
        for node in model.graph.node:
            self.assertEqual(list(node.input), ["XX"])

    def test_node_iterator_proto_copy(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "test_graph",
                [oh.make_tensor_value_info("X", m.TensorProto.FLOAT, [3])],
                [oh.make_tensor_value_info("Y", m.TensorProto.FLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        graph = m.GraphProto()
        graph.ParseFromString(model.graph.SerializeToString())
        for node in graph.node:
            self.assertEqual(list(node.input), ["X"])
            node.input.clear()
            node.input.extend(["XX"])
            self.assertEqual(list(node.input), ["XX"])
        for node in graph.node:
            self.assertEqual(list(node.input), ["XX"])
        for node in model.graph.node:
            self.assertEqual(list(node.input), ["X"])
        model.graph = graph
        for node in model.graph.node:
            self.assertEqual(list(node.input), ["XX"])

    def test_node_iterator_node(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "test_graph",
                [oh.make_tensor_value_info("X", m.TensorProto.FLOAT, [3])],
                [oh.make_tensor_value_info("Y", m.TensorProto.FLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        new_nodes = []
        for node in model.graph.node:
            n = m.NodeProto()
            n.ParseFromString(node.SerializeToString())
            self.assertEqual(list(n.input), ["X"])
            n.input.clear()
            n.input.extend(["XX"])
            self.assertEqual(list(node.input), ["X"])
            self.assertEqual(list(n.input), ["XX"])
            new_nodes.append(n)
        for node in model.graph.node:
            self.assertEqual(list(node.input), ["X"])
        model.graph.node.clear()
        model.graph.node.extend(new_nodes)
        for node in model.graph.node:
            self.assertEqual(list(node.input), ["XX"])

    def test_node_setitem(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["H"]), oh.make_node("Relu", ["H"], ["Y"])],
                "test_graph",
                [oh.make_tensor_value_info("X", m.TensorProto.FLOAT, [3])],
                [oh.make_tensor_value_info("Y", m.TensorProto.FLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        new_node = oh.make_node("Sigmoid", ["X"], ["H"])
        model.graph.node[0] = new_node
        self.assertEqual(model.graph.node[0].op_type, "Sigmoid")
        self.assertEqual(model.graph.node[1].op_type, "Relu")
        # Mutating the source node afterwards does not affect the stored copy.
        new_node.op_type = "Tanh"
        self.assertEqual(model.graph.node[0].op_type, "Sigmoid")
        # Negative index.
        model.graph.node[-1] = oh.make_node("Tanh", ["H"], ["Y"])
        self.assertEqual(model.graph.node[1].op_type, "Tanh")
        # Out-of-boundary index raises.
        with self.assertRaises(RuntimeError):
            model.graph.node[5] = oh.make_node("Relu", ["X"], ["Y"])

    def test_dict_key_hash(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "test_graph",
                [oh.make_tensor_value_info("X", m.TensorProto.FLOAT, [3])],
                [oh.make_tensor_value_info("Y", m.TensorProto.FLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        d = model.graph.node[0].input[0]
        self.assertEqual(d, "X")
        di = {d: "E"}
        self.assertIn(d, di)
        self.assertIn("X", di)
        self.assertEqual(di["X"], "E")

    def test_helpers_copy(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Relu", ["X"], ["Y"])],
                "test_graph",
                [oh.make_tensor_value_info("X", m.TensorProto.FLOAT, [3])],
                [oh.make_tensor_value_info("Y", m.TensorProto.FLOAT, [3])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        graph = oh.make_graph(
            model.graph.node,
            model.graph.name,
            model.graph.input,
            model.graph.output,
            initializer=model.graph.initializer,
        )
        new_model = oh.make_model(
            graph, ir_version=model.ir_version, opset_imports=list(model.opset_import)
        )
        self.assertEqual(len(model.graph.node), 1)
        self.assertTrue(all(n is not None for n in model.graph.node))
        self.assertEqual(len(new_model.graph.node), 1)
        self.assertTrue(all(n is not None for n in new_model.graph.node))
        del model
        model = None  # noqa: F841
        self.assertEqual(len(new_model.graph.node), 1)
        self.assertTrue(all(n is not None for n in new_model.graph.node))
        del graph
        model = None  # noqa: F841
        self.assertEqual(len(new_model.graph.node), 1)
        self.assertTrue(all(n is not None for n in new_model.graph.node))
        new_graph = new_model.graph
        self.assertEqual(len(new_graph.node), 1)
        self.assertTrue(all(n is not None for n in new_graph.node))
        del new_model
        new_model = None  # noqa: F841
        self.assertEqual(len(new_graph.node), 1)
        self.assertTrue(all(n is not None for n in new_graph.node))


class TestModelProtoFields(ExtTestCase):
    """Tests for ModelProto scalar field bindings."""

    def test_model_version_setter_accepts_int(self):
        """Tests that assigning an int to model_version succeeds."""
        model = m.ModelProto()
        self.assertFalse(model.has_model_version())
        model.model_version = 3
        self.assertTrue(model.has_model_version())
        self.assertEqual(model.model_version, 3)

    def test_configuration_field_is_accessible(self):
        """Tests that the repeated ``configuration`` field is exposed."""
        model = m.ModelProto()
        self.assertTrue(hasattr(model, "configuration"))
        self.assertEqual(len(model.configuration), 0)
        config = model.configuration.add()
        config.name = "dev0"
        config.num_devices = 2
        self.assertEqual(len(model.configuration), 1)
        self.assertEqual(model.configuration[0].name, "dev0")
        self.assertEqual(model.configuration[0].num_devices, 2)

    def test_missing_attribute_returns_false(self):
        """Tests that ``hasattr`` returns ``False`` for an unknown field."""
        model = m.ModelProto()
        self.assertFalse(hasattr(model, "not_a_real_field"))


class TestNodeProtoFields(ExtTestCase):
    """Tests for NodeProto field bindings."""

    def test_device_configurations_field_is_accessible(self):
        """Tests that the repeated ``device_configurations`` field is exposed."""
        node = m.NodeProto()
        self.assertTrue(hasattr(node, "device_configurations"))
        self.assertEqual(len(node.device_configurations), 0)
        config = node.device_configurations.add()
        config.configuration_id = "cfg0"
        self.assertEqual(len(node.device_configurations), 1)
        self.assertEqual(node.device_configurations[0].configuration_id, "cfg0")


class TestProtoRepr(ExtTestCase):
    """Tests for proto repr formatting."""

    def test_node_repr_short_stays_on_one_line(self):
        """Tests that a short NodeProto repr stays on one line."""
        node = m.NodeProto()
        node.op_type = "Relu"
        value = repr(node)
        self.assertNotIn("\n", value)
        self.assertLess(len(value), MAX_SHORT_REPR_LENGTH)

    def test_value_info_repr_short_stays_on_one_line(self):
        """Tests that a short ValueInfoProto repr stays on one line."""
        value_info = m.ValueInfoProto()
        value_info.name = "X"
        value = repr(value_info)
        self.assertNotIn("\n", value)
        self.assertLess(len(value), MAX_SHORT_REPR_LENGTH)

    def test_attribute_repr_short_stays_on_one_line(self):
        """Tests that a short AttributeProto repr stays on one line."""
        attribute = m.AttributeProto()
        attribute.name = "alpha"
        attribute.type = m.AttributeProto.FLOAT
        attribute.f = 1.0
        value = repr(attribute)
        self.assertNotIn("\n", value)
        self.assertLess(len(value), MAX_SHORT_REPR_LENGTH)

    def test_type_proto_repr_short_stays_on_one_line(self):
        """Tests that a short TypeProto repr stays on one line."""
        tp = m.TypeProto()
        tp.add_tensor_type().elem_type = m.TensorProto.FLOAT
        value = repr(tp)
        self.assertNotIn("\n", value)
        self.assertLess(len(value), MAX_SHORT_REPR_LENGTH)

    def test_tensor_shape_dimension_repr_short_stays_on_one_line(self):
        """Tests that a short TensorShapeProto.Dimension repr stays on one line."""
        dim = m.TensorShapeProto.Dimension()
        dim.dim_value = 4
        value = repr(dim)
        self.assertNotIn("\n", value)
        self.assertLess(len(value), MAX_SHORT_REPR_LENGTH)

    def test_tensor_shape_repr_short_stays_on_one_line(self):
        """Tests that a short TensorShapeProto repr stays on one line."""
        shape = m.TensorShapeProto()
        dim = shape.dim.add()
        dim.dim_value = 4
        value = repr(shape)
        self.assertNotIn("\n", value)
        self.assertLess(len(value), MAX_SHORT_REPR_LENGTH)

    def test_node_repr_long_is_flat(self):
        """Tests that a long NodeProto repr is always flat (no newlines)."""
        node = m.NodeProto()
        node.op_type = "Relu"
        node.input.extend(["x" * 30])
        node.output.extend(["y" * 30])
        value = repr(node)
        self.assertNotIn("\n", value)

    def test_node_repr_no_double_comma_with_attribute(self):
        """Tests that NodeProto repr with attributes does not contain double commas."""
        node = m.NodeProto()
        node.op_type = "Relu"
        attr = node.attribute.add()
        attr.name = "alpha"
        attr.type = m.AttributeProto.FLOAT
        attr.f = 1.0
        value = repr(node)
        self.assertNotIn(",,", value)

    def test_model_repr_short_displays_opset_import(self):
        """Tests that a short ModelProto repr includes opset_import."""
        model = m.ModelProto()
        opset = model.opset_import.add()
        opset.version = 18
        value = repr(model)
        self.assertIn("opset_import", value)
        self.assertNotIn("\n", value)

    def test_operator_set_id_repr_short_stays_on_one_line(self):
        """Tests that a short OperatorSetIdProto repr stays on one line."""
        opset = m.OperatorSetIdProto()
        opset.version = 18
        value = repr(opset)
        self.assertIn("version", value)
        self.assertNotIn("\n", value)
        self.assertLess(len(value), MAX_SHORT_REPR_LENGTH)

    def test_opset_import_repr_short_stays_on_one_line(self):
        """Tests that a short opset_import repr stays on one line."""
        model = m.ModelProto()
        opset = model.opset_import.add()
        opset.version = 18
        value = repr(model.opset_import)
        self.assertIn("version", value)
        self.assertNotIn("\n", value)
        self.assertLess(len(value), MAX_SHORT_REPR_LENGTH)

    def test_model_repr_long_is_flat(self):
        """Tests that a long ModelProto repr is always flat (no newlines)."""
        model = m.ModelProto()
        model.producer_name = "a" * 60
        value = repr(model)
        self.assertNotIn("\n", value)

    def test_attribute_data_type(self):
        dt = m.AttributeProto.AttributeType.UNDEFINED
        self.assertEqual(dt, m.AttributeProto.AttributeType.UNDEFINED)
        dti = int(dt)
        self.assertEqual(dti, 0)
        att = m.AttributeProto()
        self.assertEqual(att.type, m.AttributeProto.AttributeType.UNDEFINED)
        dti = int(att.type)
        self.assertEqual(dti, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
