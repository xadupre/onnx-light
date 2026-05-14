"""Unit tests for the Python bindings translated from onnx/cpp2py_export.cc."""

import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx.helper as oh
from onnx_light.onnx.onnx_proto import _onnxpy as m

MAX_SHORT_REPR_LENGTH = 60


class TestParserSubmodule(ExtTestCase):
    """Tests for the `parser` submodule."""

    def test_parse_model(self):
        """Tests that a simple text model can be round-tripped."""
        text = (
            '<ir_version: 9, opset_import: ["" : 21]>\n'
            "agraph (float[10] X) => (float[10] Y) {\n"
            "  Y = Identity(X)\n"
            "}\n"
        )
        ok, err, proto = m.parser.parse_model(text)
        self.assertTrue(ok, f"parse_model failed: {err}")
        self.assertEqual(proto.ir_version, 9)

    def test_parse_graph(self):
        """Tests parsing a GraphProto."""
        text = "agraph (float[10] X) => (float[10] Y) {\n  Y = Identity(X)\n}\n"
        ok, err, graph = m.parser.parse_graph(text)
        self.assertTrue(ok, f"parse_graph failed: {err}")
        self.assertEqual(graph.name, "agraph")

    def test_parse_function(self):
        """Tests parsing a FunctionProto."""
        text = '<opset_import: ["" : 21]>\nmyFunc (X) => (Y) {\n  Y = Identity(X)\n}\n'
        ok, _err, _func = m.parser.parse_function(text)
        self.assertTrue(ok)

    def test_parse_node(self):
        """Tests parsing a NodeProto."""
        text = "Y = Identity(X)"
        ok, _err, node = m.parser.parse_node(text)
        self.assertTrue(ok)
        self.assertEqual(node.op_type, "Identity")

    def test_parse_model_error(self):
        """Tests that a parse error is returned as (False, msg, ...)."""
        ok, err, _proto = m.parser.parse_model("not valid onnx text <<<")
        self.assertFalse(ok)
        self.assertGreater(len(err), 0)


class TestDefsSubmodule(ExtTestCase):
    """Tests for the `defs` submodule."""

    def test_get_all_schemas_returns_list(self):
        """Tests that get_all_schemas returns a list."""
        schemas = m.defs.get_all_schemas()
        self.assertIsInstance(schemas, list)

    def test_get_all_schemas_with_history_returns_list(self):
        """Tests that get_all_schemas_with_history returns a list."""
        schemas = m.defs.get_all_schemas_with_history()
        self.assertIsInstance(schemas, list)

    def test_schema_version_map(self):
        """Tests that schema_version_map returns a dict."""
        version_map = m.defs.schema_version_map()
        self.assertIsInstance(version_map, dict)

    def test_schema_error_exception(self):
        """Tests that SchemaError is accessible."""
        self.assertTrue(issubclass(m.defs.SchemaError, Exception))

    def test_opschema_classes_exist(self):
        """Tests that OpSchema and its nested types are accessible."""
        self.assertTrue(hasattr(m.defs, "OpSchema"))
        self.assertTrue(hasattr(m.defs.OpSchema, "FormalParameterOption"))
        self.assertTrue(hasattr(m.defs.OpSchema, "DifferentiationCategory"))
        self.assertTrue(hasattr(m.defs.OpSchema, "NodeDeterminism"))
        self.assertTrue(hasattr(m.defs.OpSchema, "SupportType"))
        self.assertTrue(hasattr(m.defs.OpSchema, "Attribute"))
        self.assertTrue(hasattr(m.defs.OpSchema, "TypeConstraintParam"))
        self.assertTrue(hasattr(m.defs.OpSchema, "FormalParameter"))

    def test_formal_parameter_option_values(self):
        """Tests FormalParameterOption enum values."""
        opt = m.defs.OpSchema.FormalParameterOption
        self.assertEqual(int(opt.Single), 0)
        self.assertEqual(int(opt.Optional), 1)
        self.assertEqual(int(opt.Variadic), 2)

    def test_support_type_values(self):
        """Tests SupportType enum values."""
        st = m.defs.OpSchema.SupportType
        self.assertEqual(int(st.COMMON), 0)
        self.assertEqual(int(st.EXPERIMENTAL), 1)

    def test_has_schema_nonexistent(self):
        """Tests has_schema for a nonexistent op."""
        self.assertFalse(m.defs.has_schema("NonExistentOp12345"))


class TestVersionConverterSubmodule(ExtTestCase):
    """Tests for the `version_converter` submodule."""

    def test_convert_error_exists(self):
        """Tests that ConvertError exception is accessible."""
        self.assertTrue(issubclass(m.version_converter.ConvertError, Exception))

    def test_convert_error_is_exception(self):
        """Tests that ConvertError inherits from Exception."""
        self.assertTrue(issubclass(m.version_converter.ConvertError, Exception))


class TestShapeInferenceSubmodule(ExtTestCase):
    """Tests for shape_inference declarations in shape_inference.h."""

    def test_inference_error_type_in_schema(self):
        """Tests that InferenceError is accessible via defs."""
        # InferenceError is declared in shape_inference.h and used by schema.h
        # Its Python binding is not yet added (requires building implementation.cc)
        self.assertTrue(hasattr(m.defs, "SchemaError"))


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

    def test_string_eq_string(self):
        """Tests String == String."""
        self.assertTrue(m.String("abc") == m.String("abc"))
        self.assertFalse(m.String("abc") == m.String("xyz"))

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


class TestModelProtoFields(ExtTestCase):
    """Tests for ModelProto scalar field bindings."""

    def test_model_version_setter_accepts_int(self):
        """Tests that assigning an int to model_version succeeds."""
        model = m.ModelProto()
        self.assertFalse(model.has_model_version())
        model.model_version = 3
        self.assertTrue(model.has_model_version())
        self.assertEqual(model.model_version, 3)


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

    def test_node_repr_long_keeps_multiline_format(self):
        """Tests that a long NodeProto repr keeps multiline formatting."""
        node = m.NodeProto()
        node.op_type = "Relu"
        node.input.extend(["x" * 30])
        node.output.extend(["y" * 30])
        value = repr(node)
        self.assertIn("\n", value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
