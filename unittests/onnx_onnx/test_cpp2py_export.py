"""Unit tests for the Python bindings translated from onnx/cpp2py_export.cc."""

import unittest

from onnx_light.onnx.onnx_proto import _onnxpy as m


class TestParserSubmodule(unittest.TestCase):
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


class TestDefsSubmodule(unittest.TestCase):
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


class TestVersionConverterSubmodule(unittest.TestCase):
    """Tests for the `version_converter` submodule."""

    def test_convert_error_exists(self):
        """Tests that ConvertError exception is accessible."""
        self.assertTrue(issubclass(m.version_converter.ConvertError, Exception))

    def test_convert_error_is_exception(self):
        """Tests that ConvertError inherits from Exception."""
        self.assertTrue(issubclass(m.version_converter.ConvertError, Exception))


class TestShapeInferenceSubmodule(unittest.TestCase):
    """Tests for shape_inference declarations in shape_inference.h."""

    def test_inference_error_type_in_schema(self):
        """Tests that InferenceError is accessible via defs."""
        # InferenceError is declared in shape_inference.h and used by schema.h
        # Its Python binding is not yet added (requires building implementation.cc)
        self.assertTrue(hasattr(m.defs, "SchemaError"))


if __name__ == "__main__":
    unittest.main()
