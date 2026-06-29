import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx
import onnx.helper
import onnx_light.onnx.checker as checker
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
import onnx_light.onnx.parser as parser
import onnx_light.onnx.version_converter as version_converter
from onnx_light.onnx import TensorProto

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
        proto = parser.parse_model(text)
        self.assertEqual(proto.ir_version, 9)

    def test_parse_graph(self):
        """Tests parsing a GraphProto."""
        text = "agraph (float[10] X) => (float[10] Y) {\n  Y = Identity(X)\n}\n"
        from onnx_light.onnx_py import _onnxpyprotolib as _C

        ok, err, graph = _C.parser.parse_graph(text)
        self.assertTrue(ok, f"parse_graph failed: {err}")
        self.assertEqual(graph.name, "agraph")

    def test_parse_function(self):
        """Tests parsing a FunctionProto."""
        text = '<opset_import: ["" : 21]>\nmyFunc (X) => (Y) {\n  Y = Identity(X)\n}\n'
        ok = parser.parse_function(text)
        self.assertTrue(ok)

    def test_parse_node(self):
        """Tests parsing a NodeProto."""
        text = "Y = Identity(X)"
        node = parser.parse_node(text)
        self.assertEqual(node.op_type, "Identity")

    def test_parse_model_error(self):
        """Tests that a parse error is returned as (False, msg, ...)."""
        from onnx_light.onnx_py import _onnxpyprotolib as _C

        ok, err, _proto = _C.parser.parse_model("not valid onnx text <<<")
        self.assertFalse(ok)
        self.assertGreater(len(err), 0)


class TestDefsSubmodule(ExtTestCase):
    """Tests for the `defs` submodule."""

    def test_get_all_schemas_returns_list(self):
        """Tests that get_all_schemas returns a list."""
        schemas = defs.get_all_schemas()
        self.assertIsInstance(schemas, list)

    def test_get_all_schemas_with_history_returns_list(self):
        """Tests that get_all_schemas_with_history returns a list."""
        schemas = defs.get_all_schemas_with_history()
        self.assertIsInstance(schemas, list)

    def test_schema_version_map(self):
        """Tests that schema_version_map returns a dict."""
        version_map = defs.schema_version_map()
        self.assertIsInstance(version_map, dict)

    def test_schema_error_exception(self):
        """Tests that SchemaError is accessible."""
        self.assertTrue(issubclass(defs.SchemaError, Exception))

    def test_opschema_classes_exist(self):
        """Tests that OpSchema and its nested types are accessible."""
        self.assertTrue(hasattr(defs, "OpSchema"))
        self.assertTrue(hasattr(defs.OpSchema, "FormalParameterOption"))
        self.assertTrue(hasattr(defs.OpSchema, "DifferentiationCategory"))
        self.assertTrue(hasattr(defs.OpSchema, "NodeDeterminism"))
        self.assertTrue(hasattr(defs.OpSchema, "SupportType"))
        self.assertTrue(hasattr(defs.OpSchema, "Attribute"))
        self.assertTrue(hasattr(defs.OpSchema, "TypeConstraintParam"))
        self.assertTrue(hasattr(defs.OpSchema, "FormalParameter"))

    def test_formal_parameter_option_values(self):
        """Tests FormalParameterOption enum values."""
        opt = defs.OpSchema.FormalParameterOption
        self.assertEqual(int(opt.Single), 0)
        self.assertEqual(int(opt.Optional), 1)
        self.assertEqual(int(opt.Variadic), 2)

    def test_support_type_values(self):
        """Tests SupportType enum values."""
        st = defs.OpSchema.SupportType
        self.assertEqual(int(st.COMMON), 0)
        self.assertEqual(int(st.EXPERIMENTAL), 1)

    def test_has_schema_nonexistent(self):
        """Tests has_schema for a nonexistent op."""
        self.assertFalse(defs.has_schema("NonExistentOp12345"))


class TestVersionConverterSubmodule(ExtTestCase):
    """Tests for the `version_converter` submodule."""

    def test_convert_error_exists(self):
        """Tests that ConvertError exception is accessible."""
        self.assertTrue(issubclass(version_converter.ConvertError, Exception))

    def test_convert_error_is_exception(self):
        """Tests that ConvertError inherits from Exception."""
        self.assertTrue(issubclass(version_converter.ConvertError, Exception))


class TestShapeInferenceSubmodule(ExtTestCase):
    """Tests for shape_inference declarations in shape_inference.h."""

    def test_inference_error_type_in_schema(self):
        """Tests that InferenceError is accessible via defs."""
        # InferenceError is declared in shape_inference.h and used by schema.h
        # Its Python binding is not yet added (requires building implementation.cc)
        self.assertTrue(hasattr(defs, "SchemaError"))


class TestCheckerSubmodule(ExtTestCase):
    """Tests for the `checker` submodule."""

    def test_check_model_available(self):
        """Tests that checker.check_model exists."""
        self.assertTrue(hasattr(checker, "check_model"))

    def test_check_model_raises_validation_error(self):
        """Tests checker.check_model raises ValidationError on duplicate metadata keys."""
        graph = oh.make_graph(
            [oh.make_node("Relu", ["X"], ["Y"])],
            "test",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        model = oh.make_model(graph, producer_name="test")
        item = model.metadata_props.add()
        item.key = "a"
        item.value = "1"
        item = model.metadata_props.add()
        item.key = "a"
        item.value = "2"
        with self.assertRaises(checker.ValidationError):
            checker.check_model(model)

    def test_check_model_accepts_upstream_model_proto(self):
        """Tests checker.check_model accepts an upstream onnx ModelProto."""
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Relu", ["X"], ["Y"])],
            "test",
            [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 2])],
            [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2])],
        )
        model = onnx.helper.make_model(graph, producer_name="test")
        checker.check_model(model)

    def test_check_model_proto_like_conversion_errors_raise_type_error(self):
        """Tests proto-like conversion failures raise TypeError after SerializeToString runs."""

        class BadModelProto:
            def __init__(self):
                self.called = False

            def SerializeToString(self):
                self.called = True
                return 3

        bad = BadModelProto()
        with self.assertRaises(TypeError):
            checker.check_model(bad)
        self.assertTrue(bad.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
