from __future__ import annotations

import locale
import platform
import unittest

from onnx_light.onnx import TensorProto, parser
from onnx_light.ext_test_case import ExtTestCase


class TestParser(ExtTestCase):
    def test_parse_float6_types(self):
        model = parser.parse_model("""
            <ir_version: 14, opset_import: ["": 28]>
            agraph (float6e2m3[N] x, float6e3m2[N] y)
                => (float6e2m3[N] z) {
                z = Identity(x)
            }
        """)
        self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, TensorProto.FLOAT6E2M3)
        self.assertEqual(model.graph.input[1].type.tensor_type.elem_type, TensorProto.FLOAT6E3M2)

    def test_parse_model_ok(self):
        model = parser.parse_model("""
            <ir_version: 7, opset_import: ["": 17]>
            agraph (float[N] x) => (float[N] y) {
                y = Identity(x)
            }
            """)
        self.assertEqual(model.graph.node[0].op_type, "Identity")

    def test_parse_graph_ok(self):
        graph = parser.parse_graph("""
            agraph (float[N] x) => (float[N] y) {
                y = Identity(x)
            }
            """)
        self.assertEqual(graph.node[0].op_type, "Identity")

    def test_parse_function_ok(self):
        function = parser.parse_function("""
            <opset_import: ["": 17], domain: "local">
            afun (x) => (y) {
                y = Identity(x)
            }
            """)
        self.assertEqual(function.node[0].op_type, "Identity")

    def test_parse_node_ok(self):
        node = parser.parse_node("y = Identity(x)")
        self.assertEqual(node.op_type, "Identity")

    def test_parse_float_attribute_from_int_literal(self):
        from onnx_light.onnx import AttributeProto

        model = parser.parse_model("""
            <ir_version: 9, opset_import: ["" : 18, "custom_domain" : 1]>
            agraph (float[N] x) => (float[N] out) {
                out = custom_domain.Foo<ord: float = 2>(x)
            }
            """)
        attr = model.graph.node[0].attribute[0]
        self.assertEqual(attr.type, AttributeProto.FLOAT)
        self.assertTrue(attr.has_f())
        self.assertFalse(attr.has_i())
        self.assertEqual(attr.f, 2.0)

    def test_locale_independent_float_parsing(self):
        original_locale = locale.setlocale(locale.LC_NUMERIC, None)

        is_windows = platform.system() == "Windows"
        candidates = (
            ("German_Germany.1252", "French_France.1252")
            if is_windows
            else ("de_DE.UTF-8", "fr_FR.UTF-8")
        )
        locale_set = False
        for candidate in candidates:
            try:
                locale.setlocale(locale.LC_NUMERIC, candidate)
                locale_set = True
                break
            except locale.Error:
                continue

        if not locale_set:
            locale.setlocale(locale.LC_NUMERIC, original_locale)
            self.skipTest("No locale with comma decimal separator available")

        try:
            model = parser.parse_model("""
                <ir_version: 7, opset_import: ["" : 13]>
                agraph (float[1, 5] X) => (float[1, 5] Y) {
                    Y = LeakyRelu <alpha = 0.123> (X)
                }
                """)
            node = model.graph.node[0]
            self.assertEqual(node.attribute[0].name, "alpha")
            self.assertAlmostEqual(node.attribute[0].f, 0.123, atol=1e-5)
        finally:
            locale.setlocale(locale.LC_NUMERIC, original_locale)

    def test_parse_model_error(self):
        with self.assertRaises(ValueError) as cm:
            parser.parse_model("not a valid model")
        self.assertIn("Failed to parse model", str(cm.exception))

    def test_parse_graph_error(self):
        with self.assertRaises(ValueError) as cm:
            parser.parse_graph("not a valid graph")
        self.assertIn("Failed to parse graph", str(cm.exception))

    def test_parse_function_error(self):
        with self.assertRaises(ValueError) as cm:
            parser.parse_function("not a valid function")
        self.assertIn("Failed to parse function", str(cm.exception))

    def test_parse_node_error(self):
        with self.assertRaises(ValueError) as cm:
            parser.parse_node("???not a valid node???")
        self.assertIn("Failed to parse node", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
