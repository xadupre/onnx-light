from __future__ import annotations

import locale
import platform
import unittest

from onnx_light.onnx import TensorProto, parser
from onnx_light.ext_test_case import ExtTestCase


class TestParser(ExtTestCase):
    def test_parse_graph_whitespace_comments(self):
        text = (
            " \t\n\v\f\r# leading comment\n"
            "agraph (float[N] x) => (float[N] y) { y = Identity(x) }"
            "# trailing comment"
        )
        graph = parser.parse_graph(text)
        self.assertEqual(graph.node[0].op_type, "Identity")

    def test_parse_graph_error_context(self):
        text = (
            "agraph (float[N] x) => (float[N] y) {\n"
            "  T = MatMul[X, W]\n"
            "  y = Identity(T)\n}"
        )
        with self.assertRaisesRegex(ValueError, r"Error context: +T = MatMul\[X, W\]"):
            parser.parse_graph(text)

    def test_incomplete_string_literal(self):
        for literal in ['"unterminated', '"trailing\\', '"escaped\\"']:
            with self.subTest(literal=literal):
                text = f"agraph (float[{literal}] x) => (float[N] y) {{ y = Identity(x) }}"
                with self.assertRaisesRegex(ValueError, "Incomplete string literal"):
                    parser.parse_graph(text)

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

    def test_string_literals_preserve_bytes(self):
        for value in ("", "before\0after", 'quote: " backslash: \\ newline: \n', "é中🙂"):
            quoted = value.replace("\\", "\\\\").replace('"', '\\"')
            body = f'Y = Constant<value_string="{quoted}">()'
            graph = f"g () => () {{ {body} }}"
            function = f'<opset_import: ["": 18]> f () => () {{ {body} }}'
            model = f'<ir_version: 8, opset_import: ["": 18]> {graph}'
            for parse, text in (
                (parser.parse_node, body),
                (parser.parse_graph, graph),
                (parser.parse_function, function),
                (parser.parse_model, model),
            ):
                with self.subTest(value=value, parse=parse.__name__):
                    result = parse(text)
                    if parse == parser.parse_model:
                        result = result.graph
                    node = result if parse == parser.parse_node else result.node[0]
                    self.assertEqual(node.attribute[0].s, value.encode("utf-8"))

    def test_numeric_literal_errors(self):
        for body in (
            "value_int=9223372036854775808",
            "value_int=-9223372036854775809",
            "value=uint64[1] {-1}",
            "value=uint64[1] {-0}",
            "value=uint64[1] {18446744073709551616}",
            "value_float: float=1e1000",
            "value=double[1] {1e1000}",
            "value_float: float=1e+",
        ):
            with (
                self.subTest(body=body),
                self.assertRaisesRegex(ValueError, "Failed to parse node"),
            ):
                parser.parse_node(f"Y = Constant<{body}>()")

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
