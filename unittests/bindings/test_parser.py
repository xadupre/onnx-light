from __future__ import annotations

import unittest

from onnx_light.onnx import parser
from onnx_light.ext_test_case import ExtTestCase


_MODEL_TEXT = """
<ir_version: 7, opset_import: ["": 17]>
agraph (float[N] x) => (float[N] y) {
    y = Identity(x)
}
"""

_GRAPH_TEXT = """
agraph (float[N] x) => (float[N] y) {
    y = Identity(x)
}
"""

_FUNCTION_TEXT = """
<opset_import: ["": 17], domain: "local">
afun (x) => (y) {
    y = Identity(x)
}
"""

_NODE_TEXT = "y = Identity(x)"


class TestParser(ExtTestCase):
    def test_parse_model_ok(self):
        model = parser.parse_model(_MODEL_TEXT)
        self.assertEqual(model.graph.node[0].op_type, "Identity")

    def test_parse_graph_ok(self):
        graph = parser.parse_graph(_GRAPH_TEXT)
        self.assertEqual(graph.node[0].op_type, "Identity")

    def test_parse_function_ok(self):
        function = parser.parse_function(_FUNCTION_TEXT)
        self.assertEqual(function.node[0].op_type, "Identity")

    def test_parse_node_ok(self):
        node = parser.parse_node(_NODE_TEXT)
        self.assertEqual(node.op_type, "Identity")

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
