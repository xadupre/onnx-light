# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Python :class:`onnx_light.onnx_core.graph_builder.GraphBuilder`.

These exercise the incremental builder API directly (naming, node creation,
opset resolution, incremental shape inference, initializers, nested builders and
finalisation). They mirror the C++ counterpart in
``unittests/cc/onnx_core/builder/test_graph_builder.cc``.
"""

from __future__ import annotations

import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

# The builder extension ships in the core build; skip the module when it (or the
# schema library it relies on) is unavailable.
GraphBuilder = import_or_skip("onnx_light.onnx_core.graph_builder", "GraphBuilder")

FLOAT = onnxl.TensorProto.FLOAT


class TestGraphBuilder(ExtTestCase):
    def test_starts_empty(self):
        builder = GraphBuilder("g")
        self.assertFalse(builder.has_name("x"))

    def test_names_are_never_reused(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        self.assertTrue(builder.has_name("x"))
        self.assertRaise(lambda: builder.make_input("x", FLOAT, [2, 3]), ValueError)

    def test_unique_name_never_collides(self):
        builder = GraphBuilder("g")
        a = builder.unique_name("t")
        b = builder.unique_name("t")
        self.assertNotEqual(a, b)
        self.assertTrue(builder.has_name(a))
        self.assertTrue(builder.has_name(b))

    def test_make_node_resolves_opset_and_infers_shape(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        outputs = builder.make_node("Add", ["x", "y"])
        self.assertEqual(len(outputs), 1)
        self.assertTrue(builder.has_name(outputs[0]))
        # The default ONNX opset was resolved from the operator schemas.
        self.assertGreater(builder.opset_version(""), 0)
        # The output shape was inferred incrementally.
        self.assertTrue(builder.has_shape(outputs[0]))
        self.assertEqual(builder.get_shape(outputs[0]).shape.rank(), 2)

    def test_make_node_uses_provided_output_name(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        outputs = builder.make_node("Add", ["x", "y"], ["z"])
        self.assertEqual(outputs, ["z"])

    def test_make_node_rejects_unknown_input(self):
        builder = GraphBuilder("g")
        self.assertRaise(lambda: builder.make_node("Add", ["missing", "y"]), ValueError)

    def test_make_node_with_attributes(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        outputs = builder.make_node("Softmax", ["x"], attributes={"axis": 1})
        self.assertEqual(len(outputs), 1)
        self.assertTrue(builder.has_shape(outputs[0]))

    def test_external_initializer_is_recorded(self):
        builder = GraphBuilder("g")
        name = builder.make_external_initializer("w", FLOAT, [4, 4], "weights.bin", 0, 64)
        self.assertEqual(name, "w")
        self.assertTrue(builder.has_name("w"))

    def test_to_model_produces_graph_with_opset(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (z,) = builder.make_node("Add", ["x", "y"])
        builder.make_output(z)
        model = builder.to_onnx("model")
        self.assertGreater(model.ir_version, 0)
        self.assertEqual(len(model.graph.node), 1)
        self.assertGreater(len(model.opset_import), 0)

    def test_to_graph_and_to_function(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (z,) = builder.make_node("Add", ["x", "y"])
        builder.make_output(z)
        graph = builder.to_onnx("graph")
        self.assertEqual(len(graph.node), 1)
        function = builder.to_onnx("function", domain="custom")
        self.assertEqual(len(function.node), 1)

    def test_to_function_rejects_initializers(self):
        builder = GraphBuilder("g")
        builder.make_external_initializer("w", FLOAT, [2, 2], "w.bin", 0, 16)
        self.assertRaise(lambda: builder.to_onnx("function", domain="custom"), ValueError)

    def test_to_onnx_rejects_unknown_kind(self):
        builder = GraphBuilder("g")
        self.assertRaise(lambda: builder.to_onnx("unknown"), ValueError)

    def test_explicit_opset_is_preserved(self):
        builder = GraphBuilder("g")
        builder.set_opset_version("", 17)
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        builder.make_node("Add", ["x", "y"])
        self.assertEqual(builder.opset_version(""), 17)

    def test_input_output_from_proto(self):
        builder = GraphBuilder("g")
        vi = oh.make_tensor_value_info("x", FLOAT, [2, 3])
        builder.make_input(vi)
        self.assertTrue(builder.has_name("x"))
        self.assertTrue(builder.has_shape("x"))
        builder.make_output("x")

    def test_to_string_describes_content(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (z,) = builder.make_node("Add", ["x", "y"])
        builder.make_output(z)
        text = builder.to_string()
        self.assertIn("GraphBuilder(name=g)", text)
        self.assertIn("inputs (2)", text)
        self.assertIn("nodes (1)", text)
        self.assertIn("Add(", text)
        self.assertEqual(text, str(builder))

    def test_local_function_is_a_nested_builder(self):
        builder = GraphBuilder("g")
        fct = builder.make_local_function("MyFct", "custom")
        fct.make_input("a", FLOAT, [2, 3])
        fct.make_input("b", FLOAT, [2, 3])
        (out,) = fct.make_node("Add", ["a", "b"])
        fct.make_output(out)
        self.assertRaise(lambda: builder.make_local_function("MyFct"), ValueError)

        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("z", FLOAT, [2, 3])
        (top,) = builder.make_node("MyFct", ["x", "z"], domain="custom")
        builder.make_output(top)
        model = builder.to_onnx("model")
        self.assertEqual(len(model.functions), 1)
        self.assertEqual(model.functions[0].name, "MyFct")

    def test_subgraph_is_a_nested_builder(self):
        builder = GraphBuilder("g")
        body = builder.make_subgraph("body")
        body.make_input("a", FLOAT, [2, 3])
        self.assertRaise(lambda: builder.make_subgraph("body"), ValueError)

    def test_remove_unused_nodes(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (used,) = builder.make_node("Add", ["x", "y"])
        # Two chained dead-end nodes that no output depends on.
        (dead,) = builder.make_node("Mul", ["x", "y"])
        builder.make_node("Neg", [dead])
        builder.make_output(used)

        self.assertEqual(len(builder.build_graph().node), 3)
        self.assertEqual(builder.remove_unused_nodes(), 2)
        graph = builder.build_graph()
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].op_type, "Add")
        # A second pass has nothing left to remove.
        self.assertEqual(builder.remove_unused_nodes(), 0)

    def test_remove_duplicate_initializers(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 2])
        w1 = oh.make_tensor("w1", FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0])
        w2 = oh.make_tensor("w2", FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0])
        builder.make_initializer(w1)
        builder.make_initializer(w2)
        (a,) = builder.make_node("Add", ["x", "w1"])
        (b,) = builder.make_node("Add", ["x", "w2"])
        builder.make_output(a)
        builder.make_output(b)

        self.assertEqual(builder.remove_duplicate_initializers(), 1)
        graph = builder.build_graph()
        self.assertEqual(len(graph.initializer), 1)
        self.assertEqual(graph.initializer[0].name, "w1")
        # The reference to the dropped duplicate is rewritten to the survivor.
        self.assertEqual(graph.node[1].input[1], "w1")
        # A second pass has nothing left to collapse.
        self.assertEqual(builder.remove_duplicate_initializers(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
