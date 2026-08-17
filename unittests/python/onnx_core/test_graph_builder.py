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

    def test_remove_duplicate_initializers_across_scopes(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 2])
        w = oh.make_tensor("w", FLOAT, [2, 2], [1.0, 1.0, 1.0, 1.0])
        builder.make_initializer(w)
        (a,) = builder.make_node("Add", ["x", "w"])
        builder.make_output(a)

        # The subgraph declares its own initializer with the same content; it is
        # collapsed onto the enclosing "w" since a subgraph body sees the outer
        # scope.
        body = builder.make_subgraph("body")
        body.make_input("b", FLOAT, [2, 2])
        c = oh.make_tensor("c", FLOAT, [2, 2], [1.0, 1.0, 1.0, 1.0])
        body.make_initializer(c)
        (s,) = body.make_node("Add", ["b", "c"])
        body.make_output(s)

        self.assertEqual(builder.remove_duplicate_initializers(), 1)
        graph = builder.build_graph()
        self.assertEqual(len(graph.initializer), 1)
        self.assertEqual(graph.initializer[0].name, "w")
        body_graph = body.build_graph()
        self.assertEqual(len(body_graph.initializer), 0)
        self.assertEqual(body_graph.node[0].input[1], "w")

    def test_remove_identity_nodes(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        # A chain of two identities feeding a live consumer.
        (i1,) = builder.make_node("Identity", ["x"])
        (i2,) = builder.make_node("Identity", [i1])
        (used,) = builder.make_node("Neg", [i2])
        builder.make_output(used)

        self.assertEqual(len(builder.build_graph().node), 3)
        self.assertEqual(builder.remove_identity_nodes(), 2)
        graph = builder.build_graph()
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].op_type, "Neg")
        # The chain collapses onto the original producer.
        self.assertEqual(graph.node[0].input[0], "x")
        # A second pass has nothing left to remove.
        self.assertEqual(builder.remove_identity_nodes(), 0)

    def test_remove_identity_nodes_keeps_graph_output(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        # The identity output is a declared graph output, so it is kept.
        (out,) = builder.make_node("Identity", ["x"])
        builder.make_output(out)

        self.assertEqual(builder.remove_identity_nodes(), 0)
        graph = builder.build_graph()
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].op_type, "Identity")

    def test_remove_identity_nodes_recurses_into_subgraphs(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        (top,) = builder.make_node("Neg", ["x"])
        builder.make_output(top)

        body = builder.make_subgraph("body")
        body.make_input("a", FLOAT, [2, 3])
        (inner,) = body.make_node("Identity", ["a"])
        (body_used,) = body.make_node("Neg", [inner])
        body.make_output(body_used)

        self.assertEqual(builder.remove_identity_nodes(), 1)
        body_graph = body.build_graph()
        self.assertEqual(len(body_graph.node), 1)
        self.assertEqual(body_graph.node[0].op_type, "Neg")
        self.assertEqual(body_graph.node[0].input[0], "a")

    def test_remove_duplicate_nodes(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        # Two Neg nodes computing the same value feed a shared consumer.
        (a,) = builder.make_node("Neg", ["x"])
        (b,) = builder.make_node("Neg", ["x"])
        (s,) = builder.make_node("Add", [a, b])
        builder.make_output(s)

        self.assertEqual(builder.remove_duplicate_nodes(), 1)
        graph = builder.build_graph()
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0].op_type, "Neg")
        self.assertEqual(graph.node[1].op_type, "Add")
        # Both Add inputs now read from the surviving Neg.
        self.assertEqual(graph.node[1].input[0], a)
        self.assertEqual(graph.node[1].input[1], a)
        # A second pass has nothing left to collapse.
        self.assertEqual(builder.remove_duplicate_nodes(), 0)

    def test_remove_duplicate_nodes_collapses_branches(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        # Two identical branches collapse level by level in a single pass.
        (a1,) = builder.make_node("Mul", ["x", "x"])
        (b1,) = builder.make_node("Mul", ["x", "x"])
        (a2,) = builder.make_node("Relu", [a1])
        (b2,) = builder.make_node("Relu", [b1])
        (s,) = builder.make_node("Add", [a2, b2])
        builder.make_output(s)

        self.assertEqual(builder.remove_duplicate_nodes(), 2)
        graph = builder.build_graph()
        self.assertEqual([n.op_type for n in graph.node], ["Mul", "Relu", "Add"])
        self.assertEqual(graph.node[2].input[0], a2)
        self.assertEqual(graph.node[2].input[1], a2)

    def test_remove_duplicate_nodes_keeps_distinct_attributes(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        (a,) = builder.make_node("LeakyRelu", ["x"], attributes={"alpha": 0.1})
        (b,) = builder.make_node("LeakyRelu", ["x"], attributes={"alpha": 0.2})
        (s,) = builder.make_node("Add", [a, b])
        builder.make_output(s)

        self.assertEqual(builder.remove_duplicate_nodes(), 0)
        self.assertEqual(len(builder.build_graph().node), 3)

    def test_remove_duplicate_nodes_keeps_graph_outputs(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        # Both duplicates are declared graph outputs, so neither can be dropped.
        (a,) = builder.make_node("Neg", ["x"])
        (b,) = builder.make_node("Neg", ["x"])
        builder.make_output(a)
        builder.make_output(b)

        self.assertEqual(builder.remove_duplicate_nodes(), 0)
        self.assertEqual(len(builder.build_graph().node), 2)

    def test_remove_duplicate_nodes_recurses_into_subgraphs(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        (top,) = builder.make_node("Neg", ["x"])
        builder.make_output(top)

        body = builder.make_subgraph("body")
        body.make_input("a", FLOAT, [2, 3])
        (n1,) = body.make_node("Neg", ["a"])
        (n2,) = body.make_node("Neg", ["a"])
        (s,) = body.make_node("Add", [n1, n2])
        body.make_output(s)

        self.assertEqual(builder.remove_duplicate_nodes(), 1)
        body_graph = body.build_graph()
        self.assertEqual(len(body_graph.node), 2)
        self.assertEqual(body_graph.node[1].op_type, "Add")
        self.assertEqual(body_graph.node[1].input[0], n1)
        self.assertEqual(body_graph.node[1].input[1], n1)

    def test_move_shape_and_size_nodes(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        (p,) = builder.make_node("Neg", ["x"])
        (a,) = builder.make_node("Add", [p, p])
        # Shape reads p but is inserted after an unrelated node; it must move
        # right after its producer, and the Size chained on it follows.
        (sh,) = builder.make_node("Shape", [p])
        (sz,) = builder.make_node("Size", [sh])
        builder.make_output(a)
        builder.make_output(sz)

        self.assertEqual(
            [n.op_type for n in builder.build_graph().node], ["Neg", "Add", "Shape", "Size"]
        )
        self.assertEqual(builder.move_shape_and_size_nodes(), 2)
        self.assertEqual(
            [n.op_type for n in builder.build_graph().node], ["Neg", "Shape", "Size", "Add"]
        )
        # A second pass leaves the already-tightened graph untouched.
        self.assertEqual(builder.move_shape_and_size_nodes(), 0)

    def test_move_shape_and_size_nodes_keeps_graph_input_reader(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        # Shape reads a graph input (no producing node), so it stays in place.
        (n,) = builder.make_node("Neg", ["x"])
        (sh,) = builder.make_node("Shape", ["x"])
        builder.make_output(n)
        builder.make_output(sh)

        self.assertEqual(builder.move_shape_and_size_nodes(), 0)
        self.assertEqual([n.op_type for n in builder.build_graph().node], ["Neg", "Shape"])

    def test_move_shape_and_size_nodes_runs_on_export(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        (p,) = builder.make_node("Neg", ["x"])
        (a,) = builder.make_node("Add", [p, p])
        (sh,) = builder.make_node("Shape", [p])
        builder.make_output(a)
        builder.make_output(sh)

        # to_onnx hoists Shape next to its producer before exporting.
        model = builder.to_onnx("model")
        self.assertEqual(
            [n.op_type for n in model.graph.node], ["Neg", "Shape", "Add"]
        )

    def test_move_shape_and_size_nodes_recurses_into_subgraphs(self):
        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        (top,) = builder.make_node("Neg", ["x"])
        builder.make_output(top)

        body = builder.make_subgraph("body")
        body.make_input("a", FLOAT, [2, 3])
        (p,) = body.make_node("Neg", ["a"])
        (u,) = body.make_node("Add", [p, p])
        (sh,) = body.make_node("Shape", [p])
        body.make_output(u)
        body.make_output(sh)

        self.assertEqual(builder.move_shape_and_size_nodes(), 1)
        self.assertEqual(
            [n.op_type for n in body.build_graph().node], ["Neg", "Shape", "Add"]
        )

    def test_inline_local_functions(self):
        builder = GraphBuilder("g")
        # A local function computing Neg(Add(a, b)).
        fct = builder.make_local_function("MyFct", domain="custom")
        fct.make_input("a", FLOAT, [2, 3])
        fct.make_input("b", FLOAT, [2, 3])
        (s,) = fct.make_node("Add", ["a", "b"])
        (n,) = fct.make_node("Neg", [s])
        fct.make_output(n)

        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("z", FLOAT, [2, 3])
        (out,) = builder.make_node("MyFct", ["x", "z"], domain="custom")
        builder.make_output(out)

        self.assertEqual(builder.inline_local_functions(), 1)
        graph = builder.build_graph()
        # The call node is replaced by the two body nodes.
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0].op_type, "Add")
        self.assertEqual(list(graph.node[0].input), ["x", "z"])
        self.assertEqual(graph.node[1].op_type, "Neg")
        self.assertEqual(graph.node[1].output[0], out)
        # The fully inlined function definition is dropped from the model.
        self.assertFalse(builder.has_local_function("MyFct"))
        model = builder.to_onnx("model")
        self.assertEqual(len(model.functions), 0)
        # A second pass has nothing left to inline.
        self.assertEqual(builder.inline_local_functions(), 0)

    def test_inline_local_functions_keeps_uncalled_function(self):
        builder = GraphBuilder("g")
        fct = builder.make_local_function("MyFct", domain="custom")
        fct.make_input("a", FLOAT, [2, 3])
        (n,) = fct.make_node("Neg", ["a"])
        fct.make_output(n)

        builder.make_input("x", FLOAT, [2, 3])
        (top,) = builder.make_node("Neg", ["x"])
        builder.make_output(top)

        # No call site, so nothing is inlined and the definition is left in place.
        self.assertEqual(builder.inline_local_functions(), 0)
        self.assertTrue(builder.has_local_function("MyFct"))

    def test_inline_local_functions_include_exclude(self):
        def build():
            builder = GraphBuilder("g")
            keep = builder.make_local_function("Keep", domain="custom")
            keep.make_input("a", FLOAT, [2, 3])
            (kn,) = keep.make_node("Neg", ["a"])
            keep.make_output(kn)

            drop = builder.make_local_function("Drop", domain="custom")
            drop.make_input("a", FLOAT, [2, 3])
            (dn,) = drop.make_node("Neg", ["a"])
            drop.make_output(dn)

            builder.make_input("x", FLOAT, [2, 3])
            (kc,) = builder.make_node("Keep", ["x"], domain="custom")
            (dc,) = builder.make_node("Drop", [kc], domain="custom")
            builder.make_output(dc)
            return builder

        # include: only "Drop" is inlined; a (domain, name) tuple selects it.
        builder = build()
        self.assertEqual(builder.inline_local_functions(include=[("custom", "Drop")]), 1)
        self.assertTrue(builder.has_local_function("Keep"))
        self.assertFalse(builder.has_local_function("Drop"))

        # exclude: everything except "Keep" is inlined.
        builder = build()
        self.assertEqual(builder.inline_local_functions(exclude=[("custom", "Keep")]), 1)
        self.assertTrue(builder.has_local_function("Keep"))
        self.assertFalse(builder.has_local_function("Drop"))

        # An empty domain matches every function sharing the name.
        builder = build()
        self.assertEqual(builder.inline_local_functions(include=[("", "Drop")]), 1)
        self.assertTrue(builder.has_local_function("Keep"))
        self.assertFalse(builder.has_local_function("Drop"))

        # An empty name matches every function in the domain (here both).
        builder = build()
        self.assertEqual(builder.inline_local_functions(include=[("custom", "")]), 2)
        self.assertFalse(builder.has_local_function("Keep"))
        self.assertFalse(builder.has_local_function("Drop"))

        # Passing both include and exclude is rejected.
        builder = build()
        with self.assertRaises(ValueError):
            builder.inline_local_functions(
                include=[("custom", "Keep")], exclude=[("custom", "Drop")]
            )

    def test_constant_folding_options_are_exposed(self):
        from onnx_light.onnx_core.graph_builder import ConstantFoldingOptions

        options = ConstantFoldingOptions()
        # Defaults mirror the C++ ConstantFoldingOptions struct.
        self.assertTrue(options.enabled)
        self.assertEqual(options.max_element_count, -1)
        self.assertTrue(options.fold_weights)
        self.assertFalse(options.raise_on_missing_weight_kernel)
        self.assertEqual(options.excluded_ops, set())

        # Every field is writable from Python.
        options.enabled = False
        options.max_element_count = 1024
        options.fold_weights = False
        options.raise_on_missing_weight_kernel = True
        options.excluded_ops = {("", "RandomNormal")}
        self.assertFalse(options.enabled)
        self.assertEqual(options.max_element_count, 1024)
        self.assertFalse(options.fold_weights)
        self.assertTrue(options.raise_on_missing_weight_kernel)
        self.assertEqual(options.excluded_ops, {("", "RandomNormal")})

    def test_constant_fold_is_callable(self):
        from onnx_light.onnx_core.graph_builder import ConstantFoldingOptions

        builder = GraphBuilder("g")
        builder.make_input("x", FLOAT, [2, 3])
        builder.make_input("y", FLOAT, [2, 3])
        (z,) = builder.make_node("Add", ["x", "y"])
        builder.make_output(z)

        # Nothing is constant here, so folding is a no-op regardless of the
        # kernels linked into this build variant.
        self.assertEqual(builder.constant_fold(), 0)
        # A disabled pass is always a no-op and accepts an explicit options object.
        options = ConstantFoldingOptions()
        options.enabled = False
        self.assertEqual(builder.constant_fold(options), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
