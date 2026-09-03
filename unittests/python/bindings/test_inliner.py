# source: https://github.com/onnx/onnx/blob/main/onnx/test/inliner_test.py
"""Tests for onnx_light.onnx.inliner, adapted from the ONNX reference inliner tests."""

from __future__ import annotations

import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx.checker as checker
import onnx_light.onnx.defs as defs
import onnx_light.onnx.inliner as inliner
import onnx_light.onnx.parser as parser


class TestInliner(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        # Register standard ONNX op schemas so schema lookups work in all tests.
        defs.register_onnx_operator_set_schema()

    def test_basic(self):
        """Inlines nested local functions: foo(x)=Add(x,x)|bar(t)=Mul(t,t)."""
        model = parser.parse_model("""
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = local.bar(temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul (x, x)
            }
        """)
        inlined = inliner.inline_local_functions(model)
        inlined_nodes = list(inlined.graph.node)
        # function-call should be replaced by Add, followed by Mul
        self.assertEqual(len(inlined_nodes), 2)
        self.assertEqual(inlined_nodes[0].op_type, "Add")
        self.assertEqual(inlined_nodes[1].op_type, "Mul")

    def test_selective_inlining(self):
        """Inlines only the 'square' function, leaving 'double_and_square' uninlined."""
        model = parser.parse_model("""
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.square (X)
                Y = local.double_and_square (T)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            double_and_square (x) => (y) {
                double = Add(x, x)
                y = local.square(double)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            square (x) => (y) {
                y = Mul (x, x)
            }
        """)
        inlined = inliner.inline_selected_functions(model, [("local", "square")], exclude=False)
        inlined_nodes = list(inlined.graph.node)
        # function-call to square should be replaced by Mul, but not double_and_square
        self.assertEqual(len(inlined_nodes), 2)
        self.assertEqual(inlined_nodes[0].op_type, "Mul")
        self.assertEqual(inlined_nodes[1].op_type, "double_and_square")

        # check call to square inside double_and_square was inlined:
        function_nodes = list(inlined.functions[0].node)
        self.assertEqual(len(function_nodes), 2)
        self.assertEqual(function_nodes[0].op_type, "Add")
        self.assertEqual(function_nodes[1].op_type, "Mul")

    def test_selective_exclusion(self):
        """Inlines all functions except 'double_and_square' (exclude mode)."""
        model = parser.parse_model("""
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.square (X)
                Y = local.double_and_square (T)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            double_and_square (x) => (y) {
                double = Add(x, x)
                y = local.square(double)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            square (x) => (y) {
                y = Mul (x, x)
            }
        """)
        inlined = inliner.inline_selected_functions(
            model, [("local", "double_and_square")], exclude=True
        )
        inlined_nodes = list(inlined.graph.node)
        # function-call to square should be replaced by Mul, but not double_and_square
        self.assertEqual(len(inlined_nodes), 2)
        self.assertEqual(inlined_nodes[0].op_type, "Mul")
        self.assertEqual(inlined_nodes[1].op_type, "double_and_square")

        # check call to square inside double_and_square was inlined:
        function_nodes = list(inlined.functions[0].node)
        self.assertEqual(len(function_nodes), 2)
        self.assertEqual(function_nodes[0].op_type, "Add")
        self.assertEqual(function_nodes[1].op_type, "Mul")

    def test_inline_rejects_cyclic_function(self):
        """Verifies that inline_local_functions raises ValidationError for cyclic functions."""
        model = parser.parse_model("""
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y) { Y = local.foo (X) }
            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) { y = local.foo (x) }
        """)
        self.assertRaises(checker.ValidationError, inliner.inline_local_functions, model)

    def test_schema_function_inlining(self):
        """Inlines schema-defined functions (e.g. Softsign) when inline_schema_functions=True."""
        import onnx_light.onnx.defs as defs

        if not defs.has_schema("Softsign", 20):
            self.skipTest("Softsign schema not available in this build")
            return
        schema = defs.get_schema("Softsign", 20)

        if not schema.has_function and not schema.has_context_dependent_function:
            self.skipTest("Softsign has no function body in this build")
            return

        model = parser.parse_model("""
            <ir_version: 8, opset_import: [ "" : 20]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = Softsign (X)
            }
        """)
        inlined = inliner.inline_selected_functions(
            model, [], exclude=True, inline_schema_functions=True
        )
        inlined_nodes = list(inlined.graph.node)
        self.assertIn("Abs", [n.op_type for n in inlined_nodes])

    def test_sequence_map_inlining_rejects_missing_input_type(self):
        model = parser.parse_model("""
            <ir_version: 13, opset_import: [ "" : 17 ]>
            missing_sequence_type (seq(float) input) => (seq(float) output) {
                output = SequenceMap (input) <
                    body = body (float body_input) => (float body_output) {
                        body_output = Identity(body_input)
                    }
                >
            }
        """)
        model.graph.input[0].ClearField("type")
        model.graph.output[0].ClearField("type")

        with self.assertRaisesRegex(ValueError, "Expected a sequence type"):
            inliner.inline_selected_functions(
                model, [], exclude=True, inline_schema_functions=True
            )

    def test_inline_ignores_constant_without_outputs(self):
        model = parser.parse_model("""
            <ir_version: 13, opset_import: [ "" : 13, "local" : 1 ]>
            constant_without_outputs (float[1] X) => (float[1] Y) {
                = Constant()
                Y = local.identity(X)
            }

            <opset_import: [ "" : 12 ], domain: "local">
            identity (X) => (Y) {
                Y = Identity(X)
            }
        """)

        inlined = inliner.inline_local_functions(model, convert_version=True)

        self.assertEqual([node.op_type for node in inlined.graph.node], ["Constant", "Identity"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
