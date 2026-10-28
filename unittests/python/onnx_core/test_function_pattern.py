# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests fixed-arity rewrites defined with ONNX functions."""

import gc
import sys
import unittest

import numpy

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
from onnx_light.onnx import AttributeProto, TensorProto
from onnx_light.onnx_core.optimization import FunctionPattern, GraphBuilder, GraphGraph


def _function(inputs, outputs, nodes, attributes=None):
    """Creates a function body at opset 18."""
    return oh.make_function(
        "patterns",
        "body",
        inputs,
        outputs,
        nodes,
        [oh.make_opsetid("", 18)],
        attributes=attributes,
    )


def _model(nodes, outputs=("y",)):
    """Creates a model with a single floating-point input."""
    return oh.make_model(
        oh.make_graph(
            nodes,
            "graph",
            [oh.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
            [oh.make_tensor_value_info(output, TensorProto.FLOAT, [2, 2]) for output in outputs],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )


def _double_neg():
    """Creates an equivalent template/replacement pair and a target model."""
    template = _function(
        ["a"], ["c"], [oh.make_node("Neg", ["a"], ["b"]), oh.make_node("Neg", ["b"], ["c"])]
    )
    replacement = _function(["input"], ["input"], [])
    model = _model(
        [oh.make_node("Neg", ["x"], ["middle"]), oh.make_node("Neg", ["middle"], ["y"])]
    )
    return template, replacement, model


class TestFunctionPattern(ExtTestCase):
    def test_rejected_match_keeps_standalone_candidate_alive(self):
        template, replacement, model = _double_neg()
        pattern = FunctionPattern(template, replacement)
        graph = GraphGraph(GraphBuilder(model), patterns=[pattern])
        candidate = oh.make_node("Abs", ["x"], ["y"])
        references = sys.getrefcount(candidate)
        match = pattern.match(graph, candidate)
        self.assertFalse(match.matched)
        self.assertGreater(sys.getrefcount(candidate), references)
        del candidate
        gc.collect()
        self.assertIn("Abs", str(match))

    def test_double_neg_and_registration(self):
        template, replacement, model = _double_neg()
        pattern = FunctionPattern(template, replacement, name="DoubleNeg", priority=0)
        self.assertEqual(pattern.fast_op_type(), {"Neg"})
        self.assertEqual(pattern.name, "DoubleNeg")
        self.assertEqual(pattern.priority, 0)
        builder = GraphBuilder(model)
        builder.register_pattern(pattern)
        rewrites = GraphGraph(builder, patterns=["DoubleNeg"]).optimize()
        self.assertTrue(rewrites.rewrites)
        optimized = builder.to_onnx()
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Identity"])
        self.assertEqual(list(optimized.graph.node[0].output), ["y"])

    def test_double_neg_numerical(self):
        evaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")
        template, replacement, model = _double_neg()
        builder = GraphBuilder(model)
        GraphGraph(builder, patterns=[FunctionPattern(template, replacement)]).optimize()
        feeds = {"x": numpy.array([[-2, 0], [1, 3]], dtype=numpy.float32)}
        self.assertEqualArray(
            evaluator(model).run(None, feeds)[0], evaluator(builder.to_onnx()).run(None, feeds)[0]
        )

    def test_reference_attributes_and_fresh_names(self):
        template_node = oh.make_node("Transpose", ["a"], ["b"])
        template_node.attribute.append(
            oh.make_attribute_ref("perm", AttributeProto.INTS, ref_attr_name="permutation")
        )
        template = _function(["a"], ["b"], [template_node], ["permutation"])
        replacement_node = oh.make_node("Transpose", ["input"], ["middle"])
        replacement_node.attribute.append(
            oh.make_attribute_ref("perm", AttributeProto.INTS, ref_attr_name="permutation")
        )
        replacement = _function(
            ["input"],
            ["output"],
            [replacement_node, oh.make_node("Identity", ["middle"], ["output"])],
            ["permutation"],
        )
        pattern = FunctionPattern(template, replacement)
        model = _model([oh.make_node("Transpose", ["x"], ["y"], perm=[1, 0])])
        builder = GraphBuilder(model)
        graph = GraphGraph(builder, patterns=[pattern])
        match = pattern.match(graph, graph.node_before("y"))
        self.assertTrue(match.matched)
        nodes = pattern.apply(graph, match.nodes)
        self.assertEqual(list(nodes[0].attribute[0].ints), [1, 0])
        self.assertFalse(nodes[0].attribute[0].ref_attr_name)
        self.assertEqual(list(nodes[-1].output), ["y"])
        again = pattern.apply(graph, match.nodes)
        self.assertNotEqual(nodes[0].output[0], again[0].output[0])

    def test_preserves_multiple_outputs(self):
        template = _function(
            ["a"], ["b", "c"], [oh.make_node("Split", ["a"], ["b", "c"], axis=0, num_outputs=2)]
        )
        replacement = _function(
            ["input"],
            ["left", "right"],
            [oh.make_node("Split", ["input"], ["left", "right"], axis=0, num_outputs=2)],
        )
        pattern = FunctionPattern(template, replacement)
        model = _model(
            [oh.make_node("Split", ["x"], ["y", "z"], axis=0, num_outputs=2)], outputs=("y", "z")
        )
        for output in model.graph.output:
            output.type.tensor_type.shape.dim[0].dim_value = 1
        builder = GraphBuilder(model)
        graph = GraphGraph(builder, patterns=[pattern])
        match = pattern.match(graph, graph.node_before("y"))
        self.assertTrue(match.matched)
        nodes = pattern.apply(graph, match.nodes)
        self.assertEqual(list(nodes[0].output), ["y", "z"])

    def test_optimizes_reference_attributes(self):
        transpose = oh.make_node("Transpose", ["a"], ["b"])
        transpose.attribute.append(oh.make_attribute_ref("perm", AttributeProto.INTS))
        template = _function(
            ["a"],
            ["d"],
            [transpose, oh.make_node("Neg", ["b"], ["c"]), oh.make_node("Neg", ["c"], ["d"])],
            ["perm"],
        )
        replacement = _function(["a"], ["b"], [transpose], ["perm"])
        model = _model(
            [
                oh.make_node("Transpose", ["x"], ["t"], perm=[1, 0]),
                oh.make_node("Neg", ["t"], ["n"]),
                oh.make_node("Neg", ["n"], ["y"]),
            ]
        )
        builder = GraphBuilder(model)
        GraphGraph(builder, patterns=[FunctionPattern(template, replacement)]).optimize()
        nodes = builder.to_onnx().graph.node
        self.assertEqual([node.op_type for node in nodes], ["Transpose"])
        self.assertEqual(list(nodes[0].attribute[0].ints), [1, 0])
        self.assertEqual(list(nodes[0].input), ["x"])
        self.assertEqual(list(nodes[0].output), ["y"])

    def test_rejects_subgraph_capture(self):
        template = _function(
            ["a"], ["c"], [oh.make_node("Neg", ["a"], ["b"]), oh.make_node("Neg", ["b"], ["c"])]
        )
        pattern = FunctionPattern(template, _function(["a"], ["a"], []))
        branch = oh.make_graph(
            [oh.make_node("Identity", ["middle"], ["branch_output"])],
            "branch",
            [],
            [oh.make_tensor_value_info("branch_output", TensorProto.FLOAT, [2, 2])],
        )
        model = _model(
            [
                oh.make_node("Neg", ["x"], ["middle"]),
                oh.make_node("Neg", ["middle"], ["y"]),
                oh.make_node("If", ["condition"], ["z"], then_branch=branch, else_branch=branch),
            ],
            outputs=("y", "z"),
        )
        model.graph.input.append(oh.make_tensor_value_info("condition", TensorProto.BOOL, []))
        graph = GraphGraph(GraphBuilder(model), patterns=[pattern])
        self.assertFalse(pattern.match(graph, graph.node_before("y")).matched)

    def test_rejects_arity_and_uncaptured_reference(self):
        template = _function(["a"], ["b"], [oh.make_node("Neg", ["a"], ["b"])])
        with self.assertRaisesRegex(ValueError, "arities"):
            FunctionPattern(template, _function(["a", "extra"], ["a"], []))
        node = oh.make_node("Transpose", ["a"], ["b"])
        node.attribute.append(oh.make_attribute_ref("perm", AttributeProto.INTS))
        with self.assertRaisesRegex(ValueError, "captured"):
            FunctionPattern(template, _function(["a"], ["b"], [node], ["perm"]))


if __name__ == "__main__":
    unittest.main()
