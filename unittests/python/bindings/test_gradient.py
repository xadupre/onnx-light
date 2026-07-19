# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the onnx_gradient Python bindings.

These tests verify that the Python-exposed gradient functions produce
FunctionProtos with the correct structure for simple ONNX graphs.

The tests do not import onnx directly (ci_no_onnx compatibility).
"""

import re
import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx_proto._helper import make_node


def _import_gradient():
    """Imports the gradient module; skips if not available (reduced build)."""
    try:
        from onnx_light.onnx_gradient import GradRegistry, gradient_of_function, gradient_of_nodes

        return gradient_of_nodes, gradient_of_function, GradRegistry
    except ImportError:
        return None, None, None


class TestGradientBindings(ExtTestCase):
    def setUp(self):
        gon, gof, gr = _import_gradient()
        if gon is None:
            self.skipTest("onnx_gradient bindings not available (reduced build)")
        self.gradient_of_nodes = gon
        self.gradient_of_function = gof
        self.GradRegistry = gr

    # ------------------------------------------------------------------ #
    # gradient_of_nodes: single MatMul                                    #
    # ------------------------------------------------------------------ #

    def test_matmul_grad_w(self):
        """Gradient of y = X @ W w.r.t. W."""
        nodes = [make_node("MatMul", ["X", "W"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W"], initializers=[], xs=["W"], y="y", zs=["X"]
        )
        # Inputs: W, X, dy
        self.assertIn("W", list(grad.input))
        self.assertIn("X", list(grad.input))
        self.assertIn("dy", list(grad.input))
        # Output: grad_W
        self.assertEqual(list(grad.output), ["grad_W"])
        # Backward nodes must include Transpose and MatMul
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("Transpose", op_types)
        self.assertIn("MatMul", op_types)

    def test_matmul_grad_xw(self):
        """Gradient of y = X @ W w.r.t. both X and W."""
        nodes = [make_node("MatMul", ["X", "W"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W"], initializers=[], xs=["X", "W"], y="y", zs=[]
        )
        self.assertEqual(list(grad.output), ["grad_X", "grad_W"])

    # ------------------------------------------------------------------ #
    # gradient_of_nodes: linear regression (MatMul + Add)                #
    # ------------------------------------------------------------------ #

    def test_linear_regression_with_bias(self):
        """Gradient of y = X @ W + b w.r.t. W and b."""
        nodes = [make_node("MatMul", ["X", "W"], ["mm"]), make_node("Add", ["mm", "b"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W", "b"], initializers=[], xs=["W", "b"], y="y", zs=["X"]
        )
        outputs = list(grad.output)
        self.assertEqual(outputs, ["grad_W", "grad_b"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("Transpose", op_types)
        self.assertIn("MatMul", op_types)

    # ------------------------------------------------------------------ #
    # gradient_of_nodes: Sub                                              #
    # ------------------------------------------------------------------ #

    def test_sub_grad(self):
        """Gradient of C = A - B w.r.t. both inputs."""
        nodes = [make_node("Sub", ["A", "B"], ["C"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["A", "B"], initializers=[], xs=["A", "B"], y="C", zs=[]
        )
        self.assertEqual(list(grad.output), ["grad_A", "grad_B"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("Neg", op_types)

    # ------------------------------------------------------------------ #
    # gradient_of_nodes: Mul                                              #
    # ------------------------------------------------------------------ #

    def test_mul_grad(self):
        """Gradient of C = A * B w.r.t. both inputs."""
        nodes = [make_node("Mul", ["A", "B"], ["C"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["A", "B"], initializers=[], xs=["A", "B"], y="C", zs=[]
        )
        self.assertEqual(list(grad.output), ["grad_A", "grad_B"])
        op_types = [str(n.op_type) for n in grad.node]
        # At least two Mul nodes: dA = dC * B, dB = dC * A
        self.assertGreaterEqual(op_types.count("Mul"), 2)

    # ------------------------------------------------------------------ #
    # gradient_of_nodes: error cases                                      #
    # ------------------------------------------------------------------ #

    def test_error_y_not_produced(self):
        """Raises ValueError when y is not produced by any node."""
        nodes = [make_node("MatMul", ["X", "W"], ["y"])]
        with self.assertRaises((ValueError, RuntimeError, Exception)):
            self.gradient_of_nodes(
                nodes=nodes,
                inputs=["X", "W"],
                initializers=[],
                xs=["W"],
                y="z",  # not produced
                zs=["X"],
            )

    def test_error_empty_xs(self):
        """Raises ValueError when xs is empty."""
        nodes = [make_node("MatMul", ["X", "W"], ["y"])]
        with self.assertRaises((ValueError, RuntimeError, Exception)):
            self.gradient_of_nodes(
                nodes=nodes, inputs=["X", "W"], initializers=[], xs=[], y="y", zs=["X"]  # empty
            )

    # ------------------------------------------------------------------ #
    # gradient_of_function                                                #
    # ------------------------------------------------------------------ #

    def test_gradient_of_function_basic(self):
        """Gradient of a FunctionProto representing y = X @ W."""
        from onnx_light.onnx_proto._helper import make_function, make_opsetid

        func = make_function(
            domain="",
            fname="linear",
            inputs=["X", "W"],
            outputs=["y"],
            nodes=[make_node("MatMul", ["X", "W"], ["y"])],
            opset_imports=[make_opsetid("", 21)],
        )
        grad = self.gradient_of_function(function=func, xs=["W"], y="y", zs=["X"])
        self.assertEqual(str(grad.name), "linear_grad")
        self.assertEqual(list(grad.output), ["grad_W"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("Transpose", op_types)
        self.assertIn("MatMul", op_types)

    def test_function_opset_import(self):
        """The returned FunctionProto imports at least one opset."""
        nodes = [make_node("MatMul", ["X", "W"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W"], initializers=[], xs=["W"], y="y", zs=["X"]
        )
        self.assertGreater(len(list(grad.opset_import)), 0)


# ------------------------------------------------------------------ #
# Backend gradient tests via make_test_class                         #
# ------------------------------------------------------------------ #


def _camel_to_snake(name):
    """Converts CamelCase to snake_case (e.g. ReduceMean -> reduce_mean)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _make_gradient_backend_validator(gradient_of_nodes):
    """Returns a backend validator that checks gradient_of_nodes on the first node."""

    def _gradient_backend_validator(model, *_inputs):
        """Validates that gradient_of_nodes succeeds for the first node of model."""
        nodes = list(model.graph.node)
        if not nodes:
            return None
        first_node = nodes[0]
        node_inputs = [str(inp) for inp in first_node.input if str(inp)]
        if not node_inputs:
            return None
        node_outputs = [str(out) for out in first_node.output if str(out)]
        if not node_outputs:
            return None
        xs = [node_inputs[0]]
        zs = node_inputs[1:]
        y = node_outputs[0]
        grad = gradient_of_nodes(
            nodes=[first_node], inputs=node_inputs, initializers=[], xs=xs, y=y, zs=zs
        )
        assert (
            len(list(grad.output)) >= 1
        ), f"Empty gradient output for op_type={first_node.op_type}"
        return None

    return _gradient_backend_validator


try:
    _make_test_class = import_or_skip("onnx_light.onnx_lib.backend.test.case", "make_test_class")
    _gradient_mod = import_or_skip("onnx_light.onnx_gradient")
    _grad_op_types = _gradient_mod.GradRegistry.default().op_types()
    _grad_include = [rf"test_{_camel_to_snake(op)}" for op in _grad_op_types]
    TestGradientBackendCases = _make_test_class(
        _make_gradient_backend_validator(_gradient_mod.gradient_of_nodes),
        include_regex=_grad_include,
    )
except (ImportError, unittest.SkipTest):
    pass


if __name__ == "__main__":
    unittest.main()
