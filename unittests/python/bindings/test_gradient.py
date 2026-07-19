# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the onnx_gradient Python bindings.

These tests verify that the Python-exposed gradient functions produce
FunctionProtos with the correct structure for simple ONNX graphs.

The tests do not import onnx directly (ci_no_onnx compatibility).
"""

import unittest

from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_proto._helper import make_node


def _import_gradient():
    """Imports the gradient module; skips if not available (reduced build)."""
    try:
        from onnx_light.onnx_gradient import gradient_of_function, gradient_of_nodes

        return gradient_of_nodes, gradient_of_function
    except ImportError:
        return None, None


class TestGradientBindings(ExtTestCase):
    def setUp(self):
        gon, gof = _import_gradient()
        if gon is None:
            self.skipTest("onnx_gradient bindings not available (reduced build)")
        self.gradient_of_nodes = gon
        self.gradient_of_function = gof

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

    def test_batch_normalization_grad(self):
        """Verifies the BatchNormalization gradient outputs."""
        nodes = [make_node("BatchNormalization", ["X", "scale", "B", "mean", "var"], ["Y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes,
            inputs=["X", "scale", "B", "mean", "var"],
            initializers=[],
            xs=["X", "scale", "B", "mean", "var"],
            y="Y",
            zs=[],
        )
        self.assertEqual(
            list(grad.output), ["grad_X", "grad_scale", "grad_B", "grad_mean", "grad_var"]
        )
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ReduceSum", op_types)
        self.assertIn("Reshape", op_types)

    def test_group_normalization_grad(self):
        """Verifies the GroupNormalization gradient structure."""
        nodes = [make_node("GroupNormalization", ["X", "scale", "bias"], ["Y"], num_groups=2)]
        grad = self.gradient_of_nodes(
            nodes=nodes,
            inputs=["X", "scale", "bias"],
            initializers=[],
            xs=["X", "scale", "bias"],
            y="Y",
            zs=[],
        )
        self.assertEqual(list(grad.output), ["grad_X", "grad_scale", "grad_bias"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ReduceMean", op_types)
        self.assertIn("Reshape", op_types)

    def test_instance_normalization_grad(self):
        """Verifies the InstanceNormalization gradient structure."""
        nodes = [make_node("InstanceNormalization", ["X", "scale", "B"], ["Y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes,
            inputs=["X", "scale", "B"],
            initializers=[],
            xs=["X", "scale", "B"],
            y="Y",
            zs=[],
        )
        self.assertEqual(list(grad.output), ["grad_X", "grad_scale", "grad_B"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ReduceMean", op_types)
        self.assertIn("ReduceSum", op_types)

    def test_layer_normalization_grad(self):
        """Verifies the LayerNormalization gradient structure."""
        nodes = [make_node("LayerNormalization", ["X", "scale", "B"], ["Y"], axis=1)]
        grad = self.gradient_of_nodes(
            nodes=nodes,
            inputs=["X", "scale", "B"],
            initializers=[],
            xs=["X", "scale", "B"],
            y="Y",
            zs=[],
        )
        self.assertEqual(list(grad.output), ["grad_X", "grad_scale", "grad_B"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("Flatten", op_types)
        self.assertIn("ReduceMean", op_types)

    def test_lp_normalization_grad(self):
        """Verifies the LpNormalization gradient structure."""
        nodes = [make_node("LpNormalization", ["X"], ["Y"], axis=-1, p=2)]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X"], initializers=[], xs=["X"], y="Y", zs=[]
        )
        self.assertEqual(list(grad.output), ["grad_X"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ReduceSum", op_types)
        self.assertIn("Sqrt", op_types)

    def test_mean_variance_normalization_grad(self):
        """Verifies the MeanVarianceNormalization gradient structure."""
        nodes = [make_node("MeanVarianceNormalization", ["X"], ["Y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X"], initializers=[], xs=["X"], y="Y", zs=[]
        )
        self.assertEqual(list(grad.output), ["grad_X"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ReduceMean", op_types)
        self.assertIn("Sqrt", op_types)

    def test_rms_normalization_grad(self):
        """Verifies the RMSNormalization gradient structure."""
        nodes = [make_node("RMSNormalization", ["X", "scale"], ["Y"], axis=-1)]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "scale"], initializers=[], xs=["X", "scale"], y="Y", zs=[]
        )
        self.assertEqual(list(grad.output), ["grad_X", "grad_scale"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("Flatten", op_types)
        self.assertIn("ReduceMean", op_types)

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
    # gradient_of_nodes: Conv                                             #
    # ------------------------------------------------------------------ #

    def test_conv_grad_dx(self):
        """Gradient of y = Conv(X, W) w.r.t. X uses ConvTranspose."""
        nodes = [make_node("Conv", ["X", "W"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W"], initializers=[], xs=["X"], y="y", zs=["W"]
        )
        self.assertEqual(list(grad.output), ["grad_X"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ConvTranspose", op_types)

    def test_conv_grad_dw(self):
        """Gradient of y = Conv(X, W) w.r.t. W uses Transpose + Conv."""
        nodes = [make_node("Conv", ["X", "W"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W"], initializers=[], xs=["W"], y="y", zs=["X"]
        )
        self.assertEqual(list(grad.output), ["grad_W"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("Transpose", op_types)
        self.assertIn("Conv", op_types)

    def test_conv_grad_dx_dw(self):
        """Gradient of y = Conv(X, W) w.r.t. both X and W."""
        nodes = [make_node("Conv", ["X", "W"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W"], initializers=[], xs=["X", "W"], y="y", zs=[]
        )
        self.assertEqual(list(grad.output), ["grad_X", "grad_W"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ConvTranspose", op_types)
        self.assertIn("Conv", op_types)

    def test_conv_grad_db(self):
        """Gradient of y = Conv(X, W, B) w.r.t. B uses ReduceSum."""
        nodes = [make_node("Conv", ["X", "W", "B"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W", "B"], initializers=[], xs=["B"], y="y", zs=["X", "W"]
        )
        self.assertEqual(list(grad.output), ["grad_B"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ReduceSum", op_types)

    def test_conv_grad_all(self):
        """Gradient of y = Conv(X, W, B) w.r.t. X, W, and B."""
        nodes = [make_node("Conv", ["X", "W", "B"], ["y"])]
        grad = self.gradient_of_nodes(
            nodes=nodes, inputs=["X", "W", "B"], initializers=[], xs=["X", "W", "B"], y="y", zs=[]
        )
        self.assertEqual(list(grad.output), ["grad_X", "grad_W", "grad_B"])
        op_types = [str(n.op_type) for n in grad.node]
        self.assertIn("ConvTranspose", op_types)
        self.assertIn("Conv", op_types)
        self.assertIn("ReduceSum", op_types)


if __name__ == "__main__":
    unittest.main()
