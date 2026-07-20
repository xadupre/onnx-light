"""Tests for the :mod:`onnx_light.onnx_core` Python facade."""

import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx_core as onnx_core
import onnx_light.onnx.helper as oh


class TestOnnxCoreModule(ExtTestCase):
    def test_tensor_type_exposed(self):
        self.assertTrue(hasattr(onnx_core, "TensorType"))

    def test_tensor_type_values(self):
        self.assertEqual(onnx_core.ToTypeString(onnx_core.TensorType.kFloat), "tensor(float)")
        self.assertEqual(onnx_core.ToTypeString(onnx_core.TensorType.kInt64), "tensor(int64)")

    def test_tensor_type_same_as_onnx_op(self):
        import onnx_light.onnx_op as onnx_op

        self.assertIs(onnx_core.TensorType, onnx_op.TensorType)

    def test_to_type_string_exposed(self):
        self.assertTrue(callable(onnx_core.ToTypeString))
        self.assertEqual(onnx_core.ToTypeString(onnx_core.TensorType.kBool), "tensor(bool)")

    def test_collect_external_inputs(self):
        nodes = [oh.make_node("Mul", ["x", "y"], ["t"]), oh.make_node("Add", ["t", "z"], ["out"])]
        self.assertEqual(onnx_core.collect_external_inputs(nodes), ["x", "y", "z"])

    def test_collect_node_inputs(self):
        node = oh.make_node("Add", ["x", "y"], ["out"])
        self.assertEqual(onnx_core.collect_node_inputs(node), ["x", "y"])

    def test_collect_node_inputs_skips_empty(self):
        node = oh.make_node("Add", ["x", ""], ["out"])
        self.assertEqual(onnx_core.collect_node_inputs(node), ["x"])

    def test_collect_remaining_inputs(self):
        nodes = [oh.make_node("Mul", ["x", "y"], ["t"]), oh.make_node("Add", ["t", "z"], ["out"])]
        self.assertEqual(
            onnx_core.collect_remaining_inputs(nodes, ["out"]), [["x", "y", "z"], ["t", "z"]]
        )

    def test_all_exports_present(self):
        for name in onnx_core.__all__:
            self.assertTrue(
                hasattr(onnx_core, name), f"onnx_light.onnx_core missing export: {name}"
            )


if __name__ == "__main__":
    unittest.main()
