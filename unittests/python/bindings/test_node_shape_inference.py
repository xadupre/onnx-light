# source: https://github.com/onnx/onnx/blob/main/onnx/test/node_shape_inference_test.py
import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference


class TestNodeShapeInference(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        onnxl.defs.register_onnx_operator_set_schema()

    def _check_comparison_op(self, op_type: str) -> None:
        """Checks that comparison operators infer boolean output with broadcast shape."""
        node = oh.make_node(op_type, ["x", "y"], ["z"])
        schema = onnxl.defs.get_schema(node.op_type, 23, "")
        xtype = oh.make_tensor_type_proto(onnxl.TensorProto.INT32, [1, 10])
        ytype = oh.make_tensor_type_proto(onnxl.TensorProto.INT32, [10, 1])
        result = shape_inference.infer_node_outputs(schema, node, {"x": xtype, "y": ytype})
        self.assertEqual(list(result.keys()), ["z"])
        self.assertEqual(result["z"].tensor_type.elem_type, onnxl.TensorProto.BOOL)
        self.assertEqual([dim.dim_value for dim in result["z"].tensor_type.shape.dim], [10, 10])

    def test_comparison_op_greater_or_equal(self) -> None:
        self._check_comparison_op("GreaterOrEqual")

    def test_comparison_op_less_or_equal(self) -> None:
        self._check_comparison_op("LessOrEqual")

    def _check_conv_weight_rank_mismatch_raises(self, opset: int) -> None:
        """Conv with weight rank != input rank and no kernel_shape attribute
        must fail shape inference instead of reading past the end of the
        dilations/pads vectors (GHSA-r6v2-894m-m2v4)."""
        schema = onnxl.defs.get_schema("Conv", opset, "")
        node = oh.make_node("Conv", ["x", "w"], ["z"])
        # Weight has more spatial dims than input.
        xtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [1, 4, 8, 8])
        wtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [5, 4, 3, 3, 3])
        with self.assertRaises(shape_inference.InferenceError) as cm:
            shape_inference.infer_node_outputs(schema, node, {"x": xtype, "w": wtype})
        self.assertIn("weight tensor", str(cm.exception))

        # Weight has fewer spatial dims than input.
        xtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [1, 4, 8, 8, 8])
        wtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [5, 4, 3, 3])
        with self.assertRaises(shape_inference.InferenceError) as cm:
            shape_inference.infer_node_outputs(schema, node, {"x": xtype, "w": wtype})
        self.assertIn("weight tensor", str(cm.exception))

    def test_conv_weight_rank_mismatch_raises_opset1_to_11(self) -> None:
        # Conv-1 -> convPoolShapeInference_opset1_to_11
        self._check_conv_weight_rank_mismatch_raises(1)

    def test_conv_weight_rank_mismatch_raises_opset19(self) -> None:
        # Conv-11 -> convPoolShapeInference_opset19
        self._check_conv_weight_rank_mismatch_raises(11)

    def test_conv_weight_rank_mismatch_raises_latest(self) -> None:
        # Conv-22 -> convPoolShapeInference
        self._check_conv_weight_rank_mismatch_raises(onnxl.defs.get_schema("Conv").since_version)


if __name__ == "__main__":
    unittest.main()
