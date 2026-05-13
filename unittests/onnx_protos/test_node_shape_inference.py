# source: https://github.com/onnx/onnx/blob/main/onnx/test/node_shape_inference_test.py
import unittest

import onnx.defs

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference


class TestNodeShapeInference(unittest.TestCase):
    def _check_comparison_op(self, op_type: str) -> None:
        """Checks that comparison operators infer boolean output with broadcast shape."""
        node = oh.make_node(op_type, ["x", "y"], ["z"])
        schema = onnx.defs.get_schema(node.op_type, 23, "")
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


if __name__ == "__main__":
    unittest.main()