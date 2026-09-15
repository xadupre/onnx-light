# source: https://github.com/onnx/onnx/blob/main/onnx/test/node_shape_inference_test.py
import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.defs as defs
import onnx_light.onnx.shape_inference as shape_inference


class TestNodeShapeInference(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    def _check_comparison_op(self, op_type: str) -> None:
        """Checks that comparison operators infer boolean output with broadcast shape."""
        node = oh.make_node(op_type, ["x", "y"], ["z"])
        schema = defs.get_schema(node.op_type, 23, "")
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
        schema = defs.get_schema("Conv", opset, "")
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
        self._check_conv_weight_rank_mismatch_raises(defs.get_schema("Conv").since_version)

    def _check_conv_transpose_group_divisibility_raises(self, opset: int) -> None:
        """Checks that ConvTranspose with input channels C not divisible by
        group fails shape inference (propagated from onnx/onnx#7821)."""
        schema = defs.get_schema("ConvTranspose", opset, "")
        node = oh.make_node("ConvTranspose", ["x", "w"], ["z"], group=3)
        xtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [1, 32, 14, 14])
        wtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [32, 64, 3, 3])
        with self.assertRaises(shape_inference.InferenceError) as cm:
            shape_inference.infer_node_outputs(schema, node, {"x": xtype, "w": wtype})
        self.assertIn("Input channels C must be divisible by group", str(cm.exception))

    def test_conv_transpose_group_divisibility_raises_opset1(self) -> None:
        # ConvTranspose-1 -> convTransposeShapeInference_opset1
        self._check_conv_transpose_group_divisibility_raises(1)

    def test_conv_transpose_group_divisibility_raises_opset11(self) -> None:
        # ConvTranspose-11 -> convTransposeShapeInference_opset11
        self._check_conv_transpose_group_divisibility_raises(11)

    def test_conv_transpose_group_divisibility_raises_latest(self) -> None:
        # ConvTranspose latest -> convTransposeShapeInference
        self._check_conv_transpose_group_divisibility_raises(
            defs.get_schema("ConvTranspose").since_version
        )

    def _check_conv_transpose_non_positive_group_raises(self, opset: int) -> None:
        """Checks that ConvTranspose with a non-positive group attribute fails
        shape inference (propagated from onnx/onnx#7821)."""
        schema = defs.get_schema("ConvTranspose", opset, "")
        node = oh.make_node("ConvTranspose", ["x", "w"], ["z"], group=0)
        xtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [1, 32, 14, 14])
        wtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [32, 64, 3, 3])
        with self.assertRaises(shape_inference.InferenceError) as cm:
            shape_inference.infer_node_outputs(schema, node, {"x": xtype, "w": wtype})
        self.assertIn("Attribute group must be > 0", str(cm.exception))

    def test_conv_transpose_non_positive_group_raises_opset1(self) -> None:
        self._check_conv_transpose_non_positive_group_raises(1)

    def test_conv_transpose_non_positive_group_raises_opset11(self) -> None:
        self._check_conv_transpose_non_positive_group_raises(11)

    def test_conv_transpose_non_positive_group_raises_latest(self) -> None:
        self._check_conv_transpose_non_positive_group_raises(
            defs.get_schema("ConvTranspose").since_version
        )

    def _make_scan_body_identity(self):
        """Creates a minimal Scan body graph: ``y_elt = Identity(x_elt)``."""
        body_graph = oh.make_graph(
            [oh.make_node("Identity", ["x_elt"], ["y_elt"])],
            "scan_body",
            [oh.make_tensor_value_info("x_elt", onnxl.TensorProto.FLOAT, None)],
            [oh.make_tensor_value_info("y_elt", onnxl.TensorProto.FLOAT, None)],
        )
        return body_graph

    def _check_scan_num_scan_inputs_out_of_range(self, opset: int) -> None:
        """Checks that Scan with num_scan_inputs > num_inputs raises InferenceError.

        GHSA-qrhj-v62m-vmpf: num_scan_inputs > num_inputs caused a size_t
        underflow in ScanInferenceFunction leading to out-of-bounds indexing
        and potential memory corruption.  The fix validates the bounds before
        the subtraction.
        """
        schema = defs.get_schema("Scan", opset, "")
        body = self._make_scan_body_identity()
        # 1 input but num_scan_inputs=9 — previously caused size_t underflow.
        node = oh.make_node("Scan", ["x"], ["y"], body=body, num_scan_inputs=9)
        xtype = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [4, 3])
        with self.assertRaises(shape_inference.InferenceError) as cm:
            shape_inference.infer_node_outputs(schema, node, {"x": xtype})
        self.assertIn("num_scan_inputs", str(cm.exception))

    def test_scan_num_scan_inputs_out_of_range_opset9(self) -> None:
        self._check_scan_num_scan_inputs_out_of_range(9)

    def test_scan_num_scan_inputs_out_of_range_opset11(self) -> None:
        self._check_scan_num_scan_inputs_out_of_range(11)

    def test_scan_num_scan_inputs_out_of_range_latest(self) -> None:
        self._check_scan_num_scan_inputs_out_of_range(defs.get_schema("Scan").since_version)

    def test_if_subgraph_with_unknown_op_does_not_use_temporary_map(self) -> None:
        branch_output = oh.make_tensor_value_info("branch_result", onnxl.TensorProto.FLOAT, ())
        branches = [
            oh.make_graph(
                [oh.make_node("UnknownOp", [], ["branch_result"], domain="local.unknown")],
                name,
                [],
                [branch_output],
            )
            for name in ("then_branch", "else_branch")
        ]
        schema = defs.get_schema("If")
        result = shape_inference.infer_node_outputs(
            schema,
            oh.make_node(
                "If", ["cond"], ["result"], then_branch=branches[0], else_branch=branches[1]
            ),
            {"cond": oh.make_tensor_type_proto(onnxl.TensorProto.BOOL, ())},
            opset_imports=[
                oh.make_opsetid("", schema.since_version),
                oh.make_opsetid("local.unknown", 1),
            ],
        )

        self.assertEqual(
            result, {"result": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, ())}
        )


if __name__ == "__main__":
    unittest.main()
