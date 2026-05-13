# source: https://github.com/onnx/onnx/blob/main/onnx/test/shape_inference_test.py
import unittest
import onnx_light.onnx.defs
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference


class TestShapeInference(unittest.TestCase):
    def _infer_output(
        self,
        op_type: str,
        input_types: dict[str, onnxl.TypeProto],
        *,
        input_data: dict[str, onnxl.TensorProto] | None = None,
        **attrs,
    ) -> onnxl.TypeProto:
        """Infers the single output type for a node."""
        node = oh.make_node(op_type, list(input_types), ["z"], **attrs)
        schema = onnx_light.onnx.defs.get_schema(op_type, 23, "")
        result = shape_inference.infer_node_outputs(
            schema, node, input_types, input_data=input_data or {}
        )
        self.assertEqual(list(result), ["z"])
        return result["z"]

    def test_flatten(self) -> None:
        """Checks that Flatten infers the expected rank-2 output shape."""
        result = self._infer_output(
            "Flatten",
            {"x": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, 3, 4, 5])},
            axis=2,
        )

        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [6, 20])

    def test_shape(self) -> None:
        """Checks that Shape infers an INT64 vector sized by the input rank."""
        result = self._infer_output(
            "Shape", {"x": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, 4, 3])}
        )

        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.INT64)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [3])

    def test_space_to_depth(self) -> None:
        """Checks that SpaceToDepth infers the transformed tensor shape."""
        result = self._infer_output(
            "SpaceToDepth",
            {"x": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, 3, 100, 100])},
            blocksize=10,
        )

        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual(
            [dim.dim_value for dim in result.tensor_type.shape.dim], [2, 300, 10, 10]
        )

    def test_logical_not(self) -> None:
        """Checks that Not preserves the input boolean tensor shape."""
        result = self._infer_output(
            "Not", {"x": oh.make_tensor_type_proto(onnxl.TensorProto.BOOL, [30, 4, 5])}
        )

        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.BOOL)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [30, 4, 5])

    def test_expand_with_shape_data(self) -> None:
        """Checks that Expand uses the shape tensor data when provided."""
        result = self._infer_output(
            "Expand",
            {
                "x": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [1, 3, 1]),
                "shape": oh.make_tensor_type_proto(onnxl.TensorProto.INT64, [3]),
            },
            input_data={
                "shape": oh.make_tensor("shape", onnxl.TensorProto.INT64, [3], [2, 3, 4])
            },
        )

        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [2, 3, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
