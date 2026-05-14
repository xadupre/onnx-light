# source: https://github.com/onnx/onnx/blob/main/onnx/test/data_propagation_test.py
import unittest

import onnx
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference

_TEST_OPSET_VERSION = 23


class TestDataPropagation(ExtTestCase):
    def _infer_output(
        self,
        op_type: str,
        input_types: dict[str, onnxl.TypeProto],
        output_name: str,
        *,
        input_data: dict[str, onnxl.TensorProto] | None = None,
        **attrs,
    ) -> onnxl.TypeProto:
        """Infers and returns one output type for one node."""
        node = oh.make_node(op_type, list(input_types), [output_name], **attrs)
        schema = onnx.defs.get_schema(op_type, _TEST_OPSET_VERSION, "")
        result = shape_inference.infer_node_outputs(
            schema, node, input_types, input_data=input_data or {}
        )
        self.assertEqual(list(result), [output_name])
        return result[output_name]

    def _assert_shape(
        self, value: onnxl.TypeProto, expected_shape: list[int | None], expected_dtype: int
    ) -> None:
        """Checks output element type and tensor shape."""
        self.assertEqual(value.tensor_type.elem_type, expected_dtype)
        self.assertEqual(
            [
                dim.dim_value if dim.has_dim_value() else None
                for dim in value.tensor_type.shape.dim
            ],
            expected_shape,
        )

    def test_expand_symbolic_input(self) -> None:
        """Checks that Expand uses a propagated shape tensor value."""
        z_type = self._infer_output(
            "Expand",
            {
                "x": oh.make_tensor_type_proto(onnxl.TensorProto.INT32, [3, 1, 2]),
                "shape": oh.make_tensor_type_proto(onnxl.TensorProto.INT64, [3]),
            },
            "z",
            input_data={
                "shape": oh.make_tensor("shape", onnxl.TensorProto.INT64, [3], [3, 4, 2])
            },
        )

        self._assert_shape(z_type, [3, 4, 2], onnxl.TensorProto.INT32)

    def test_constantofshape_with_symbolic_shape(self) -> None:
        """Checks that ConstantOfShape uses a propagated shape tensor value."""
        y_type = self._infer_output(
            "ConstantOfShape",
            {"shape": oh.make_tensor_type_proto(onnxl.TensorProto.INT64, [3])},
            "y",
            input_data={
                "shape": oh.make_tensor("shape", onnxl.TensorProto.INT64, [3], [3, 4, 5])
            },
            value=oh.make_tensor("value", onnxl.TensorProto.INT32, [1], [2]),
        )

        self._assert_shape(y_type, [3, 4, 5], onnxl.TensorProto.INT32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
