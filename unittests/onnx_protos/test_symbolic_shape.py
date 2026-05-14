# source: https://github.com/onnx/onnx/blob/main/onnx/test/symbolic_shape_test.py
import unittest
import onnx
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference


class TestSymbolicShape(ExtTestCase):
    def _infer_output(
        self, op_type: str, input_types: dict[str, onnxl.TypeProto], output_name: str, **attrs
    ) -> onnxl.TypeProto:
        """Infers and returns one output type for one node."""
        node = oh.make_node(op_type, list(input_types), [output_name], **attrs)
        schema = onnx.defs.get_schema(op_type, 23, "")
        result = shape_inference.infer_node_outputs(schema, node, input_types)
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

    def test_concat_enable_symbolic(self) -> None:
        c_type = self._infer_output(
            "Concat",
            {
                "A": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, "A"]),
                "B": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, 3]),
            },
            "C",
            axis=1,
        )
        output_type = self._infer_output(
            "Cast", {"C": c_type}, "output", to=onnxl.TensorProto.FLOAT
        )

        self._assert_shape(c_type, [2, None], onnxl.TensorProto.FLOAT)
        self._assert_shape(output_type, [2, None], onnxl.TensorProto.FLOAT)
        self.assertEqual(output_type.tensor_type.shape, c_type.tensor_type.shape)

    def test_two_symbolic_concat(self) -> None:
        c_type = self._infer_output(
            "Concat",
            {
                "A": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, "A"]),
                "B": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, 3]),
            },
            "C",
            axis=1,
        )
        e_type = self._infer_output(
            "Concat",
            {"C": c_type, "D": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, "D"])},
            "E",
            axis=1,
        )
        output_type = self._infer_output(
            "Cast", {"E": e_type}, "output", to=onnxl.TensorProto.FLOAT
        )

        self._assert_shape(c_type, [2, None], onnxl.TensorProto.FLOAT)
        self._assert_shape(e_type, [2, None], onnxl.TensorProto.FLOAT)
        self._assert_shape(output_type, [2, None], onnxl.TensorProto.FLOAT)
        self.assertEqual(output_type.tensor_type.shape, e_type.tensor_type.shape)

    def test_unknown_shape(self) -> None:
        c_type = self._infer_output(
            "Concat",
            {
                "A": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [3, None]),
                "B": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [3, None]),
            },
            "C",
            axis=1,
        )
        output_type = self._infer_output(
            "Cast", {"C": c_type}, "output", to=onnxl.TensorProto.FLOAT
        )

        self._assert_shape(c_type, [3, None], onnxl.TensorProto.FLOAT)
        self._assert_shape(output_type, [3, None], onnxl.TensorProto.FLOAT)
        self.assertEqual(output_type.tensor_type.shape, c_type.tensor_type.shape)


if __name__ == "__main__":
    unittest.main()
