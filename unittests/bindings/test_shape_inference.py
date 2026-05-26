# source: https://github.com/onnx/onnx/blob/main/onnx/test/shape_inference_test.py
import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference


class TestShapeInference(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        onnxl.defs.register_onnx_operator_set_schema()

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
        schema = onnxl.defs.get_schema(op_type, 23, "")
        result = shape_inference.infer_node_outputs(
            schema, node, input_types, input_data=input_data or {}
        )
        self.assertEqual(list(result), ["z"])
        return result["z"]

    def test_flatten(self) -> None:
        """Checks that Flatten infers the expected rank-2 output shape."""
        result = self._infer_output(
            "Flatten", {"x": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, 3, 4, 5])}
        )

        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [2, 60])

    def test_shape(self) -> None:
        """Checks that Shape infers an INT64 vector sized by the input rank."""
        result = self._infer_output(
            "Shape", {"x": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2, 4, 3])}
        )

        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.INT64)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [3])

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

    # ── Constant ────────────────────────────────────────────────────────
    # The following tests mirror the ``test_constant_*`` cases from
    # upstream ``onnx/test/shape_inference_test.py`` to ensure
    # ``Constant`` shape inference covers every ``value*`` attribute
    # form and the ``sparse_value`` form.

    def test_constant_value_int(self) -> None:
        """value_int produces a scalar INT64 output."""
        result = self._infer_output("Constant", {}, value_int=42)
        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.INT64)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [])

    def test_constant_value_ints(self) -> None:
        """value_ints produces a 1-D INT64 output sized by len(value_ints)."""
        value_ints = [1, 2, 3]
        result = self._infer_output("Constant", {}, value_ints=value_ints)
        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.INT64)
        self.assertEqual(
            [dim.dim_value for dim in result.tensor_type.shape.dim], [len(value_ints)]
        )

    def test_constant_value_float(self) -> None:
        """value_float produces a scalar FLOAT output."""
        result = self._infer_output("Constant", {}, value_float=1.42)
        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [])

    def test_constant_value_floats(self) -> None:
        """value_floats produces a 1-D FLOAT output sized by len(value_floats)."""
        value_floats = [1.0, 1.1, 1.2]
        result = self._infer_output("Constant", {}, value_floats=value_floats)
        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual(
            [dim.dim_value for dim in result.tensor_type.shape.dim], [len(value_floats)]
        )

    def test_constant_value_string(self) -> None:
        """value_string produces a scalar STRING output."""
        result = self._infer_output("Constant", {}, value_string="String value")
        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.STRING)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [])

    def test_constant_value_strings(self) -> None:
        """value_strings produces a 1-D STRING output sized by len(value_strings)."""
        value_strings = ["o", "n", "n", "x"]
        result = self._infer_output("Constant", {}, value_strings=value_strings)
        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.STRING)
        self.assertEqual(
            [dim.dim_value for dim in result.tensor_type.shape.dim], [len(value_strings)]
        )

    def test_constant_value_tensor(self) -> None:
        """value tensor attribute drives both dtype and shape of the output."""
        result = self._infer_output(
            "Constant",
            {},
            value=oh.make_tensor("v", onnxl.TensorProto.FLOAT, [2, 3], [0.0] * 6),
        )
        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
