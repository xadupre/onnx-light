# source: https://github.com/onnx/onnx/blob/main/onnx/test/data_propagation_test.py
import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference

_TEST_OPSET_VERSION = 23


class TestDataPropagation(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        onnxl.defs.register_onnx_operator_set_schema()

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
        schema = onnxl.defs.get_schema(op_type, _TEST_OPSET_VERSION, "")
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
        if not onnxl.defs.has_schema("ConstantOfShape", _TEST_OPSET_VERSION, ""):
            self.skipTest("ConstantOfShape schema is not available in this build.")
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

    def test_unsqueeze_inmemory_int64_axes(self) -> None:
        """Regression test mirroring microsoft/onnxruntime#28778: shape inference
        on ``Shape -> Identity -> Unsqueeze(axes=INT64 initializer)`` must complete
        without overrun and propagate the expected output shape.
        """
        axis_count = 16
        input_tensor = oh.make_tensor_value_info(
            "input", onnxl.TensorProto.FLOAT, [1] * axis_count
        )
        output_tensor = oh.make_tensor_value_info(
            "output", onnxl.TensorProto.INT64, [1] * axis_count + [axis_count]
        )
        shape_node = oh.make_node("Shape", ["input"], ["shape_out"])
        identity_node = oh.make_node("Identity", ["shape_out"], ["identity_out"])
        axes = oh.make_tensor(
            "unsq_axes",
            onnxl.TensorProto.INT64,
            [axis_count],
            list(range(axis_count)),
        )
        unsqueeze_node = oh.make_node(
            "Unsqueeze", ["identity_out", "unsq_axes"], ["output"]
        )
        graph = oh.make_graph(
            [shape_node, identity_node, unsqueeze_node],
            "Unsqueeze_InMemory_INT64_Axes",
            [input_tensor],
            [output_tensor],
            initializer=[axes],
        )
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18)]
        )
        model.ir_version = 8

        shape_inference.infer_shapes(model)

        # The Shape -> Identity chain carries an INT64 vector of length axis_count.
        value_info_by_name = {vi.name: vi for vi in model.graph.value_info}
        for name in ("shape_out", "identity_out"):
            self.assertIn(name, value_info_by_name)
            self._assert_shape(
                value_info_by_name[name].type, [axis_count], onnxl.TensorProto.INT64
            )

        # The Unsqueeze output keeps the declared rank with the expected last dim.
        out = model.graph.output[0]
        self.assertEqual(out.name, "output")
        self._assert_shape(
            out.type, [1] * axis_count + [axis_count], onnxl.TensorProto.INT64
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
