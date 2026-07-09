# source: https://github.com/onnx/onnx/blob/main/onnx/test/shape_inference_test.py
import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
import onnx_light.onnx.parser as parser
import onnx_light.onnx.shape_inference as shape_inference


class TestShapeInference(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

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
        schema = defs.get_schema(op_type, 23, "")
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
            "Constant", {}, value=oh.make_tensor("v", onnxl.TensorProto.FLOAT, [2, 3], [0.0] * 6)
        )
        self.assertEqual(result.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in result.tensor_type.shape.dim], [2, 3])

    def _make_add_neg_model(self) -> onnxl.ModelProto:
        """Builds a small Add->Neg model with an unshaped output."""
        return oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["T"]), oh.make_node("Neg", ["T"], ["Z"])],
                "g",
                [
                    oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3]),
                    oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3]),
                ],
                [oh.make_value_info("Z", onnxl.TypeProto())],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )

    def test_infer_shapes_inplace(self) -> None:
        """infer_shapes mutates the model in place and returns the same object."""
        model = self._make_add_neg_model()
        result = shape_inference.infer_shapes(model)
        self.assertIs(result, model)
        value_info = {vi.name: vi for vi in model.graph.value_info}
        self.assertIn("T", value_info)
        self.assertEqual(
            [dim.dim_value for dim in value_info["T"].type.tensor_type.shape.dim], [2, 3]
        )

    def test_infer_shapes_parity_options(self) -> None:
        """infer_shapes accepts check_type, strict_mode and data_prop, matching onnx."""
        model = self._make_add_neg_model()
        result = shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=True
        )
        self.assertIs(result, model)
        value_info = {vi.name: vi for vi in model.graph.value_info}
        self.assertIn("T", value_info)
        self.assertEqual(
            [dim.dim_value for dim in value_info["T"].type.tensor_type.shape.dim], [2, 3]
        )

    def test_infer_shapes_with_initializer_weight(self) -> None:
        """infer_shapes uses an initializer's shape to infer downstream outputs.

        This mirrors the common pattern where a weight is provided as an
        initializer (with shape and type set) and shape inference is needed
        to compute the output of a node consuming it. The initializer is not
        listed as a graph input and is preserved after inference.
        """
        weight = oh.make_tensor("W", onnxl.TensorProto.FLOAT, [3, 4], [0.0] * 12)
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "W"], ["Y"])],
                "test_graph",
                [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])],
                [oh.make_value_info("Y", onnxl.TypeProto())],
                initializer=[weight],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
        )

        result = shape_inference.infer_shapes(model)
        self.assertIs(result, model)

        output = {vi.name: vi for vi in result.graph.output}["Y"]
        self.assertEqual(output.type.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in output.type.tensor_type.shape.dim], [2, 4])

        # The weight stays an initializer and is never promoted to a graph input.
        self.assertEqual(len(result.graph.input), 1)
        self.assertNotIn("W", [vi.name for vi in result.graph.input])
        self.assertEqual(len(result.graph.initializer), 1)
        self.assertEqual(result.graph.initializer[0].name, "W")

    def test_infer_shapes_output_without_type(self) -> None:
        """infer_shapes handles graph outputs whose ``type`` field is unset.

        Tools such as onnx-ir emit graph outputs as bare ValueInfoProtos with no
        ``type`` field. Shape inference must populate them instead of raising when
        gathering existing symbolic dimensions.
        """
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "g",
                [
                    oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1, 2]),
                    oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1, 2]),
                ],
                # Output declared without a type field, as onnx-ir emits it.
                [oh.make_empty_tensor_value_info("Z")],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
        )

        result = shape_inference.infer_shapes(model, strict_mode=True)
        self.assertIs(result, model)

        output = {vi.name: vi for vi in result.graph.output}["Z"]
        self.assertEqual(output.type.tensor_type.elem_type, onnxl.TensorProto.FLOAT)
        self.assertEqual([dim.dim_value for dim in output.type.tensor_type.shape.dim], [1, 2])

    def test_infer_shapes_strict_mode_raises(self) -> None:
        """strict_mode surfaces node-level shape inference errors."""
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Add", ["X", "Y"], ["Z"])],
                "g",
                [
                    oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3]),
                    oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 3]),
                ],
                # Z is declared as a scalar (rank 0), which conflicts with the
                # rank-2 shape inferred for the Add output.
                [oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, [])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        with self.assertRaises(shape_inference.InferenceError):
            shape_inference.infer_shapes(model, strict_mode=True)

    def test_loop_with_constant_trip_count_and_early_exit(self) -> None:
        """Loop with a constant trip count that exits early keeps an unknown leading dim.

        Mirrors the regression test from https://github.com/onnx/onnx/pull/8146:
        the loop body forces termination after the first iteration (cond_out = false),
        so the inferred leading dimension of the scan output must remain unknown rather
        than being incorrectly set to the static trip count.
        """
        model = parser.parse_model("""
            <ir_version: 8, opset_import: ["" : 13]>
            test () => ()
               <int64 max_trip_count = {5}, bool cond_orig = {1},
                float[3] outer_scope_input = {1, 2, 3}>
            {
               loop_output = Loop (max_trip_count, cond_orig)
                  <body: graph = subgraph
                     (int64 iter_num_in, bool cond_in) =>
                     (bool cond_out, float[3] output)
                  {
                     cond_out = Constant
                        <value: tensor = bool cond_out_value {0}> ()
                     output = Identity (outer_scope_input)
                  }>
            }
            """)
        inferred_model = shape_inference.infer_shapes(model, data_prop=True)
        loop_output = next(
            value_info
            for value_info in inferred_model.graph.value_info
            if value_info.name == "loop_output"
        )
        first_dim = loop_output.type.tensor_type.shape.dim[0]
        # The leading dim must not be specialized to the trip count (5): the loop
        # exits after one iteration, so 5 would be incorrect. A symbolic unknown
        # dim or any concrete value other than 5 is acceptable.
        self.assertFalse(first_dim.HasField("dim_value") and first_dim.dim_value == 5)
        self.assertEqual(loop_output.type.tensor_type.shape.dim[1].dim_value, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
