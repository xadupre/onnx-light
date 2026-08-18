# source: https://github.com/onnx/onnx/blob/main/onnx/test/shape_inference_test.py
import unittest
from typing import Any
from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.checker as checker
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

    def test_pad_with_constant_value_ints(self) -> None:
        """Checks that Pad infers its padded output shape from a Constant using value_ints."""
        graph = oh.make_graph(
            [
                oh.make_node("Constant", [], ["pads"], value_ints=[0, 1, 0, 1]),
                oh.make_node("Pad", ["x", "pads"], ["y"]),
            ],
            "test_pad_with_constant_value_ints",
            [oh.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, (1, 2))],
            [],
        )
        self._assert_inferred(
            graph,
            [
                oh.make_tensor_value_info("pads", onnxl.TensorProto.INT64, (4,)),
                oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, (1, 4)),
            ],
        )

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

    # ── Helpers for higher-order op (subgraph) shape inference tests ─────────

    def _compare_value_infos(
        self, vi_type: onnxl.TypeProto, inferred_vi_type: onnxl.TypeProto
    ) -> None:
        """Compares two TypeProto objects for compatible type and shape."""
        if vi_type.has_tensor_type():
            self.assertTrue(
                inferred_vi_type.has_tensor_type(),
                f"Expected tensor type, got {inferred_vi_type!r}",
            )
            self.assertEqual(
                vi_type.tensor_type.elem_type,
                inferred_vi_type.tensor_type.elem_type,
                f"\n{vi_type}\n{inferred_vi_type}\n",
            )
            self.assertEqual(
                vi_type.tensor_type.has_shape(),
                inferred_vi_type.tensor_type.has_shape(),
                f"\n{vi_type}\n{inferred_vi_type}\n",
            )
            if vi_type.tensor_type.has_shape():
                self.assertEqual(
                    len(vi_type.tensor_type.shape.dim),
                    len(inferred_vi_type.tensor_type.shape.dim),
                    f"\n{vi_type}\n{inferred_vi_type}\n",
                )
                for dim_i, dim in enumerate(vi_type.tensor_type.shape.dim):
                    inferred_dim = inferred_vi_type.tensor_type.shape.dim[dim_i]
                    if dim.dim_param:
                        self.assertEqual(
                            dim.dim_param,
                            inferred_dim.dim_param,
                            f"\n{vi_type}\n{inferred_vi_type}\n",
                        )
                    else:
                        self.assertEqual(
                            dim.dim_value,
                            inferred_dim.dim_value,
                            f"\n{vi_type}\n{inferred_vi_type}\n",
                        )
        elif vi_type.has_sequence_type():
            self.assertTrue(inferred_vi_type.has_sequence_type())
            self._compare_value_infos(
                vi_type.sequence_type.elem_type, inferred_vi_type.sequence_type.elem_type
            )
        elif vi_type.has_optional_type():
            self.assertTrue(inferred_vi_type.has_optional_type())
            self._compare_value_infos(
                vi_type.optional_type.elem_type, inferred_vi_type.optional_type.elem_type
            )
        elif vi_type.has_map_type():
            self.assertTrue(inferred_vi_type.has_map_type())
            self.assertEqual(vi_type.map_type.key_type, inferred_vi_type.map_type.key_type)
            self._compare_value_infos(
                vi_type.map_type.value_type, inferred_vi_type.map_type.value_type
            )
        elif vi_type == onnxl.TypeProto():
            self.assertEqual(inferred_vi_type, onnxl.TypeProto())
        else:
            raise NotImplementedError(
                f"Unrecognized value info type in _compare_value_infos: {vi_type!r}"
            )

    def _inferred(
        self, graph_or_model: onnxl.GraphProto | onnxl.ModelProto, **kwargs: Any
    ) -> onnxl.ModelProto:
        """Wraps a GraphProto in a ModelProto, runs shape inference, and returns the model."""
        data_prop = kwargs.pop("data_prop", False)
        opset_imports = kwargs.pop("opset_imports", None)
        if isinstance(graph_or_model, onnxl.GraphProto):
            model = oh.make_model(
                graph_or_model, producer_name="onnx-test", opset_imports=opset_imports
            )
        else:
            model = graph_or_model
        shape_inference.infer_shapes(model, data_prop=data_prop)
        return model

    def _assert_inferred(
        self,
        graph_or_model: onnxl.GraphProto | onnxl.ModelProto,
        inferred_value_infos: list[onnxl.ValueInfoProto],
        **kwargs: Any,
    ) -> None:
        """Runs shape inference and verifies the expected types/shapes are produced.

        ``inferred_value_infos`` specifies the expected delta produced by type/shape
        inference.  Names not in ``inferred_value_infos`` retain their original
        type/shape from the input model.  Inferred type info may appear in either
        ``graph.value_info`` (intermediate values) or ``graph.output`` (typed graph
        outputs), so both are checked.
        """
        graph = (
            graph_or_model
            if isinstance(graph_or_model, onnxl.GraphProto)
            else graph_or_model.graph
        )
        names_in_inferred_value_infos = {x.name for x in inferred_value_infos}
        # Build the expected mapping from names not provided in inferred_value_infos,
        # sourced from the input graph's value_info and output lists.
        expected: dict[str, onnxl.ValueInfoProto] = {}
        for x in [*graph.value_info, *graph.output]:
            if x.name in names_in_inferred_value_infos:
                continue
            if x.name in expected:
                self._compare_value_infos(expected[x.name].type, x.type)
            else:
                expected[x.name] = x
        expected.update({x.name: x for x in inferred_value_infos})
        inferred_model = self._inferred(graph_or_model, **kwargs)
        inferred_graph = inferred_model.graph
        # Inferred type info may be recorded either in value_info (intermediate
        # values, and outputs that were untyped in the input model) or directly on
        # the graph outputs (outputs that were already typed).  Merge both by name.
        # An untyped graph output may appear in both value_info (with the inferred
        # type) and in graph.output (still with empty type); prefer the typed entry.
        empty_type = onnxl.TypeProto()
        inferred: dict[str, onnxl.ValueInfoProto] = {}
        for x in [*inferred_graph.value_info, *inferred_graph.output]:
            if x.name in inferred:
                existing = inferred[x.name]
                if x.type == empty_type:
                    pass  # duplicate is untyped; keeps the already-recorded typed entry
                elif existing.type == empty_type:
                    inferred[x.name] = x  # replaces untyped with the typed duplicate
                else:
                    self._compare_value_infos(existing.type, x.type)
            else:
                inferred[x.name] = x
        self.assertEqual(
            expected.keys(),
            inferred.keys(),
            f"\nExpected value infos for: {sorted(expected)}"
            f"\nInferred value infos for: {sorted(inferred)}\n",
        )
        for name, expected_vi in expected.items():
            self._compare_value_infos(expected_vi.type, inferred[name].type)

    # ── Scan ─────────────────────────────────────────────────────────────────

    def test_scan(self) -> None:
        """Scan (opset 8): shape inference propagates types through the body subgraph."""
        # The Scan outputs are left untyped so shape inference must compute their type/shape.
        # The leading "" input is the (unused) optional sequence_lens input of opset-8 Scan.
        # Body placeholders were declared as UNDEFINED in the original make_graph version;
        # the parser leaves them untyped, which shape inference fills identically.
        graph = parser.parse_graph("""
            agraph (float[1, 3] loop_state_orig, float[1, sequence, 2] scan_input)
                => (loop_state_final, scan_output)
            {
                loop_state_final, scan_output = Scan ("", loop_state_orig, scan_input) <
                    num_scan_inputs = 1,
                    body = subgraph (loop_state_in, input) => (loop_state_out, output) {
                        loop_state_out = Identity(loop_state_in)
                        output = Identity(input)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph,
            [
                oh.make_tensor_value_info("loop_state_final", onnxl.TensorProto.FLOAT, (1, 3)),
                oh.make_tensor_value_info(
                    "scan_output", onnxl.TensorProto.FLOAT, (1, "sequence", 2)
                ),
            ],
            opset_imports=[oh.make_opsetid(defs.ONNX_DOMAIN, 8)],
        )

    def test_scan_opset9(self) -> None:
        """Scan (opset 9): shape inference propagates types through the body subgraph."""
        # Body placeholders were declared as UNDEFINED in the original make_graph version;
        # the parser leaves them untyped, which shape inference fills identically.
        graph = parser.parse_graph("""
            agraph (float[3] loop_state_orig, float[sequence, 2] scan_input)
                => (loop_state_final, scan_output)
            {
                loop_state_final, scan_output = Scan (loop_state_orig, scan_input) <
                    num_scan_inputs = 1,
                    body = subgraph (loop_state_in, input) => (loop_state_out, output) {
                        loop_state_out = Identity(loop_state_in)
                        output = Identity(input)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph,
            [
                oh.make_tensor_value_info("loop_state_final", onnxl.TensorProto.FLOAT, (3,)),
                oh.make_tensor_value_info(
                    "scan_output", onnxl.TensorProto.FLOAT, ("sequence", 2)
                ),
            ],
            opset_imports=[oh.make_opsetid(defs.ONNX_DOMAIN, 9)],
        )

    def test_scan_opset9_axes(self) -> None:
        """Scan (opset 9): scan_input_axes remaps the scanned dimension."""
        # Body placeholders were declared as UNDEFINED in the original make_graph version;
        # the parser leaves them untyped, which shape inference fills identically.
        graph = parser.parse_graph("""
            agraph (float[3] loop_state_orig, float[axis0, sequence, 2] scan_input)
                => (loop_state_final, scan_output)
            {
                loop_state_final, scan_output = Scan (loop_state_orig, scan_input) <
                    num_scan_inputs = 1, scan_input_axes = [1],
                    body = subgraph (loop_state_in, input) => (loop_state_out, output) {
                        loop_state_out = Identity(loop_state_in)
                        output = Identity(input)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph,
            [
                oh.make_tensor_value_info("loop_state_final", onnxl.TensorProto.FLOAT, (3,)),
                oh.make_tensor_value_info(
                    "scan_output", onnxl.TensorProto.FLOAT, ("sequence", "axis0", 2)
                ),
            ],
            opset_imports=[oh.make_opsetid(defs.ONNX_DOMAIN, 9)],
        )

    def test_scan_opset9_negative_axes(self) -> None:
        """Scan (opset 9): negative scan axes are resolved correctly."""
        # Body placeholders were declared as UNDEFINED in the original make_graph version;
        # the parser leaves them untyped, which shape inference fills identically.
        graph = parser.parse_graph("""
            agraph (float[3] loop_state_orig, float[axis0, sequence, 2] scan_input)
                => (loop_state_final, scan_output)
            {
                loop_state_final, scan_output = Scan (loop_state_orig, scan_input) <
                    num_scan_inputs = 1, scan_input_axes = [-2], scan_output_axes = [-2],
                    body = subgraph (loop_state_in, input) => (loop_state_out, output) {
                        loop_state_out = Identity(loop_state_in)
                        output = Identity(input)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph,
            [
                oh.make_tensor_value_info("loop_state_final", onnxl.TensorProto.FLOAT, (3,)),
                oh.make_tensor_value_info(
                    "scan_output", onnxl.TensorProto.FLOAT, ("axis0", "sequence", 2)
                ),
            ],
            opset_imports=[oh.make_opsetid(defs.ONNX_DOMAIN, 9)],
        )

    # ── If ───────────────────────────────────────────────────────────────────

    def test_if_ver1(self) -> None:
        """If (opset 1): shape inference merges branch output types."""
        # Branch outputs were declared as UNDEFINED in the original make_graph version;
        # the parser leaves them untyped, which shape inference fills identically.
        graph = parser.parse_graph("""
            agraph (
                bool[1] cond, float[1] current_value,
                float[1] add_value, float[1] sub_value
            ) => (if_output)
            {
                if_output = If (cond) <
                    then_branch = then_subgraph () => (then_output) {
                        then_output = Add(current_value, add_value)
                    },
                    else_branch = else_subgraph () => (else_output) {
                        else_output = Sub(current_value, sub_value)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph,
            [oh.make_tensor_value_info("if_output", onnxl.TensorProto.FLOAT, (1,))],
            opset_imports=[oh.make_opsetid(defs.ONNX_DOMAIN, 1)],
        )

    def test_if(self) -> None:
        """If: shape inference merges branch output types (current opset)."""
        # Branch outputs were declared as UNDEFINED in the original make_graph version;
        # the parser leaves them untyped, which shape inference fills identically.
        graph = parser.parse_graph("""
            agraph (
                bool[1] cond, float[1] current_value,
                float[1] add_value, float[1] sub_value
            ) => (if_output)
            {
                if_output = If (cond) <
                    then_branch = then_subgraph () => (then_output) {
                        then_output = Add(current_value, add_value)
                    },
                    else_branch = else_subgraph () => (else_output) {
                        else_output = Sub(current_value, sub_value)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph, [oh.make_tensor_value_info("if_output", onnxl.TensorProto.FLOAT, (1,))]
        )

    def test_if_with_different_shapes_in_then_else_branches(self) -> None:
        """If: when branches produce different shapes the merged output has unknown dim."""
        # Branch outputs declared as UNDEFINED with shapes (1,)/(5,) in original make_graph;
        # the parser leaves them untyped -- shape inference computes merged shape (None,)
        # either way.
        graph = parser.parse_graph("""
            agraph (
                bool[1] cond, float[1] current_value,
                float[1] add_value, float[5] sub_value
            ) => (if_output)
            {
                if_output = If (cond) <
                    then_branch = then_subgraph () => (then_output) {
                        then_output = Add(current_value, add_value)
                    },
                    else_branch = else_subgraph () => (else_output) {
                        else_output = Sub(current_value, sub_value)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph, [oh.make_tensor_value_info("if_output", onnxl.TensorProto.FLOAT, (None,))]
        )

    def test_if_no_shape_in_then_branch(self) -> None:
        """If: when one branch produces no shape the merged output has unknown rank."""
        # Branches reference X/axes from the enclosing scope. if_output's inferred type
        # has unknown rank (no shape field), so it is kept as an intermediate value
        # (checked via value_info) rather than a graph output.
        graph = parser.parse_graph("""
            agraph (bool[1] cond, float[4,8,16] X, int64[1] axes) => ()
            {
                if_output = If (cond) <
                    then_branch = then_graph () => (then_output) {
                        then_output = ReduceSum <keepdims=0> (X, axes)
                    },
                    else_branch = else_graph () => (else_output) {
                        else_output = ReduceSum <keepdims=0> (X)
                    }
                >
            }
        """)
        self._assert_inferred(
            graph, [oh.make_tensor_value_info("if_output", onnxl.TensorProto.FLOAT, None)]
        )

    def test_if_no_shape_in_else_branch(self) -> None:
        """If: when the else branch produces no shape the merged output has unknown rank."""
        # Branches reference X/axes from the enclosing scope.
        graph = parser.parse_graph("""
            agraph (bool[1] cond, float[4,8,16] X, int64[1] axes) => ()
            {
                if_output = If (cond) <
                    then_branch = then_graph () => (then_output) {
                        then_output = ReduceSum <keepdims=0> (X)
                    },
                    else_branch = else_graph () => (else_output) {
                        else_output = ReduceSum <keepdims=0> (X, axes)
                    }
                >
            }
        """)
        self._assert_inferred(
            graph, [oh.make_tensor_value_info("if_output", onnxl.TensorProto.FLOAT, None)]
        )

    # ── Loop ─────────────────────────────────────────────────────────────────

    def test_loop(self) -> None:
        """Loop: shape inference populates loop state and accumulated outputs."""
        # cond_in and loop_state_in are left untyped (types supplied by Loop); the body
        # references outer_scope_input from the enclosing graph. loop_state_final has
        # unknown rank (no shape field) since its rank may change between iterations;
        # it is kept as an intermediate value (checked via value_info) rather than a
        # graph output. loop_output is an untyped graph output whose type/shape is
        # computed by shape inference.
        graph = parser.parse_graph("""
            agraph (
                int64[1] max_trip_count, float[1] cond_orig,
                float[2] loop_state_orig, float[3] outer_scope_input
            ) => (loop_output)
            {
                loop_state_final, loop_output = Loop (
                    max_trip_count, cond_orig, loop_state_orig
                ) <
                    body = subgraph (int64[1] iter_num_in, cond_in, loop_state_in)
                        => (cond_out, loop_state_out, float[3] output) {
                        cond_out = Identity(cond_in)
                        loop_state_out = Identity(loop_state_in)
                        output = Identity(outer_scope_input)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph,
            [
                oh.make_tensor_value_info("loop_state_final", onnxl.TensorProto.FLOAT, None),
                oh.make_tensor_value_info("loop_output", onnxl.TensorProto.FLOAT, (None, 3)),
            ],
        )

    def test_loop_no_state(self) -> None:
        """Loop with no loop-carried state: shape inference populates the accumulated output."""
        # cond_in is left untyped (type supplied by Loop); the body references
        # outer_scope_input from the enclosing graph. The Loop output is untyped so
        # shape inference must compute its type/shape.
        graph = parser.parse_graph("""
            agraph (int64[1] max_trip_count, float[1] cond_orig, float[3] outer_scope_input)
                => (loop_output)
            {
                loop_output = Loop (max_trip_count, cond_orig) <
                    body = subgraph (int64[1] iter_num_in, cond_in)
                        => (cond_out, float[3] output) {
                        cond_out = Identity(cond_in)
                        output = Identity(outer_scope_input)
                    }
                >
            }
        """)

        self._assert_inferred(
            graph, [oh.make_tensor_value_info("loop_output", onnxl.TensorProto.FLOAT, (None, 3))]
        )

    # ── SequenceMap ───────────────────────────────────────────────────────────

    def test_sequence_map_identity_known_dims(self) -> None:
        """SequenceMap: shape inference propagates element shape when dims are known."""
        graph = parser.parse_graph("""
            agraph (float[220,220,3] input1, float[220,220,3] input2, float[220,220,3] input3)
                => (out_sequence)
            {
                in_sequence = SequenceConstruct(input1, input2, input3)
                out_sequence = SequenceMap (in_sequence) <
                    body = body_graph (float[220,220,3] input) => (float[220,220,3] output) {
                        output = Identity(input)
                    }
                >
            }
        """)
        self._assert_inferred(
            graph,
            [
                oh.make_tensor_sequence_value_info(
                    "in_sequence", onnxl.TensorProto.FLOAT, (220, 220, 3)
                ),
                oh.make_tensor_sequence_value_info(
                    "out_sequence", onnxl.TensorProto.FLOAT, (220, 220, 3)
                ),
            ],
        )

    def test_sequence_map_identity_unknown_dims(self) -> None:
        """SequenceMap: shape inference uses symbolic dims when element shapes differ."""
        graph = parser.parse_graph("""
            agraph (float[200,300,3] input1, float[100,200,3] input2, float[5,1,3] input3)
                => (out_sequence)
            {
                in_sequence = SequenceConstruct(input1, input2, input3)
                out_sequence = SequenceMap (in_sequence) <
                    body = body_graph (float[H,W,3] input) => (float[H,W,3] output) {
                        output = Identity(input)
                    }
                >
            }
        """)
        self._assert_inferred(
            graph,
            [
                oh.make_tensor_sequence_value_info(
                    "in_sequence", onnxl.TensorProto.FLOAT, (None, None, 3)
                ),
                oh.make_tensor_sequence_value_info(
                    "out_sequence", onnxl.TensorProto.FLOAT, (None, None, 3)
                ),
            ],
        )

    def test_sequence_map_different_tensor_type(self) -> None:
        """SequenceMap: output element dtype may differ from input (Shape -> INT64)."""
        graph = parser.parse_graph("""
            agraph (float[220,310,3] input1, float[110,210,3] input2, float[90,110,3] input3)
                => (shapes)
            {
                in_sequence = SequenceConstruct(input1, input2, input3)
                shapes = SequenceMap (in_sequence) <
                    body = body_graph (float[H,W,C] x) => (int64[3] shape) {
                        shape = Shape(x)
                    }
                >
            }
        """)
        self._assert_inferred(
            graph,
            [
                oh.make_tensor_sequence_value_info(
                    "in_sequence", onnxl.TensorProto.FLOAT, (None, None, 3)
                ),
                oh.make_tensor_sequence_value_info("shapes", onnxl.TensorProto.INT64, (3,)),
            ],
        )

    def test_scan_num_scan_inputs_out_of_range_opset21(self) -> None:
        """num_scan_inputs > num_inputs must raise InferenceError (opset 21)."""
        # 2 inputs but num_scan_inputs = 9; shape inference must fail, not underflow.
        model = parser.parse_model("""
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (float[2] in0, float[3, 2] in1) => (out) {
                out = Scan <num_scan_inputs = 9, body = b (float a, float x) => (float c) {
                    c = Add(a, x)
                }> (in0, in1)
            }
            """)
        with self.assertRaisesRegex(shape_inference.InferenceError, "num_scan_inputs"):
            shape_inference.infer_shapes(model, strict_mode=True)

    def test_scan_num_scan_inputs_out_of_range_opset9(self) -> None:
        """num_scan_inputs > num_inputs must raise InferenceError (opset 9)."""
        model = parser.parse_model("""
            <ir_version: 8, opset_import: ["" : 9]>
            agraph (float[2] in0, float[3, 2] in1) => (out) {
                out = Scan <num_scan_inputs = 9, body = b (float a, float x) => (float c) {
                    c = Add(a, x)
                }> (in0, in1)
            }
            """)
        with self.assertRaisesRegex(shape_inference.InferenceError, "num_scan_inputs"):
            shape_inference.infer_shapes(model, strict_mode=True)

    def test_scan_num_scan_inputs_out_of_range_opset8(self) -> None:
        """num_scan_inputs > num_inputs must raise InferenceError (opset 8)."""
        # Opset-8 Scan has sequence_lens as its first input; 3 inputs total but
        # num_scan_inputs = 9 still exceeds the available slots.
        model = parser.parse_model("""
            <ir_version: 8, opset_import: ["" : 8]>
            agraph (float[1, 3] ls, float[1, 2, 2] si) => (out) {
                out = Scan <num_scan_inputs = 9, body = b (float a, float x) => (float c) {
                    c = Add(a, x)
                }> ("", ls, si)
            }
            """)
        with self.assertRaisesRegex(shape_inference.InferenceError, "num_scan_inputs"):
            shape_inference.infer_shapes(model, strict_mode=True)

    def test_scan_loop_state_vars_exceed_outputs_opset21(self) -> None:
        """More loop state vars than outputs must raise InferenceError (opset 21)."""
        # 3 inputs, num_scan_inputs = 1 → 2 loop state vars; only 1 output declared.
        model = parser.parse_model("""
            <ir_version: 10, opset_import: ["" : 21]>
            agraph (float[2] in0, float[2] in1, float[3, 2] in2) => (out) {
                out = Scan <num_scan_inputs = 1,
                    body = b (float a, float b, float x) => (float c) {
                        c = Identity(a)
                    }> (in0, in1, in2)
            }
            """)
        with self.assertRaisesRegex(shape_inference.InferenceError, "loop state variables"):
            shape_inference.infer_shapes(model, strict_mode=True)

    def test_scan_loop_state_vars_exceed_outputs_opset9(self) -> None:
        """More loop state vars than outputs must raise InferenceError (opset 9)."""
        model = parser.parse_model("""
            <ir_version: 8, opset_import: ["" : 9]>
            agraph (float[2] in0, float[2] in1, float[3, 2] in2) => (out) {
                out = Scan <num_scan_inputs = 1,
                    body = b (float a, float b, float x) => (float c) {
                        c = Identity(a)
                    }> (in0, in1, in2)
            }
            """)
        with self.assertRaisesRegex(shape_inference.InferenceError, "loop state variables"):
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

    def test_einsum_output_label_missing_from_inputs(self) -> None:
        """Einsum output labels absent from the inputs must raise InferenceError.

        Mirrors the fix from https://github.com/onnx/onnx/pull/8147: the missing
        label previously default-inserted index 0 instead of failing inference.
        """
        model = parser.parse_model("""
            <ir_version: 10, opset_import: ["" : 20]>
            agraph (float[2, 3] x) => (out) {
                out = Einsum <equation = "ij->k"> (x)
            }
            """)
        with self.assertRaisesRegex(shape_inference.InferenceError, "missing from the inputs"):
            shape_inference.infer_shapes(model, strict_mode=True)

    def test_hann_window_output_datatype_out_of_range(self) -> None:
        """Window ops must reject an output_datatype that does not fit in int32.

        Mirrors the fix from https://github.com/onnx/onnx/pull/8147: the attribute
        is now converted with the checked ``narrow`` helper so wrapped values fail
        inference instead of silently truncating.
        """
        model = parser.parse_model("""
            <ir_version: 10, opset_import: ["" : 20]>
            agraph (int32[1] size) => (out) {
                out = HannWindow <output_datatype = 999999999999> (size)
            }
            """)
        with self.assertRaisesRegex(shape_inference.InferenceError, "narrow"):
            shape_inference.infer_shapes(model, strict_mode=True)

    def test_function_missing_input_used_as_output_does_not_crash(self) -> None:
        """A missing model-local function input reused as an output must not crash.

        Mirrors the fix from https://github.com/onnx/onnx/pull/8305: shape
        inference could dereference a null pointer for a missing function input.
        """
        model = parser.parse_model("""
            <ir_version: 8, opset_import: ["": 25, "local": 1]>
            g (bool condition) => (float output) { output = local.F(condition) }
            <opset_import: ["": 25], domain: "local">
            F (condition, missing) => (missing) { unused = Identity(condition) }
            """)

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

    def test_function_subgraph_initializer_replaces_missing_outer_type(self) -> None:
        """A subgraph initializer shadowing a missing function input must infer.

        Mirrors the fix from https://github.com/onnx/onnx/pull/8305: shape
        inference could dereference a null pointer when a missing model-local
        function input shares its name with an initializer inside a nested
        subgraph.
        """
        model = parser.parse_model("""
            <ir_version: 8, opset_import: ["": 25, "local": 1]>
            g (bool condition) => (float output) { output = local.F(condition) }
            <opset_import: ["": 25], domain: "local">
            F (condition, missing) => (output) {
                output = If(condition) <
                    then_branch = then () => (float output)
                        <float missing = {1.0}> { output = Identity(missing) },
                    else_branch = else () => (float output)
                        <float one = {1.0}> { output = Identity(one) }
                >
            }
            """)

        checker.check_model(model)
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        assert inferred.graph.output[0].type.tensor_type.elem_type == onnxl.TensorProto.FLOAT
        assert len(inferred.graph.output[0].type.tensor_type.shape.dim) == 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
