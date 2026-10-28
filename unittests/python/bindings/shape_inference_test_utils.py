from typing import Any

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference


class TestShapeInferenceHelper(ExtTestCase):
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
