# source: https://github.com/onnx/onnx/blob/main/onnx/test/checker_test.py
import unittest

from onnx_light.ext_test_case import ExtTestCase
import onnx_light.onnx as onnxl
import onnx_light.onnx.checker as checker
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
import onnx_light.onnx.shape_inference as shape_inference


class TestChecker(ExtTestCase):
    def make_code_model(self, nodes=None):
        """Builds a minimal valid model for error-code tests."""
        return oh.make_model(
            oh.make_graph(
                nodes if nodes is not None else [oh.make_node("Identity", ["x"], ["y"])],
                "codes",
                [oh.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, [2])],
                [oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [2])],
            ),
            ir_version=10,
            opset_imports=[oh.make_opsetid("", 21)],
        )

    def assert_model_code(self, model, code):
        """Checks the stable code of a rejected model."""
        with self.assertRaises(checker.ValidationError) as caught:
            checker.check_model(model)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_validation_error_model_codes(self):
        for version, code in (
            (0, "ir-version-missing"),
            (defs.IR_VERSION + 1, "ir-version-unsupported"),
        ):
            with self.subTest(code=code):
                model = self.make_code_model()
                model.ir_version = version
                error = self.assert_model_code(model, code)
                if version == 0:
                    self.assertEqual(
                        str(error), "The model does not have an ir_version set properly."
                    )

        model = self.make_code_model()
        for value in ("a", "b"):
            entry = model.metadata_props.add()
            entry.key, entry.value = "key", value
        self.assert_model_code(model, "metadata-props-duplicate-key")

        model = self.make_code_model()
        model.graph.name = ""
        self.assert_model_code(model, "required-field-empty")
        model = self.make_code_model()
        value = onnxl.ValueInfoProto()
        value.name = "x"
        model.graph.input[0].CopyFrom(value)
        self.assert_model_code(model, "required-field-missing")

        model = self.make_code_model()
        model.graph.input[0].type.tensor_type.elem_type = onnxl.TensorProto.FLOAT4E2M1
        self.assert_model_code(model, "data-type-requires-newer-ir-version")
        model = self.make_code_model()
        del model.opset_import[:]
        self.assert_model_code(model, "opset-import-missing")
        model = self.make_code_model()
        model.graph.output[0].name = "never_produced"
        self.assert_model_code(model, "graph-output-not-produced")

    def test_validation_error_node_codes(self):
        cases = [
            ([oh.make_node("Identity", ["typo"], ["y"])], "input-not-defined"),
            (
                [
                    oh.make_node("Identity", ["later"], ["y"]),
                    oh.make_node("Identity", ["x"], ["later"]),
                ],
                "input-not-defined",
            ),
            ([oh.make_node("Identity", ["x"], ["y"])] * 2, "not-ssa"),
            ([oh.make_node("NoSuchOp", ["x"], ["y"])], "operator-not-registered"),
            ([oh.make_node("Op", ["x"], ["y"], domain="custom")], "opset-import-missing"),
            ([oh.make_node("Add", ["x"], ["y"])], "node-schema-mismatch"),
            ([oh.make_node("Identity", [""], ["y"])], "node-schema-mismatch"),
            ([oh.make_node("Identity", ["x"], [""])], "node-schema-mismatch"),
            ([oh.make_node("Cast", ["x"], ["y"])], "node-schema-mismatch"),
            ([oh.make_node("LeakyRelu", ["x"], ["y"], alpha=1)], "node-schema-mismatch"),
            ([oh.make_node("Identity", ["x"], ["y"], unknown=1)], "node-schema-mismatch"),
            ([oh.make_node("Upsample", ["x"], ["y"])], "operator-deprecated"),
        ]
        for nodes, code in cases:
            with self.subTest(op=nodes[0].op_type, code=code):
                model = self.make_code_model(nodes)
                error = self.assert_model_code(model, code)
                if code in {"node-schema-mismatch", "operator-not-registered"}:
                    self.assertIn("==> Context:", str(error))

        model = self.make_code_model()
        for _ in range(2):
            model.graph.node[0].attribute.append(oh.make_attribute("alpha", 1.0))
        self.assert_model_code(model, "node-schema-mismatch")
        model = self.make_code_model()
        for _ in range(2):
            model.graph.initializer.append(
                oh.make_tensor("w", onnxl.TensorProto.FLOAT, [2], [0.0, 0.0])
            )
        self.assert_model_code(model, "not-ssa")

    def test_validation_error_nested_context_and_function(self):
        branch = oh.make_graph(
            [oh.make_node("NoSuchOp", ["x"], ["z"])],
            "branch",
            [],
            [oh.make_tensor_value_info("z", onnxl.TensorProto.FLOAT, [2])],
        )
        model = self.make_code_model(
            [oh.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch)]
        )
        model.graph.input.append(oh.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, []))
        error = self.assert_model_code(model, "operator-not-registered")
        self.assertIn("==> Context:", str(error))

        for nodes, code in (
            ([oh.make_node("Identity", ["undefined"], ["b"])], "input-not-defined"),
            ([oh.make_node("Op", ["a"], ["b"], domain="custom")], "opset-import-missing"),
            (
                [oh.make_node("Identity", ["a"], ["b"]), oh.make_node("Identity", ["a"], ["b"])],
                "not-ssa",
            ),
        ):
            with self.subTest(code=code):
                model = self.make_code_model()
                model.functions.append(
                    oh.make_function("local", "f", ["a"], ["b"], nodes, [oh.make_opsetid("", 21)])
                )
                self.assert_model_code(model, code)

    def test_validation_error_external_codes(self):
        for entries, inline, code in (
            ({}, False, "external-data-entry-invalid"),
            ({"location": ""}, False, "external-data-location-invalid"),
            ({"location": "#memory", "offset": "-1"}, False, "external-data-entry-invalid"),
            ({"location": "#memory"}, True, "external-data-entry-invalid"),
        ):
            with self.subTest(entries=entries, inline=inline):
                model = self.make_code_model()
                tensor = onnxl.TensorProto()
                tensor.name = "w"
                tensor.data_type = onnxl.TensorProto.FLOAT
                tensor.dims.append(2)
                tensor.data_location = onnxl.TensorProto.EXTERNAL
                if inline:
                    tensor.raw_data = b"\x00" * 8
                for key, value in entries.items():
                    entry = tensor.external_data.add()
                    entry.key, entry.value = key, value
                model.graph.initializer.append(tensor)
                self.assert_model_code(model, code)
        for location in ("#memory", "./#memory", "#unused/../#memory"):
            with self.subTest(location=location):
                tensor.ClearField("raw_data")
                tensor.external_data[0].value = location
                model.graph.initializer[0].CopyFrom(tensor)
                checker.check_model(model)

    def test_validation_error_no_code_compatibility(self):
        self.assertIsNone(checker.ValidationError("message").code)
        model = self.make_code_model()
        model.ir_version = 2
        self.assert_model_code(model, None)
        with self.assertRaises(checker.ValidationError) as caught:
            checker.check_attribute(onnxl.AttributeProto())
        self.assertIsNone(caught.exception.code)

    def test_validation_error_infer_node_outputs(self):
        for node in (
            oh.make_node("Add", ["x"], ["y"]),
            oh.make_node("Add", ["x", "x"], []),
            oh.make_node("Add", ["x", "x"], ["y"], unknown=1),
        ):
            with self.subTest(node=node):
                with self.assertRaises(checker.ValidationError) as caught:
                    shape_inference.infer_node_outputs(
                        defs.get_schema("Add", 21),
                        node,
                        {"x": oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [2])},
                    )
                self.assertEqual(caught.exception.code, "node-schema-mismatch")

    def test_operator_function_definitions_are_accepted(self):
        for domain in ("", "ai.onnx", "onnx.ai", "ai.onnx.ml"):
            for overload in ("", "custom"):
                with self.subTest(domain=domain, overload=overload):
                    name = "Scaler" if domain == "ai.onnx.ml" else "Add"
                    function = oh.make_function(
                        domain,
                        name,
                        ["X"],
                        ["Y"],
                        [oh.make_node("Identity", ["X"], ["Y"])],
                        [oh.make_opsetid("", 18)],
                    )
                    function.overload = overload
                    model = oh.make_model(
                        oh.make_graph([], "unused_function", [], []),
                        functions=[function],
                        opset_imports=[oh.make_opsetid("", 18)],
                    )
                    checker.check_model(model)

    def test_maximum_protobuf_matches_cpp_parser_limit(self) -> None:
        """Checks that the exported protobuf size limit matches the C++ parser boundary."""
        self.assertEqual(checker.MAXIMUM_PROTOBUF, 2147483647)

    def make_sparse(
        self,
        shape: tuple[int, ...],
        values: list[int],
        indices_shape: tuple[int, ...],
        indices: list[int],
    ) -> onnxl.SparseTensorProto:
        """Builds a sparse tensor proto with explicit values and indices."""
        sparse = onnxl.SparseTensorProto()
        sparse.dims.extend(shape)
        nnz = len(values)
        sparse.values.CopyFrom(oh.make_tensor("spval", onnxl.TensorProto.INT64, (nnz,), values))
        sparse.indices.CopyFrom(
            oh.make_tensor("spind", onnxl.TensorProto.INT64, indices_shape, indices)
        )
        return sparse

    def test_check_attribute(self) -> None:
        """Checks that a single attribute value passes checker validation."""
        attr = onnxl.AttributeProto()
        attr.name = "test"
        attr.i = 2
        checker.check_attribute(attr)

    def test_check_attribute_fails_without_value(self) -> None:
        """Checks that an attribute without value fails checker validation."""
        attr = onnxl.AttributeProto()
        attr.name = "test"
        with self.assertRaises(checker.ValidationError):
            checker.check_attribute(attr)

    def test_check_attribute_fails_with_two_values(self) -> None:
        """Checks that an attribute with multiple values fails validation."""
        attr = onnxl.AttributeProto()
        attr.name = "test"
        attr.i = 2
        attr.f = 1.0
        with self.assertRaises(checker.ValidationError):
            checker.check_attribute(attr)

    def test_check_model_type_attributes(self) -> None:
        """Checks singular and repeated types against the model catalogue."""
        declaration = onnxl.StructTypeProto(
            type_id=1,
            bit_packing=onnxl.StructTypeProto.BitPacking(
                dimension=1,
                component=[onnxl.StructTypeProto.BitPacking.Component(name="code", bit_width=8)],
            ),
        )
        tensor_type = oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, [1])
        types = {
            "tensor": tensor_type,
            "known_reference": onnxl.TypeProto(struct_type=onnxl.StructTypeProto(type_ref=1)),
            "unknown_reference": onnxl.TypeProto(struct_type=onnxl.StructTypeProto(type_ref=99)),
            "empty": onnxl.TypeProto(),
        }
        for repeated in (False, True):
            for name, value in types.items():
                with self.subTest(repeated=repeated, value=name):
                    attribute = onnxl.AttributeProto(name="type")
                    if repeated:
                        attribute.type = onnxl.AttributeProto.TYPE_PROTOS
                        attribute.type_protos.extend([tensor_type, value])
                    else:
                        attribute.type = onnxl.AttributeProto.TYPE_PROTO
                        attribute.tp.CopyFrom(value)
                    node = oh.make_node("TypeCarrier", ["X"], ["Y"], domain="test.types")
                    node.attribute.append(attribute)
                    graph = oh.make_graph(
                        [node],
                        "type_attributes",
                        [oh.make_value_info("X", tensor_type)],
                        [oh.make_value_info("Y", tensor_type)],
                    )
                    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("test.types", 1)])
                    model.struct_types.append(declaration)
                    if name in {"tensor", "known_reference"}:
                        checker.check_model(model)
                    else:
                        with self.assertRaises(checker.ValidationError):
                            checker.check_model(model)

    def test_check_sparse_tensor(self) -> None:
        """Checks that a 2D sparse tensor shape passes checker validation."""
        sparse = self.make_sparse((2, 3), [1, 2], (2, 2), [0, 1, 1, 2])
        checker.check_sparse_tensor(sparse)

    def test_check_sparse_tensor_invalid_shape(self) -> None:
        """Checks that a non-2D sparse tensor shape fails validation."""
        sparse = self.make_sparse((6,), [1, 2], (2,), [0, 5])
        with self.assertRaises(checker.ValidationError):
            checker.check_sparse_tensor(sparse)

    def test_check_graph_type_validation(self) -> None:
        """Verifies that a non-GraphProto input fails graph validation."""
        with self.assertRaises(checker.ValidationError):
            checker.check_graph("not a graph")  # type: ignore[arg-type]

    def test_check_model_metadata_props(self) -> None:
        """Checks that duplicated metadata keys fail model validation."""
        node = oh.make_node("Relu", ["X"], ["Y"])
        graph = oh.make_graph(
            [node],
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1, 2])],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1, 2])],
        )
        model_def = oh.make_model(graph, producer_name="test")
        oh.set_model_props(model_def, {"Title": "my graph", "Keywords": "test;graph"})
        checker.check_model(model_def)
        dupe = model_def.metadata_props.add()
        dupe.key = "Title"
        dupe.value = "Other"
        with self.assertRaises(checker.ValidationError):
            checker.check_model(model_def)

    def test_checker_check_model_raises_checker_validation_error(self) -> None:
        """Checks that checker.check_model raises checker.ValidationError on invalid model."""
        graph = oh.make_graph(
            [oh.make_node("Relu", ["X"], ["Y"])],
            "test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1, 2])],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1, 2])],
        )
        model_def = oh.make_model(graph, producer_name="test")
        duplicate = model_def.metadata_props.add()
        duplicate.key = "a"
        duplicate.value = "1"
        duplicate = model_def.metadata_props.add()
        duplicate.key = "a"
        duplicate.value = "2"

        with self.assertRaises(checker.ValidationError):
            checker.check_model(model_def)

    def test_check_model_external_data_empty_location_bypass(self) -> None:
        """Checks that an empty external data location can be bypassed on request."""
        tensor = onnxl.TensorProto()
        tensor.name = "W"
        tensor.data_type = onnxl.TensorProto.FLOAT
        tensor.dims.append(1)
        tensor.data_location = onnxl.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key = "location"
        entry.value = ""  # empty location is normally rejected
        graph = oh.make_graph(
            [oh.make_node("Relu", ["W"], ["Y"])],
            "test",
            [],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1])],
            initializer=[tensor],
        )
        model_def = oh.make_model(graph, producer_name="test")
        with self.assertRaises(checker.ValidationError):
            checker.check_model(model_def)
        # Not recommended, but explicitly requested: the check is bypassed.
        checker.check_model(model_def, skip_external_data_location_check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
