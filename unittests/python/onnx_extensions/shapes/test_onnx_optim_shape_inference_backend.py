import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx as onnxl
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx_core.shape_inference import (
    NODE_TAG_METADATA_KEY,
    VALUE_TAG_METADATA_KEY,
    VALUE_TAGS_METADATA_KEY,
    infer_shapes_model,
    write_value_and_node_tags_to_metadata,
)
from onnx_light.onnx_py._onnxpycore import shape_inference as si

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
collect_test_cases = import_or_skip("onnx_light.onnx.backend", "collect_test_cases")


def _value_tags(graph) -> dict:
    """Reconstructs the value-name -> tag map from per-value metadata.

    The value tags live on each ValueInfoProto (inputs, value_info, outputs)
    and initializer TensorProto, never on the graph-level metadata.
    """
    tags: dict = {}
    for field in ("input", "value_info", "output", "initializer"):
        for value in getattr(graph, field):
            meta = {entry.key: entry.value for entry in value.metadata_props}
            if VALUE_TAG_METADATA_KEY in meta:
                tags[value.name] = meta[VALUE_TAG_METADATA_KEY]
    return tags


class TestOnnxOptimShapeInferenceModelBackend(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_collect_test_cases_by_name(self):
        shape_tests = []
        for test in collect_test_cases():
            if "test_cc_shape_inference_add_concat_reshape" == test.name:
                shape_tests.append(test)
        self.assertEqual(len(shape_tests), 1)
        tests = [
            test
            for test in collect_test_cases("shape")
            if "test_cc_shape_inference_add_concat_reshape" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        self.assertEqual(tests[0].name, shape_tests[0].name)

    def test_dataset_repr(self):
        abs_tests = [test for test in collect_test_cases("Abs") if test.name == "test_cc_abs"]
        self.assertEqual(len(abs_tests), 1)
        self.assertEqual(repr(abs_tests[0].data_sets[0]), "DataSet(inputs=1, outputs=1)")

    def test_tensor_repr(self):
        abs_tests = [test for test in collect_test_cases("Abs") if test.name == "test_cc_abs"]
        self.assertEqual(len(abs_tests), 1)
        ds = abs_tests[0].data_sets[0]
        self.assertEqual(repr(ds.inputs[0]), "Tensor(name='x', data_type=FLOAT, shape=[2, 3])")
        self.assertEqual(repr(ds.outputs[0]), "Tensor(name='y', data_type=FLOAT, shape=[2, 3])")

    def test_inference_shape(self):
        tests = [
            test
            for test in collect_test_cases("shape")
            if "test_cc_shape_inference_add_concat_reshape" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        model = onnxl.ModelProto()
        model.CopyFrom(test.model)
        model.graph.value_info.clear()
        infer_shapes_model(model)
        self.assertEqual(len(test.model.graph.value_info), len(model.graph.value_info))
        for expected, inferred in zip(test.model.graph.value_info, model.graph.value_info):
            self.assertEqual(expected, inferred)

    def test_inference_by_node(self):
        tests = [
            test
            for test in collect_test_cases("shape")
            if "test_cc_shape_inference_add_concat_reshape" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        expected = {info.name: info for info in test.model.graph.value_info}

        ctx = si.ShapesContext()
        for opset in test.model.opset_import:
            ctx.set_opset_version(opset.domain, opset.version)
        for inp in test.model.graph.input:
            tt = inp.type.tensor_type
            dims = [d.dim_value if d.dim_value else d.dim_param for d in tt.shape.dim]
            t = si.SymTensor(tt.elem_type, dims)
            ctx.set(inp.name, t)
        for init in test.model.graph.initializer:
            t = si.SymTensor(init.data_type, list(init.dims))
            a = onh.to_array(init)
            t.set_value_as_shape([int(i) for i in a])
            ctx.set(init.name, t)

        for node in test.model.graph.node:
            si.compute_shape_node(ctx, node)
            for out_name in node.output:
                if not out_name or out_name in {"Z"}:
                    continue
                t = ctx.get(str(out_name))
                self.assertIn(out_name, expected)
                v = expected[out_name]
                self.assertEqual(len(v.type.tensor_type.shape.dim), len(t.shape))
                for a, b in zip(v.type.tensor_type.shape.dim, t.shape):
                    self.assertEqual(a.dim_param, b)

        # outputs
        self.assertEqual(["batch", "seq", "2*d_model"], list(ctx.get("Z").shape))

    def test_inference_shape_backend_constraints(self):
        tests = [
            test
            for test in collect_test_cases("shape")
            if "test_cc_shape_inference_nonzero_chain_named" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        model = onnxl.ModelProto()
        model.CopyFrom(test.model)
        model.graph.value_info.clear()
        infer_shapes_model(model, prefill_with_value_info_output=True)
        expected_info = {
            info.name: info for info in [*test.model.graph.value_info, *test.model.graph.output]
        }
        computed = {info.name: info for info in [*model.graph.value_info, *model.graph.output]}
        self.assertEqual(set(expected_info), set(computed))
        for name in expected_info:
            expected = expected_info[name]
            inferred = computed[name]
            self.assertEqual(expected, inferred, f"{name!r} failed\n{expected=}\n--\n{inferred=}")

    def test_inference_shape_backend_16_dimension(self):
        tests = [
            test
            for test in collect_test_cases("shape")
            if "test_cc_shape_inference_shape_identity_unsqueeze" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        model = onnxl.ModelProto()
        model.CopyFrom(test.model)
        model.graph.value_info.clear()
        infer_shapes_model(model)
        expected_info = {
            info.name: info for info in [*test.model.graph.value_info, *test.model.graph.output]
        }
        computed = {info.name: info for info in [*model.graph.value_info, *model.graph.output]}
        self.assertEqual(set(expected_info), set(computed))
        for name in expected_info:
            expected = expected_info[name]
            inferred = computed[name]
            self.assertEqual(expected, inferred, f"{name!r} failed\n{expected=}\n--\n{inferred=}")

    def test_inference_shape_backend_non_zero_expression(self):
        tests = [
            test
            for test in collect_test_cases("shape")
            if "test_cc_shape_inference_nonzero_plus_expression" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        model = onnxl.ModelProto()
        model.CopyFrom(test.model)
        model.graph.value_info.clear()
        infer_shapes_model(model, prefill_with_value_info_output=True)
        expected_info = {
            info.name: info for info in [*test.model.graph.value_info, *test.model.graph.output]
        }
        computed = {info.name: info for info in [*model.graph.value_info, *model.graph.output]}
        self.assertEqual(set(expected_info), set(computed))
        for name in expected_info:
            expected = expected_info[name]
            inferred = computed[name]
            self.assertEqual(expected, inferred, f"{name!r} failed\n{expected=}\n--\n{inferred=}")

    def test_peak_memory_backend_case_metadata(self):
        """Checks that backend peak-memory cases reproduce their pre-embedded node metadata."""
        tests = list(collect_test_cases("peak_memory"))
        self.assertNotEmpty(tests)

        for test in tests:
            with self.subTest(name=test.name):
                expected_node_meta = [
                    {entry.key: entry.value for entry in node.metadata_props}
                    for node in test.model.graph.node
                ]

                model_copy = onnxl.ModelProto()
                model_copy.CopyFrom(test.model)
                for node in model_copy.graph.node:
                    node.metadata_props.clear()

                ctx = si.ShapesContext()
                si.compute_shape_model(ctx, model_copy)
                si.write_peak_memory_to_metadata(ctx, model_copy.graph, si.Device.kCPU)

                self.assertEqual(len(model_copy.graph.node), len(expected_node_meta))
                for node, expected_meta in zip(model_copy.graph.node, expected_node_meta):
                    got_meta = {entry.key: entry.value for entry in node.metadata_props}
                    self.assertEqual(got_meta, expected_meta)

    def test_inference_shape_backend_floordiv_offset_expression(self):
        tests = [
            test
            for test in collect_test_cases("shape")
            if "test_cc_shape_inference_floordiv_offset_expression" == test.name
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]
        model = onnxl.ModelProto()
        model.CopyFrom(test.model)
        model.graph.value_info.clear()
        infer_shapes_model(model)
        output_shapes = {info.name: info for info in model.graph.output}
        self.assertIn("Y", output_shapes)
        dims = output_shapes["Y"].type.tensor_type.shape.dim
        self.assertEqual(len(dims), 3)
        self.assertEqual(dims[0].dim_param, "batch")
        self.assertEqual(dims[1].dim_param, "channel")
        self.assertEqual(dims[2].dim_param, "seq//5+2")

    def test_shape_tag_backend_case_metadata(self):
        """Verifies that the shape-tag backend case has expected metadata pre-embedded
        and that write_value_and_node_tags_to_metadata reproduces it on a blank copy."""
        tests = [
            test
            for test in collect_test_cases("shape_tag")
            if test.name == "test_cc_shape_tag_shape_reshape"
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]

        # Verify pre-embedded graph metadata.
        graph_meta = {entry.key: entry.value for entry in test.model.graph.metadata_props}
        self.assertNotIn(VALUE_TAGS_METADATA_KEY, graph_meta)
        value_tags = _value_tags(test.model.graph)
        self.assertEqual(value_tags.get("S"), "shape")

        # Verify pre-embedded node metadata on the Shape node (node 0).
        node_meta = {entry.key: entry.value for entry in test.model.graph.node[0].metadata_props}
        self.assertEqual(node_meta.get(NODE_TAG_METADATA_KEY), "shape")

        # Make a blank copy (strip all metadata) and recompute.
        model_copy = onnxl.ModelProto()
        model_copy.CopyFrom(test.model)
        model_copy.graph.metadata_props.clear()
        for node in model_copy.graph.node:
            node.metadata_props.clear()
        for vi in model_copy.graph.value_info:
            vi.metadata_props.clear()

        write_value_and_node_tags_to_metadata(model_copy.graph)

        computed_graph_meta = {
            entry.key: entry.value for entry in model_copy.graph.metadata_props
        }
        self.assertNotIn(VALUE_TAGS_METADATA_KEY, computed_graph_meta)
        computed_value_tags = _value_tags(model_copy.graph)
        self.assertEqual(computed_value_tags, value_tags)

        computed_node_meta = {
            entry.key: entry.value for entry in model_copy.graph.node[0].metadata_props
        }
        self.assertEqual(computed_node_meta.get(NODE_TAG_METADATA_KEY), "shape")

        # Verify onnx_light.value_tag is also written on value_info for "S".
        s_vi = next(vi for vi in model_copy.graph.value_info if vi.name == "S")
        s_vi_meta = {entry.key: entry.value for entry in s_vi.metadata_props}
        self.assertEqual(s_vi_meta.get(VALUE_TAG_METADATA_KEY), "shape")

    def test_shape_tag_constant_reshape_backend_case_metadata(self):
        """Verifies that the shape-tag backend case (Constant→Reshape) has expected
        metadata pre-embedded and that write_value_and_node_tags_to_metadata reproduces it.
        Shape tag wins over weight tag, so S receives ``"shape"`` (not ``"ambiguous"``)."""
        tests = [
            test
            for test in collect_test_cases("shape_tag")
            if test.name == "test_cc_shape_tag_constant_reshape_ambiguous"
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]

        # Verify pre-embedded graph metadata.
        graph_meta = {entry.key: entry.value for entry in test.model.graph.metadata_props}
        self.assertNotIn(VALUE_TAGS_METADATA_KEY, graph_meta)
        value_tags = _value_tags(test.model.graph)
        self.assertEqual(value_tags.get("S"), "shape")

        # Verify pre-embedded node metadata on the Constant node (node 0).
        node_meta = {entry.key: entry.value for entry in test.model.graph.node[0].metadata_props}
        self.assertEqual(node_meta.get(NODE_TAG_METADATA_KEY), "shape")

        # Reshape (node 1) must have node_tag = "weight" (inherited from graph input X).
        reshape_meta = {
            entry.key: entry.value for entry in test.model.graph.node[1].metadata_props
        }
        self.assertEqual(reshape_meta.get(NODE_TAG_METADATA_KEY), "weight")

        # Make a blank copy (strip all metadata) and recompute.
        model_copy = onnxl.ModelProto()
        model_copy.CopyFrom(test.model)
        model_copy.graph.metadata_props.clear()
        for node in model_copy.graph.node:
            node.metadata_props.clear()
        for vi in model_copy.graph.value_info:
            vi.metadata_props.clear()

        write_value_and_node_tags_to_metadata(model_copy.graph)

        computed_graph_meta = {
            entry.key: entry.value for entry in model_copy.graph.metadata_props
        }
        self.assertNotIn(VALUE_TAGS_METADATA_KEY, computed_graph_meta)
        computed_value_tags = _value_tags(model_copy.graph)
        self.assertEqual(computed_value_tags, value_tags)

        computed_node_meta = {
            entry.key: entry.value for entry in model_copy.graph.node[0].metadata_props
        }
        self.assertEqual(computed_node_meta.get(NODE_TAG_METADATA_KEY), "shape")

        # Verify onnx_light.value_tag = "shape" is written on value_info for "S".
        s_vi = next(vi for vi in model_copy.graph.value_info if vi.name == "S")
        s_vi_meta = {entry.key: entry.value for entry in s_vi.metadata_props}
        self.assertEqual(s_vi_meta.get(VALUE_TAG_METADATA_KEY), "shape")

    def test_shape_tag_constant_mul_concat_reshape_backend_case_metadata(self):
        """Verifies the Constant → Mul → Concat → Reshape shape-tag backend case.

        Checks that the pre-embedded metadata is consistent and that
        write_value_and_node_tags_to_metadata reproduces it on a blank copy.
        Expected tags: S1, S2, S_full → ``"shape"``; two, X, Y → ``"weight"``.
        """
        tests = [
            test
            for test in collect_test_cases("shape_tag")
            if test.name == "test_cc_shape_tag_constant_mul_concat_reshape"
        ]
        self.assertEqual(len(tests), 1)
        test = tests[0]

        # Verify pre-embedded graph metadata.
        graph_meta = {entry.key: entry.value for entry in test.model.graph.metadata_props}
        self.assertNotIn(VALUE_TAGS_METADATA_KEY, graph_meta)
        value_tags = _value_tags(test.model.graph)
        self.assertEqual(value_tags.get("S1"), "shape")
        self.assertEqual(value_tags.get("S2"), "shape")
        self.assertEqual(value_tags.get("S_full"), "shape")
        self.assertEqual(value_tags.get("two"), "weight")

        # Verify pre-embedded node metadata.
        node0_meta = {entry.key: entry.value for entry in test.model.graph.node[0].metadata_props}
        self.assertEqual(node0_meta.get(NODE_TAG_METADATA_KEY), "shape")  # Constant → S1

        node1_meta = {entry.key: entry.value for entry in test.model.graph.node[1].metadata_props}
        self.assertEqual(node1_meta.get(NODE_TAG_METADATA_KEY), "weight")  # Constant → two

        node2_meta = {entry.key: entry.value for entry in test.model.graph.node[2].metadata_props}
        self.assertEqual(node2_meta.get(NODE_TAG_METADATA_KEY), "shape")  # Mul

        node3_meta = {entry.key: entry.value for entry in test.model.graph.node[3].metadata_props}
        self.assertEqual(node3_meta.get(NODE_TAG_METADATA_KEY), "shape")  # Concat

        node4_meta = {entry.key: entry.value for entry in test.model.graph.node[4].metadata_props}
        self.assertEqual(node4_meta.get(NODE_TAG_METADATA_KEY), "weight")  # Reshape → "weight"

        # Make a blank copy (strip all metadata) and recompute.
        model_copy = onnxl.ModelProto()
        model_copy.CopyFrom(test.model)
        model_copy.graph.metadata_props.clear()
        for node in model_copy.graph.node:
            node.metadata_props.clear()
        for vi in model_copy.graph.value_info:
            vi.metadata_props.clear()

        write_value_and_node_tags_to_metadata(model_copy.graph)

        computed_graph_meta = {
            entry.key: entry.value for entry in model_copy.graph.metadata_props
        }
        self.assertNotIn(VALUE_TAGS_METADATA_KEY, computed_graph_meta)
        computed_value_tags = _value_tags(model_copy.graph)
        self.assertEqual(computed_value_tags, value_tags)

        # Verify onnx_light.value_tag is written on each value_info.
        vi_by_name = {vi.name: vi for vi in model_copy.graph.value_info}
        for tensor_name, expected_tag in [
            ("S1", "shape"),
            ("two", "weight"),
            ("S2", "shape"),
            ("S_full", "shape"),
        ]:
            vi_meta = {entry.key: entry.value for entry in vi_by_name[tensor_name].metadata_props}
            self.assertEqual(
                vi_meta.get(VALUE_TAG_METADATA_KEY),
                expected_tag,
                f"value_tag mismatch for {tensor_name!r}",
            )

    def test_sub_backward_propagation_tags_mask_tensors(self):
        """Verifies that Sub backward propagation tags mask_float and mask_4d as
        `"weight"` in the tiny-llm shape-inference case.

        The propagation chain is:
          Sub(mask_one:"weight", mask_4d:untagged) → mask_inv:"weight"
          Sub backward {0,1} → mask_4d ← "weight"
          Unsqueeze backward {0} → mask_float ← "weight"
          Cast backward {0} → attention_mask ← "weight"
        """
        tests = [
            test
            for test in collect_test_cases("shape")
            if test.name == "test_cc_shape_inference_tiny_llm"
        ]
        self.assertEqual(len(tests), 1, "test_cc_shape_inference_tiny_llm not found")
        test = tests[0]

        # Verify pre-embedded graph-level value_tags include mask_float, mask_4d, and
        # attention_mask (now tagged via Cast backward propagation).
        graph_meta = {entry.key: entry.value for entry in test.model.graph.metadata_props}
        self.assertNotIn(VALUE_TAGS_METADATA_KEY, graph_meta)
        value_tags = _value_tags(test.model.graph)
        self.assertEqual(value_tags.get("mask_float"), "weight")
        self.assertEqual(value_tags.get("mask_4d"), "weight")
        self.assertEqual(value_tags.get("attention_mask"), "weight")

        # Make a blank copy (strip all metadata) and recompute.
        model_copy = onnxl.ModelProto()
        model_copy.CopyFrom(test.model)
        model_copy.graph.metadata_props.clear()
        for node in model_copy.graph.node:
            node.metadata_props.clear()
        for vi in model_copy.graph.value_info:
            vi.metadata_props.clear()
        for init in model_copy.graph.initializer:
            init.metadata_props.clear()

        write_value_and_node_tags_to_metadata(model_copy.graph)

        computed_graph_meta = {
            entry.key: entry.value for entry in model_copy.graph.metadata_props
        }
        self.assertNotIn(VALUE_TAGS_METADATA_KEY, computed_graph_meta)
        computed_value_tags = _value_tags(model_copy.graph)
        self.assertEqual(computed_value_tags.get("mask_float"), "weight")
        self.assertEqual(computed_value_tags.get("mask_4d"), "weight")
        self.assertEqual(computed_value_tags.get("attention_mask"), "weight")
        self.assertEqual(computed_value_tags, value_tags)


if __name__ == "__main__":
    unittest.main()
