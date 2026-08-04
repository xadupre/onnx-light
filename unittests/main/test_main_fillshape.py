# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``python -m onnx_light fillshape`` (``onnx_light.__main__``)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import defs, load


def _make_add_model(input_shape=None) -> onnxl.ModelProto:
    """Builds ``Y = Add(X, X)`` with unresolved output shape.

    :param input_shape: Shape for input X. Defaults to ``[2, 3]``. Each entry
        may be an ``int`` (concrete dim) or a ``str`` (symbolic dim_param).
    """
    if input_shape is None:
        input_shape = [2, 3]
    add = oh.make_node("Add", inputs=["X", "X"], outputs=["Y"])
    x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, input_shape)
    y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
    graph = oh.make_graph([add], "g", [x], [y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = 8
    return model


class TestMainFillshape(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    def _save_model(self, model: onnxl.ModelProto, path: str) -> None:
        from onnx_light.onnx import save

        save(model, path)

    def test_fillshape_overwrites_in_place(self):
        """fillshape with no extra options writes shapes back to the input file."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_add_model(), model_path)

            main(["fillshape", model_path])

            result = load(model_path)
            y = result.graph.output[0]
            self.assertEqual(y.name, "Y")
            dims = list(y.type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            self.assertEqual(dims[0].dim_value, 2)
            self.assertEqual(dims[1].dim_value, 3)

    def test_fillshape_output_option(self):
        """fillshape --output writes to the given file and leaves the input unchanged."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            output_path = os.path.join(tmp, "out.onnx")
            self._save_model(_make_add_model(), model_path)

            main(["fillshape", model_path, "--output", output_path])

            # Output file must exist and carry shapes.
            self.assertTrue(os.path.exists(output_path))
            result = load(output_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)

            # Input file must be unchanged (no shapes).
            original = load(model_path)
            orig_dims = list(original.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(orig_dims), 0)

    def test_fillshape_show_option(self, capsys=None):
        """fillshape --show prints shapes and does not save the model."""
        import io
        from contextlib import redirect_stdout

        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            mtime_before = None
            self._save_model(_make_add_model(), model_path)
            mtime_before = os.path.getmtime(model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["fillshape", model_path, "--show"])

            output = buf.getvalue()
            self.assertIn("Add", output)

            # File must not have been modified.
            self.assertEqual(os.path.getmtime(model_path), mtime_before)

    def test_fillshape_keep_option(self):
        """fillshape --keep uses existing shapes as anchors."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Use a symbolic input dim "N" so the inferred output dim is also
            # symbolic and the anchor "ANCHOR" is non-conflicting (both symbolic).
            model = _make_add_model(input_shape=["N", 3])
            # Pre-set the output with a symbolic anchor dim.
            model.graph.output[0].CopyFrom(
                oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, ["ANCHOR", 3])
            )
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--keep"])

            result = load(model_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            # Anchor should be preserved by --keep.
            self.assertTrue(dims[0].has_dim_param())
            self.assertEqual(dims[0].dim_param, "ANCHOR")

    def test_fillshape_verbose_default_level(self):
        """fillshape --verbose prints a summary of shape-inference events."""
        import io
        from contextlib import redirect_stdout

        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_add_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["fillshape", model_path, "--verbose"])

            output = buf.getvalue()
            self.assertIn("[fillshape] shape inference events:", output)
            self.assertNotIn("action=compute_node", output)

    def test_fillshape_verbose_level_2(self):
        """fillshape --verbose 2 prints detailed shape-inference events."""
        import io
        from contextlib import redirect_stdout

        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_add_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["fillshape", model_path, "--verbose", "2"])

            output = buf.getvalue()
            self.assertIn("[fillshape] shape inference events:", output)
            self.assertIn("action=compute_node", output)

    def test_fillshape_inplace_info_option(self):
        """fillshape --inplace-info writes reuse metadata into node metadata_props."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Build Abs(X)->A, Abs(A)->Y: node 1 can reuse A's buffer.
            abs0 = oh.make_node("Abs", inputs=["X"], outputs=["A"])
            abs1 = oh.make_node("Abs", inputs=["A"], outputs=["Y"])
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
            graph = oh.make_graph([abs0, abs1], "g", [x], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--inplace-info"])

            result = load(model_path)
            # Shapes must also be filled.
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)

            # Node 1 (Abs A->Y) should have in-place and release metadata.
            node1_meta = {entry.key: entry.value for entry in result.graph.node[1].metadata_props}
            self.assertIn("onnx_light.inplace_reuse", node1_meta)
            self.assertEqual(node1_meta["onnx_light.release_after"], "A")

    def test_fillshape_release_info_option(self):
        """fillshape --release-info writes release metadata into node metadata_props."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Build Abs(X)->A, Abs(A)->Y: node 1 can release A after execution.
            abs0 = oh.make_node("Abs", inputs=["X"], outputs=["A"])
            abs1 = oh.make_node("Abs", inputs=["A"], outputs=["Y"])
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
            graph = oh.make_graph([abs0, abs1], "g", [x], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--release-info"])

            result = load(model_path)
            # Shapes must also be filled.
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)

            # Node 1 (Abs A->Y) should have release metadata.
            node1_meta = {entry.key: entry.value for entry in result.graph.node[1].metadata_props}
            self.assertEqual(node1_meta["onnx_light.release_after"], "A")
            self.assertNotIn("onnx_light.inplace_reuse", node1_meta)

    def test_fillshape_shape_tag_option(self):
        """fillshape --shape-tag writes value/node tag metadata into the model."""
        from onnx_light.__main__ import main
        from onnx_light.onnx_core.shape_inference import (
            NODE_TAG_METADATA_KEY,
            VALUE_TAG_METADATA_KEY,
            VALUE_TAGS_METADATA_KEY,
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Build Shape(X)->S, Reshape(X, S)->Y: S should get the "shape" tag.
            shape_node = oh.make_node("Shape", inputs=["X"], outputs=["S"])
            reshape_node = oh.make_node("Reshape", inputs=["X", "S"], outputs=["Y"])
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
            graph = oh.make_graph([shape_node, reshape_node], "g", [x], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--shape-tag"])

            result = load(model_path)
            # Shapes must also be filled.
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)

            # No feature metadata must be stored on the graph metadata.
            graph_meta = {entry.key: entry.value for entry in result.graph.metadata_props}
            self.assertNotIn(VALUE_TAGS_METADATA_KEY, graph_meta)

            # Shape node (node 0) should carry the "shape" node tag.
            node0_meta = {entry.key: entry.value for entry in result.graph.node[0].metadata_props}
            self.assertIn(NODE_TAG_METADATA_KEY, node0_meta)
            self.assertEqual(node0_meta[NODE_TAG_METADATA_KEY], "shape")

            # The graph output Y inherits "weight" from X (first Reshape input).
            output_meta = {
                entry.key: entry.value for entry in result.graph.output[0].metadata_props
            }
            self.assertEqual(output_meta.get(VALUE_TAG_METADATA_KEY), "weight")

    def test_fillshape_shape_tag_output_is_shape(self):
        """fillshape --shape-tag writes 'shape' value_tag on a model output that is a shape tensor."""  # noqa: E501
        from onnx_light.__main__ import main
        from onnx_light.onnx_core.shape_inference import (
            NODE_TAG_METADATA_KEY,
            VALUE_TAG_METADATA_KEY,
            VALUE_TAGS_METADATA_KEY,
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Build Shape(X) → Y where Y is directly the graph output.
            shape_node = oh.make_node("Shape", inputs=["X"], outputs=["Y"])
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.INT64, None)
            graph = oh.make_graph([shape_node], "g", [x], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--shape-tag"])

            result = load(model_path)
            # Shape inference must fill the output shape ([2] — one dimension per input dim).
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 1)

            # No feature metadata must be stored on the graph metadata.
            graph_meta = {entry.key: entry.value for entry in result.graph.metadata_props}
            self.assertNotIn(VALUE_TAGS_METADATA_KEY, graph_meta)

            # The Shape node should carry the "shape" node tag.
            node0_meta = {entry.key: entry.value for entry in result.graph.node[0].metadata_props}
            self.assertEqual(node0_meta.get(NODE_TAG_METADATA_KEY), "shape")

            # The graph output Y must carry onnx_light.value_tag = "shape".
            output_meta = {
                entry.key: entry.value for entry in result.graph.output[0].metadata_props
            }
            self.assertEqual(output_meta.get(VALUE_TAG_METADATA_KEY), "shape")

    def test_fillshape_missing_file_raises(self):
        """fillshape raises when the model file does not exist."""
        from onnx_light.__main__ import main

        with self.assertRaises(OSError):
            main(["fillshape", "/nonexistent/path/model.onnx"])

    def test_fillshape_external_data_not_loaded(self):
        """fillshape loads the model without fetching large external weight bytes.

        Large tensors (at or above the tiny-tensor threshold) must remain as
        external-data references after fillshape so that the .data file is not
        inlined unnecessarily.
        """
        from onnx_light.__main__ import main
        from onnx_light.onnx import save
        from onnx_light.onnx_lib.external_data_helper import uses_external_data

        # W is 8 × 8 × float32 = 256 bytes, well above the 128-byte tiny-tensor
        # threshold, so it must remain external after fillshape.
        values = np.zeros((8, 8), dtype=np.float32)
        init = oh.make_tensor("W", onnxl.TensorProto.FLOAT, [8, 8], values.tobytes(), raw=True)
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [8, 8])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        node = oh.make_node("Add", ["X", "W"], ["Y"])
        graph = oh.make_graph([node], "g", [x], [y], [init])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = os.path.join(model_dir, "model.onnx")
            save(model, model_path, save_as_external_data=True, size_threshold=0)

            # The weight file must exist beside the model.
            weight_file = model_path + ".data"
            self.assertTrue(os.path.exists(weight_file))

            # fillshape must succeed even though weights are external.
            main(["fillshape", model_path])

            # The output model should still reference external data (W is large).
            result = load(model_path, load_external_data=False)
            self.assertTrue(any(uses_external_data(i) for i in result.graph.initializer))

            # Shapes must be filled.
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)

    def test_fillshape_tiny_external_tensor_loaded_for_shape_inference(self):
        """fillshape loads tiny external tensors so shape inference can use their values.

        When a model is saved with ``size_threshold=0`` all tensors go to the
        external data file, including small shape-constant tensors such as the
        ``shape`` input of a Reshape node.  fillshape must still infer the
        output shape correctly by loading those tiny tensors from disk.
        """
        from onnx_light.__main__ import main
        from onnx_light.onnx import save

        # shape initializer: [2, 3] as int64 raw_data = 16 bytes (< 128-byte threshold).
        shape_values = np.array([2, 3], dtype=np.int64)
        shape_init = oh.make_tensor(
            "shape", onnxl.TensorProto.INT64, [2], shape_values.tobytes(), raw=True
        )
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [6])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        node = oh.make_node("Reshape", ["X", "shape"], ["Y"])
        graph = oh.make_graph([node], "g", [x], [y], [shape_init])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = os.path.join(model_dir, "model.onnx")
            # Save with size_threshold=0 so even the tiny shape tensor is external.
            save(model, model_path, save_as_external_data=True, size_threshold=0)

            weight_file = model_path + ".data"
            self.assertTrue(os.path.exists(weight_file))

            # fillshape must infer Y's shape correctly despite the shape tensor
            # being stored externally.
            main(["fillshape", model_path])

            # Reload and check that Y's shape is fully resolved.
            result = load(model_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            self.assertEqual(dims[0].dim_value, 2)
            self.assertEqual(dims[1].dim_value, 3)

    def test_fillshape_external_data_output_beside_weights(self):
        """--output with an external-data model places the output beside the weight file."""
        from onnx_light.__main__ import main
        from onnx_light.onnx import save

        values = np.zeros((8, 8), dtype=np.float32)
        init = oh.make_tensor("W", onnxl.TensorProto.FLOAT, [8, 8], values.tobytes(), raw=True)
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [8, 8])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        node = oh.make_node("Add", ["X", "W"], ["Y"])
        graph = oh.make_graph([node], "g", [x], [y], [init])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        with (
            tempfile.TemporaryDirectory() as model_dir,
            tempfile.TemporaryDirectory() as other_dir,
        ):
            model_path = os.path.join(model_dir, "model.onnx")
            save(model, model_path, save_as_external_data=True, size_threshold=0)

            weight_file = model_path + ".data"
            self.assertTrue(os.path.exists(weight_file))

            # Ask for output in a completely different directory.
            output_path = os.path.join(other_dir, "out.onnx")
            main(["fillshape", model_path, "--output", output_path])

            # Output must be placed beside the original model (not in other_dir).
            expected_dest = os.path.join(model_dir, "out.onnx")
            self.assertTrue(os.path.exists(expected_dest))
            self.assertFalse(os.path.exists(output_path))

            # Weight file must NOT have been recreated in other_dir.
            self.assertFalse(
                any(f.endswith(".data") for f in os.listdir(other_dir)),
                "No weight file should have been created in the output directory",
            )

    def test_fillshape_inplace_info_and_shape_tag_combined(self):
        """--inplace-info and --shape-tag together must both write their metadata.

        Regression test for issue #3004: when both flags are used, neither set
        of metadata should be overwritten by the other.  The saved model must
        carry node-level inplace-reuse/release metadata (from --inplace-info)
        AND per-value value_tag + per-node node_tag metadata (from
        --shape-tag).
        """
        from onnx_light.__main__ import main
        from onnx_light.onnx_core.shape_inference import (
            INPLACE_REUSE_METADATA_KEY,
            NODE_TAG_METADATA_KEY,
            RELEASE_AFTER_METADATA_KEY,
            VALUE_TAGS_METADATA_KEY,
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Build Shape(X)->S, Reshape(X, S)->Y, Abs(Y)->Z.
            #  * Shape and Reshape give shape-tag annotations on S and nodes.
            #  * Abs(Y)->Z: Z can reuse Y's buffer (same shape) → inplace reuse.
            shape_node = oh.make_node("Shape", inputs=["X"], outputs=["S"])
            reshape_node = oh.make_node("Reshape", inputs=["X", "S"], outputs=["Y"])
            abs_node = oh.make_node("Abs", inputs=["Y"], outputs=["Z"])
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
            z = oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, None)
            graph = oh.make_graph([shape_node, reshape_node, abs_node], "g", [x], [z])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--inplace-info", "--shape-tag"])

            result = load(model_path)

            # Shapes must be filled.
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)

            # --shape-tag: no feature metadata must be on the graph metadata.
            graph_meta = {entry.key: entry.value for entry in result.graph.metadata_props}
            self.assertNotIn(VALUE_TAGS_METADATA_KEY, graph_meta)

            # --shape-tag: Shape node (node 0) must carry the "shape" node tag.
            node0_meta = {entry.key: entry.value for entry in result.graph.node[0].metadata_props}
            self.assertIn(NODE_TAG_METADATA_KEY, node0_meta)
            self.assertEqual(node0_meta[NODE_TAG_METADATA_KEY], "shape")

            # --inplace-info: Abs node (node 2) must carry inplace/release metadata.
            node2_meta = {entry.key: entry.value for entry in result.graph.node[2].metadata_props}
            self.assertIn(INPLACE_REUSE_METADATA_KEY, node2_meta)
            self.assertIn(RELEASE_AFTER_METADATA_KEY, node2_meta)

    def test_fillshape_show_with_inplace_info_and_shape_tag(self):
        """--show with --inplace-info and --shape-tag must include their annotations.

        Regression test for issue #3004: the --show output must include the
        inplace/release annotations (from --inplace-info) and the node-tag
        prefix (from --shape-tag), not just the bare op-type lines.
        """
        import io
        from contextlib import redirect_stdout

        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Abs(X)->A, Abs(A)->Y: node 1 can reuse A's buffer (inplace).
            abs0 = oh.make_node("Abs", inputs=["X"], outputs=["A"])
            abs1 = oh.make_node("Abs", inputs=["A"], outputs=["Y"])
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
            graph = oh.make_graph([abs0, abs1], "g", [x], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["fillshape", model_path, "--inplace-info", "--shape-tag", "--show"])

            output = buf.getvalue()
            # Inplace annotation must appear in --show output.
            self.assertIn("inplace:", output)
            # Release annotation must appear in --show output.
            self.assertIn("release:", output)

    def test_fillshape_generates_metadata_before_writing_it(self):
        """Verifies that fillshape computes metadata before persisting it."""
        from unittest.mock import patch

        from onnx_light.__main__ import main

        order = []

        class _FakeComputeContext:
            def compute_inplace_reuse_graph(self, graph, ctx, allow_input_overwrite=False):
                del graph, ctx, allow_input_overwrite
                order.append("compute_inplace_reuse_graph")

            def compute_value_and_node_tags(self, graph):
                del graph
                order.append("compute_value_and_node_tags")
                return {}, []

            def write_to_metadata(self, graph):
                del graph
                order.append("write_inplace_reuse_to_metadata")

        def _record_write_value_and_node_tags_to_metadata(graph):
            del graph
            order.append("write_value_and_node_tags_to_metadata")

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_add_model(), model_path)

            with (
                patch("onnx_light.onnx_core.shape_inference.ComputeContext", _FakeComputeContext),
                patch(
                    "onnx_light.onnx_core.shape_inference.write_value_and_node_tags_to_metadata",
                    side_effect=_record_write_value_and_node_tags_to_metadata,
                ),
            ):
                main(["fillshape", model_path, "--inplace-info", "--shape-tag", "--show"])

        self.assertEqual(
            order,
            [
                "compute_inplace_reuse_graph",
                "write_inplace_reuse_to_metadata",
                "compute_value_and_node_tags",
                "write_value_and_node_tags_to_metadata",
            ],
        )


class TestParseTokenSpec(ExtTestCase):
    """Unit tests for the ``_parse_token_spec`` helper."""

    def test_range(self):
        from onnx_light.__main__ import _parse_token_spec

        name, lo, hi = _parse_token_spec("seq=1:128")
        self.assertEqual(name, "seq")
        self.assertEqual(lo, 1)
        self.assertEqual(hi, 128)

    def test_range_equal_bounds(self):
        from onnx_light.__main__ import _parse_token_spec

        name, lo, hi = _parse_token_spec("n=7:7")
        self.assertEqual(name, "n")
        self.assertEqual(lo, 7)
        self.assertEqual(hi, 7)

    def test_missing_equals_raises(self):
        from onnx_light.__main__ import _parse_token_spec

        with self.assertRaisesRegex(ValueError, "NAME=LOW:HIGH"):
            _parse_token_spec("batch4")

    def test_missing_colon_raises(self):
        from onnx_light.__main__ import _parse_token_spec

        with self.assertRaisesRegex(ValueError, "NAME=LOW:HIGH"):
            _parse_token_spec("batch=4")

    def test_inverted_bounds_raises(self):
        from onnx_light.__main__ import _parse_token_spec

        with self.assertRaisesRegex(ValueError, "lower bound.*upper bound"):
            _parse_token_spec("seq=128:1")


class TestFillshapeTokenOption(ExtTestCase):
    """Integration tests for ``fillshape --token``."""

    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    def _save_model(self, model: onnxl.ModelProto, path: str) -> None:
        from onnx_light.onnx import save

        save(model, path)

    def test_token_range_uses_lower_bound_for_inference(self):
        """--token batch=4:4 uses lower bound 4 for shape propagation → concrete output shape."""
        from onnx_light.__main__ import main

        model = _make_add_model(input_shape=["batch", 3])
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--token", "batch=4:4"])

            result = load(model_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            self.assertEqual(dims[0].dim_value, 4)
            self.assertEqual(dims[1].dim_value, 3)

    def test_token_range_asymmetric(self):
        """--token seq=1:128 uses lower bound 1 for shape propagation."""
        from onnx_light.__main__ import main

        model = _make_add_model(input_shape=["seq", 8])
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--token", "seq=1:128"])

            result = load(model_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            self.assertEqual(dims[0].dim_value, 1)
            self.assertEqual(dims[1].dim_value, 8)

    def test_multiple_tokens(self):
        """Multiple --token args each substitute their respective dim."""
        from onnx_light.__main__ import main

        # Two-input Add: both symbolic, both overridden.
        add = oh.make_node("Add", inputs=["X", "X"], outputs=["Y"])
        x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["batch", "seq"])
        y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
        graph = oh.make_graph([add], "g", [x], [y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        model.ir_version = 8

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--token", "batch=4:4", "--token", "seq=16:16"])

            result = load(model_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            self.assertEqual(dims[0].dim_value, 4)
            self.assertEqual(dims[1].dim_value, 16)

    def test_uncovered_symbolic_dim_remains_symbolic(self):
        """Dims not covered by --token stay symbolic in the inferred output."""
        from onnx_light.__main__ import main

        model = _make_add_model(input_shape=["batch", "seq"])
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(model, model_path)

            # Only override 'batch'; 'seq' should remain symbolic.
            main(["fillshape", model_path, "--token", "batch=2:2"])

            result = load(model_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            self.assertEqual(dims[0].dim_value, 2)
            # 'seq' was not overridden → still symbolic.
            self.assertTrue(dims[1].has_dim_param())
            self.assertEqual(dims[1].dim_param, "seq")

    def test_token_with_output_option(self):
        """--token works together with --output."""
        from onnx_light.__main__ import main

        model = _make_add_model(input_shape=["batch", 5])
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            output_path = os.path.join(tmp, "out.onnx")
            self._save_model(model, model_path)

            main(["fillshape", model_path, "--output", output_path, "--token", "batch=8:8"])

            result = load(output_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            self.assertEqual(dims[0].dim_value, 8)
            self.assertEqual(dims[1].dim_value, 5)


if __name__ == "__main__":
    unittest.main()
