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

            # Node 1 (Abs A->Y) should have inplace_reuse metadata.
            node1_meta = {entry.key: entry.value for entry in result.graph.node[1].metadata_props}
            self.assertIn("onnx_light.inplace_reuse", node1_meta)

    def test_fillshape_missing_file_raises(self):
        """fillshape raises when the model file does not exist."""
        from onnx_light.__main__ import main

        with self.assertRaises(OSError):
            main(["fillshape", "/nonexistent/path/model.onnx"])

    def test_fillshape_external_data_not_loaded(self):
        """fillshape loads the model without fetching external weight bytes."""
        from onnx_light.__main__ import main
        from onnx_light.onnx import save
        from onnx_light.onnx_lib.external_data_helper import uses_external_data

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

            # The output model should still reference external data.
            result = load(model_path, load_external_data=False)
            self.assertTrue(any(uses_external_data(i) for i in result.graph.initializer))

            # Shapes must be filled.
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)

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


if __name__ == "__main__":
    unittest.main()
