# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``python -m onnx_light fillshape`` (``onnx_light.__main__``)."""

from __future__ import annotations

import os
import tempfile
import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import defs, load


def _make_add_model() -> onnxl.ModelProto:
    """Builds ``Y = Add(X, X)`` with unresolved output shape."""
    add = oh.make_node("Add", inputs=["X", "X"], outputs=["Y"])
    x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
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

            ret = main(["fillshape", model_path])
            self.assertEqual(ret, 0)

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

            ret = main(["fillshape", model_path, "--output", output_path])
            self.assertEqual(ret, 0)

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
                ret = main(["fillshape", model_path, "--show"])
            self.assertEqual(ret, 0)

            output = buf.getvalue()
            self.assertIn("Add", output)

            # File must not have been modified.
            self.assertEqual(os.path.getmtime(model_path), mtime_before)

    def test_fillshape_keep_option(self):
        """fillshape --keep uses existing shapes as anchors."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            model = _make_add_model()
            # Pre-set the output with a symbolic anchor dim.
            model.graph.output[0].CopyFrom(
                oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, ["ANCHOR", 3])
            )
            self._save_model(model, model_path)

            ret = main(["fillshape", model_path, "--keep"])
            self.assertEqual(ret, 0)

            result = load(model_path)
            dims = list(result.graph.output[0].type.tensor_type.shape.dim)
            self.assertEqual(len(dims), 2)
            # Anchor should be preserved by --keep.
            self.assertTrue(dims[0].has_dim_param())
            self.assertEqual(dims[0].dim_param, "ANCHOR")

    def test_fillshape_invalid_model(self):
        """fillshape returns non-zero when the model file does not exist."""
        from onnx_light.__main__ import main

        ret = main(["fillshape", "/nonexistent/path/model.onnx"])
        self.assertNotEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()
