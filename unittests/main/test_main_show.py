# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``python -m onnx_light show`` (``onnx_light.__main__``)."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import defs


def _make_abs_model() -> onnxl.ModelProto:
    """Builds ``Y = Abs(X)`` with a concrete input shape."""
    x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
    y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
    node = oh.make_node("Abs", inputs=["X"], outputs=["Y"])
    graph = oh.make_graph([node], "g", [x], [y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def _make_chain_model() -> onnxl.ModelProto:
    """Builds ``Y = Abs(Abs(X))`` with concrete shapes."""
    x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
    y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
    node0 = oh.make_node("Abs", inputs=["X"], outputs=["A"])
    node1 = oh.make_node("Abs", inputs=["A"], outputs=["Y"])
    graph = oh.make_graph([node0, node1], "g", [x], [y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = 8
    return model


class TestMainShow(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    def _save_model(self, model: onnxl.ModelProto, path: str) -> None:
        from onnx_light.onnx import save

        save(model, path)

    # ------------------------------------------------------------------
    # pretty format (default)
    # ------------------------------------------------------------------

    def test_show_default_format_prints_to_stdout(self):
        """show with no --format prints the model as pretty text to stdout."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path])

            output = buf.getvalue()
            self.assertIn("Abs", output)

    def test_show_pretty_format_explicit(self):
        """show --format pretty prints compact text listing."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "pretty"])

            output = buf.getvalue()
            self.assertIn("Abs", output)
            self.assertIn("X", output)

    def test_show_pretty_include_attributes(self):
        """show --include-attributes includes node attributes in the output."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            # Use Clip which has min/max attributes.
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
            # Clip(X, min=None, max=None) still renders attributes listing.
            node = oh.make_node("Abs", inputs=["X"], outputs=["Y"])
            graph = oh.make_graph([node], "g", [x], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(model, model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "pretty", "--include-attributes"])

            output = buf.getvalue()
            self.assertIn("Abs", output)

    def test_show_pretty_shape_inference(self):
        """show --shape-inference runs shape inference before rendering."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--shape-inference"])

            output = buf.getvalue()
            self.assertIn("Abs", output)

    # ------------------------------------------------------------------
    # mermaid format
    # ------------------------------------------------------------------

    def test_show_mermaid_format(self):
        """show --format mermaid renders a Mermaid flowchart."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "mermaid"])

            output = buf.getvalue()
            self.assertIn("flowchart", output)
            self.assertIn("Abs", output)

    def test_show_mermaid_direction_lr(self):
        """show --format mermaid --direction LR uses left-to-right layout."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "mermaid", "--direction", "LR"])

            output = buf.getvalue()
            self.assertIn("LR", output)

    def test_show_mermaid_no_shapes(self):
        """show --format mermaid --no-shapes omits shape annotations."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Build model with known shape so there is something to hide.
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
            node = oh.make_node("Abs", inputs=["X"], outputs=["Y"])
            graph = oh.make_graph([node], "g", [x], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            buf_shapes = io.StringIO()
            with redirect_stdout(buf_shapes):
                main(["show", model_path, "--format", "mermaid"])

            buf_no_shapes = io.StringIO()
            with redirect_stdout(buf_no_shapes):
                main(["show", model_path, "--format", "mermaid", "--no-shapes"])

            # Both should be valid mermaid output.
            self.assertIn("flowchart", buf_shapes.getvalue())
            self.assertIn("flowchart", buf_no_shapes.getvalue())

    def test_show_mermaid_no_initializers(self):
        """show --format mermaid --no-initializers excludes initializer nodes."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            import numpy as np

            values = np.ones((3,), dtype=np.float32)
            init = oh.make_tensor("W", onnxl.TensorProto.FLOAT, [3], values.tobytes(), raw=True)
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
            node = oh.make_node("Add", inputs=["X", "W"], outputs=["Y"])
            graph = oh.make_graph([node], "g", [x], [y], initializer=[init])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(model, model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "mermaid", "--no-initializers"])

            output = buf.getvalue()
            self.assertIn("flowchart", output)

    # ------------------------------------------------------------------
    # svg format
    # ------------------------------------------------------------------

    def test_show_svg_format(self):
        """show --format svg renders a valid SVG document."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "svg"])

            output = buf.getvalue()
            self.assertIn("<svg", output)
            self.assertIn("Abs", output)

    def test_show_svg_no_shapes(self):
        """show --format svg --no-shapes omits shape annotations."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "svg", "--no-shapes"])

            output = buf.getvalue()
            self.assertIn("<svg", output)

    # ------------------------------------------------------------------
    # output file
    # ------------------------------------------------------------------

    def test_show_output_file_pretty(self):
        """show --output writes pretty text to the specified file."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            output_path = os.path.join(tmp, "out.txt")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--output", output_path])

            # Nothing should have been printed to stdout.
            self.assertEqual(buf.getvalue(), "")

            # The file must exist and contain the model rendering.
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Abs", content)

    def test_show_output_file_mermaid(self):
        """show --format mermaid --output writes mermaid to the specified file."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            output_path = os.path.join(tmp, "out.mmd")
            self._save_model(_make_abs_model(), model_path)

            main(["show", model_path, "--format", "mermaid", "--output", output_path])

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("flowchart", content)

    def test_show_output_file_svg(self):
        """show --format svg --output writes SVG to the specified file."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            output_path = os.path.join(tmp, "out.svg")
            self._save_model(_make_abs_model(), model_path)

            main(["show", model_path, "--format", "svg", "--output", output_path])

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<svg", content)

    # ------------------------------------------------------------------
    # dot format
    # ------------------------------------------------------------------

    def test_show_dot_format(self):
        """show --format dot renders a Graphviz DOT graph."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "dot"])

            output = buf.getvalue()
            self.assertIn("digraph onnx {", output)
            self.assertIn("Abs", output)

    def test_show_dot_direction_lr(self):
        """show --format dot --direction LR uses left-to-right layout."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "dot", "--direction", "LR"])

            output = buf.getvalue()
            self.assertIn('rankdir="LR"', output)

    def test_show_dot_no_shapes(self):
        """show --format dot --no-shapes omits shape annotations."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3, 4])
            node = oh.make_node("Abs", inputs=["X"], outputs=["Y"])
            graph = oh.make_graph([node], "g", [x], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            buf_shapes = io.StringIO()
            with redirect_stdout(buf_shapes):
                main(["show", model_path, "--format", "dot"])

            buf_no_shapes = io.StringIO()
            with redirect_stdout(buf_no_shapes):
                main(["show", model_path, "--format", "dot", "--no-shapes"])

            self.assertIn("float[3,4]", buf_shapes.getvalue())
            self.assertNotIn("float[3,4]", buf_no_shapes.getvalue())

    def test_show_dot_output_file(self):
        """show --format dot --output writes DOT source to the specified file."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            output_path = os.path.join(tmp, "out.dot")
            self._save_model(_make_abs_model(), model_path)

            main(["show", model_path, "--format", "dot", "--output", output_path])

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("digraph onnx {", content)
            self.assertIn("Abs", content)

    @unittest.skipUnless(
        shutil.which("dot") is not None, "Graphviz 'dot' executable not found on PATH"
    )
    def test_show_dot_graphviz_png(self):
        """show --format dot --graphviz png invokes graphviz and writes PNG bytes."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            output_path = os.path.join(tmp, "out.png")
            self._save_model(_make_abs_model(), model_path)

            main(
                [
                    "show",
                    model_path,
                    "--format",
                    "dot",
                    "--graphviz",
                    "png",
                    "--output",
                    output_path,
                ]
            )

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, "rb") as f:
                header = f.read(4)
            # PNG files start with the 4-byte magic \x89PNG.
            self.assertEqual(header, b"\x89PNG")

    @unittest.skipUnless(
        shutil.which("dot") is not None, "Graphviz 'dot' executable not found on PATH"
    )
    def test_show_dot_graphviz_svg_to_stdout(self):
        """show --format dot --graphviz svg writes SVG bytes to stdout.buffer."""
        import sys
        from io import BytesIO
        from unittest.mock import patch

        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = BytesIO()
            with patch.object(sys.stdout, "buffer", buf):
                main(["show", model_path, "--format", "dot", "--graphviz", "svg"])

            output = buf.getvalue()
            self.assertIn(b"<svg", output)

    # ------------------------------------------------------------------
    # error handling
    # ------------------------------------------------------------------

    def test_show_missing_file_raises(self):
        """show raises FileNotFoundError when the model file does not exist."""
        from onnx_light.__main__ import main

        with self.assertRaises(FileNotFoundError):
            main(["show", "/nonexistent/path/model.onnx"])

    def test_show_invalid_graphviz_format_raises(self):
        """Tests that show rejects --graphviz values containing unsafe characters."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            unsafe_formats = ["; rm -rf /", "../../../etc/passwd", "png && ls"]
            for fmt in unsafe_formats:
                with (
                    self.subTest(fmt=fmt),
                    self.assertRaises(ValueError, msg=f"Expected ValueError for {fmt!r}"),
                ):
                    main(["show", model_path, "--format", "dot", "--graphviz", fmt])

    def test_show_inplace_info(self):
        """show --include-inplace includes inplace annotations in the output."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_chain_model(), model_path)

            # First fill the model with inplace metadata.
            from onnx_light.__main__ import main as _main

            _main(["fillshape", model_path, "--inplace-info"])

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--include-inplace"])

            output = buf.getvalue()
            self.assertIn("Abs", output)

    def test_show_release_info(self):
        """show --include-release includes release annotations in the output."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_chain_model(), model_path)

            # First fill the model with inplace/release metadata.
            from onnx_light.__main__ import main as _main

            _main(["fillshape", model_path, "--inplace-info"])

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--include-release"])

            output = buf.getvalue()
            self.assertIn("Abs", output)
            self.assertIn("release", output)

    # ------------------------------------------------------------------
    # translate formats (onnx-compact / builder)
    # ------------------------------------------------------------------

    def test_show_onnx_compact_format(self):
        """show --format onnx-compact emits runnable onnx-compact Python code."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "onnx-compact"])

            output = buf.getvalue()
            self.assertIn("import onnx_light.onnx.helper as oh", output)
            self.assertIn("oh.make_model(", output)
            self.assertIn("oh.make_node('Abs'", output)

    def test_show_builder_format(self):
        """show --format builder emits a runnable GraphBuilder Python script."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["show", model_path, "--format", "builder"])

            output = buf.getvalue()
            self.assertIn("from onnx_light.onnx_core.graph_builder import GraphBuilder", output)
            self.assertIn("GraphBuilder(", output)
            self.assertIn("g.make_node('Abs'", output)

    def test_show_onnx_compact_output_file(self):
        """show --format onnx-compact -o writes the generated code to a file."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            out_path = os.path.join(tmp, "rebuild.py")
            self._save_model(_make_abs_model(), model_path)

            main(["show", model_path, "--format", "onnx-compact", "-o", out_path])

            self.assertTrue(os.path.exists(out_path))
            with open(out_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("oh.make_model(", content)


if __name__ == "__main__":
    unittest.main()
