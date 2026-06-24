# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``python -m onnx_light run`` (``onnx_light.__main__``)."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import defs, save


def _make_abs_model(input_shape=None) -> onnxl.ModelProto:
    """Builds ``Y = Abs(X)`` with the given input shape.

    Args:
        input_shape: Shape for input X. Defaults to ``[2, 3]``. Each entry
            may be an ``int`` (concrete dim) or a ``str`` (symbolic dim_param).

    Returns:
        A simple single-op ONNX model.
    """
    if input_shape is None:
        input_shape = [2, 3]
    node = oh.make_node("Abs", inputs=["X"], outputs=["Y"])
    x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, input_shape)
    y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
    graph = oh.make_graph([node], "g", [x], [y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def _make_int_abs_model() -> onnxl.ModelProto:
    """Builds ``Y = Abs(X)`` with INT64 input and concrete shape ``[3]``.

    Returns:
        A single-op ONNX model with integer input.
    """
    node = oh.make_node("Abs", inputs=["X"], outputs=["Y"])
    x = oh.make_tensor_value_info("X", onnxl.TensorProto.INT64, [3])
    y = oh.make_tensor_value_info("Y", onnxl.TensorProto.INT64, None)
    graph = oh.make_graph([node], "g", [x], [y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = 8
    return model


class TestMainRun(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    def _save_model(self, model: onnxl.ModelProto, path: str) -> None:
        save(model, path)

    def test_run_basic(self):
        """run produces output for a model with concrete input shapes."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["run", model_path])

            output = buf.getvalue()
            self.assertIn("output 'Y'", output)

    def test_run_verbose(self):
        """run --verbose prints input/output details."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["run", model_path, "--verbose"])

            output = buf.getvalue()
            self.assertIn("Model:", output)
            self.assertIn("input 'X'", output)
            self.assertIn("output 'Y'", output)
            self.assertIn("values:", output)

    def test_run_verbose_short_flag(self):
        """run -v is equivalent to run --verbose."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["run", model_path, "-v"])

            output = buf.getvalue()
            self.assertIn("Model:", output)

    def test_run_dim_override(self):
        """run --dim resolves a symbolic dimension."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(input_shape=["batch", 3]), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["run", model_path, "--dim", "batch=4"])

            output = buf.getvalue()
            # Output shape should include the resolved batch dimension.
            self.assertIn("[4, 3]", output)

    def test_run_dynamic_dim_defaults_to_one(self):
        """Unspecified symbolic dims default to 1."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(input_shape=["batch", 3]), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["run", model_path])

            output = buf.getvalue()
            self.assertIn("[1, 3]", output)

    def test_run_integer_input(self):
        """run works for models with integer inputs."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_int_abs_model(), model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["run", model_path])

            output = buf.getvalue()
            self.assertIn("output 'Y'", output)

    def test_run_seed_determinism(self):
        """Two runs with the same seed produce identical outputs."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                main(["run", model_path, "--seed", "42", "--verbose"])

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                main(["run", model_path, "--seed", "42", "--verbose"])

            self.assertEqual(buf1.getvalue(), buf2.getvalue())

    def test_run_different_seeds_differ(self):
        """Two runs with different seeds produce different outputs."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                main(["run", model_path, "--seed", "1", "--verbose"])

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                main(["run", model_path, "--seed", "2", "--verbose"])

            self.assertNotEqual(buf1.getvalue(), buf2.getvalue())

    def test_run_missing_file_raises(self):
        """run raises when the model file does not exist."""
        from onnx_light.__main__ import main

        with self.assertRaises(FileNotFoundError):
            main(["run", "/nonexistent/path/model.onnx"])

    def test_run_invalid_dim_format_raises(self):
        """run raises ValueError for malformed --dim arguments."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            self._save_model(_make_abs_model(), model_path)

            with self.assertRaises(ValueError):
                main(["run", model_path, "--dim", "batch4"])

    def test_run_multiple_dim_overrides(self):
        """run handles multiple --dim arguments."""
        from onnx_light.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.onnx")
            # Two-input model: both symbolic.
            node = oh.make_node("Add", inputs=["X", "Z"], outputs=["Y"])
            x = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["batch", "seq"])
            z = oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, ["batch", "seq"])
            y = oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)
            graph = oh.make_graph([node], "g", [x, z], [y])
            model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
            model.ir_version = 8
            self._save_model(model, model_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["run", model_path, "--dim", "batch=2", "--dim", "seq=5"])

            output = buf.getvalue()
            self.assertIn("[2, 5]", output)


class TestMakeRandomInput(ExtTestCase):
    """Unit tests for ``_make_random_input``."""

    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    def test_float_input(self):
        """Float input is in [0, 1) range."""
        from onnx_light.__main__ import _make_random_input

        arr = _make_random_input(int(onnxl.TensorProto.FLOAT), [4, 4], seed=0)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr.shape, (4, 4))

    def test_int64_input(self):
        """INT64 input has correct dtype and values in [0, 10)."""
        from onnx_light.__main__ import _make_random_input

        arr = _make_random_input(int(onnxl.TensorProto.INT64), [3], seed=0)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.int64)
        self.assertTrue(np.all(arr >= 0))
        self.assertTrue(np.all(arr < 10))

    def test_bool_input(self):
        """BOOL input has bool dtype."""
        from onnx_light.__main__ import _make_random_input

        arr = _make_random_input(int(onnxl.TensorProto.BOOL), [5], seed=0)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.bool_)

    def test_double_input(self):
        """DOUBLE input has float64 dtype."""
        from onnx_light.__main__ import _make_random_input

        arr = _make_random_input(int(onnxl.TensorProto.DOUBLE), [2, 3], seed=0)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float64)

    def test_string_input_raises(self):
        """STRING inputs raise NotImplementedError."""
        from onnx_light.__main__ import _make_random_input

        with self.assertRaises(NotImplementedError):
            _make_random_input(int(onnxl.TensorProto.STRING), [3], seed=0)


class TestResolveInputShape(ExtTestCase):
    """Unit tests for ``_resolve_input_shape``."""

    @classmethod
    def setUpClass(cls):
        defs.register_onnx_operator_set_schema()

    def test_concrete_shape(self):
        """Concrete dims are returned unchanged."""
        from onnx_light.__main__ import _resolve_input_shape

        vi = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 3])
        shape = _resolve_input_shape(vi.type, {}, "X")
        self.assertEqual(shape, [2, 3])

    def test_symbolic_dim_with_override(self):
        """Symbolic dim is resolved from the dim_overrides dict."""
        from onnx_light.__main__ import _resolve_input_shape

        vi = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["batch", 3])
        shape = _resolve_input_shape(vi.type, {"batch": 4}, "X")
        self.assertEqual(shape, [4, 3])

    def test_symbolic_dim_defaults_to_one(self):
        """Unresolved symbolic dim falls back to 1."""
        from onnx_light.__main__ import _resolve_input_shape

        vi = oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["batch", 3])
        shape = _resolve_input_shape(vi.type, {}, "X")
        self.assertEqual(shape, [1, 3])

    def test_non_tensor_type_raises(self):
        """Non-tensor types raise ValueError."""
        from onnx_light.__main__ import _resolve_input_shape

        # A sequence-typed value info has no tensor_type.
        vi = oh.make_tensor_sequence_value_info("X", onnxl.TensorProto.FLOAT, None)
        with self.assertRaises(ValueError):
            _resolve_input_shape(vi.type, {}, "X")


if __name__ == "__main__":
    unittest.main()
