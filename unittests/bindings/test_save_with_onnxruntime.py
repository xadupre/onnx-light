"""Round-trip tests: saving an ``onnx_light`` ModelProto with various
combinations of ``alignment`` and ``max_external_file_size`` (single file,
single external weights file, multiple external weights files), then loading
and running it with ``onnxruntime``.

These tests verify that the on-disk layout produced by ``onnx_light`` is
interoperable with ``onnxruntime``'s model loader for every combination of
``alignment`` and ``max_external_file_size`` options exposed by
``SerializeOptions``.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.ext_test_case import ExtTestCase


def _make_model_with_initializers() -> (
    tuple[onnxl.ModelProto, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Builds a tiny ``MatMul + Add`` model with two raw-data initializers.

    The two initializers are intentionally identical in size so that a small
    ``max_external_file_size`` can split them into separate weights files.
    """
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 8)).astype(np.float32)
    b = rng.standard_normal((8,)).astype(np.float32)

    tfloat = oh.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("MatMul", ["X", "W"], ["XW"]),
                oh.make_node("Add", ["XW", "B"], ["Y"]),
            ],
            "g",
            [oh.make_tensor_value_info("X", tfloat, [None, 16])],
            [oh.make_tensor_value_info("Y", tfloat, [None, 8])],
            [onh.from_array(w, name="W"), onh.from_array(b, name="B")],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )
    x = rng.standard_normal((3, 16)).astype(np.float32)
    expected = x @ w + b
    return model, x, w, b, expected


class TestSaveAndRunWithOnnxRuntime(ExtTestCase):
    """Verifies models saved by ``onnx_light.save`` load and run in ORT."""

    def _check_with_ort(self, model_path: str, x: np.ndarray, expected: np.ndarray) -> None:
        import onnxruntime as ort

        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        (got,) = sess.run(None, {"X": x})
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    # ------------------------------------------------------------------ #
    # Single file (no external data).
    # ------------------------------------------------------------------ #

    def test_single_file_no_alignment(self) -> None:
        model, x, _, _, expected = _make_model_with_initializers()
        name = self.get_dump_file("test_ort_single_file_no_alignment.onnx")
        onnxl.save(model, name)
        self._check_with_ort(name, x, expected)

    def test_single_file_with_alignment(self) -> None:
        # Alignment is not exposed through onnx_light.onnx.save, so go through
        # SerializeToFile + SerializeOptions to exercise the aligned-inline
        # raw-data writer.
        model, x, _, _, expected = _make_model_with_initializers()
        name = self.get_dump_file("test_ort_single_file_alignment.onnx")
        opts = onnxl.SerializeOptions()
        opts.alignment = 64
        model.SerializeToFile(name, opts)
        self._check_with_ort(name, x, expected)

    # ------------------------------------------------------------------ #
    # External data — single weights file.
    # ------------------------------------------------------------------ #

    def test_external_data_single_file_no_alignment(self) -> None:
        model, x, _, _, expected = _make_model_with_initializers()
        name = self.get_dump_file("test_ort_external_single_no_alignment.onnx")
        onnxl.save(model, name, save_as_external_data=True, size_threshold=0)
        self.assertTrue(os.path.exists(name + ".data"))
        self._check_with_ort(name, x, expected)

    def test_external_data_single_file_with_alignment(self) -> None:
        model, x, _, _, expected = _make_model_with_initializers()
        name = self.get_dump_file("test_ort_external_single_alignment.onnx")
        location = name + ".data"
        opts = onnxl.SerializeOptions()
        opts.raw_data_threshold = 0
        opts.alignment = 64
        model.SerializeToFile(name, opts, location)
        self.assertTrue(os.path.exists(location))
        self._check_with_ort(name, x, expected)

    # ------------------------------------------------------------------ #
    # External data — multiple weights files.
    # ------------------------------------------------------------------ #

    def _assert_multiple_external_files(self, prefix: str) -> None:
        """Asserts that at least two external weights files were produced."""
        directory = os.path.dirname(prefix) or "."
        basename = os.path.basename(prefix)
        produced = sorted(
            f for f in os.listdir(directory) if f.startswith(basename)
        )
        self.assertGreaterEqual(
            len(produced),
            2,
            f"Expected at least two external weights files with prefix "
            f"{basename!r}, got {produced}.",
        )

    def test_external_data_multiple_files_no_alignment(self) -> None:
        model, x, w, b, expected = _make_model_with_initializers()
        name = self.get_dump_file("test_ort_external_multi_no_alignment.onnx")
        location = name + ".data"
        # Cap below the size of the larger initializer so that the writer
        # starts a second external weights file.
        max_size = max(w.nbytes, b.nbytes) - 1
        onnxl.save(
            model,
            name,
            location=location,
            save_as_external_data=True,
            size_threshold=0,
            max_external_file_size=max_size,
        )
        self._assert_multiple_external_files(location)
        self._check_with_ort(name, x, expected)

    def test_external_data_multiple_files_with_alignment(self) -> None:
        model, x, w, b, expected = _make_model_with_initializers()
        name = self.get_dump_file("test_ort_external_multi_alignment.onnx")
        location = name + ".data"
        opts = onnxl.SerializeOptions()
        opts.raw_data_threshold = 0
        opts.alignment = 64
        opts.max_external_file_size = max(w.nbytes, b.nbytes) - 1
        model.SerializeToFile(name, opts, location)
        self._assert_multiple_external_files(location)
        self._check_with_ort(name, x, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
