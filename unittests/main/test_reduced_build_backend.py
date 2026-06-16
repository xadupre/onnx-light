# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import sys
import unittest
from pathlib import Path

from onnx_light._reduced_build import ReducedBuildError, kernels_required
from onnx_light.ext_test_case import ExtTestCase

_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _ROOT / "_build_backend"


def _load_backend():
    """Imports the in-tree onnx_light_build backend module."""
    sys.path.insert(0, str(_BACKEND_DIR))
    try:
        module = importlib.import_module("onnx_light_build")
        return importlib.reload(module)
    finally:
        sys.path.pop(0)


class TestReducedBuild(ExtTestCase):
    def setUp(self):
        self._saved = os.environ.pop("ONNX_LIGHT_REDUCED", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("ONNX_LIGHT_REDUCED", None)
        else:
            os.environ["ONNX_LIGHT_REDUCED"] = self._saved

    def test_kernels_required_raises(self):
        """Tests that kernels_required raises an explicit ReducedBuildError."""
        with self.assertRaises(ReducedBuildError) as ctx:
            kernels_required("some feature")
        self.assertIn("some feature", str(ctx.exception))
        self.assertIn("ONNX_LIGHT_BUILD_KERNELS=OFF", str(ctx.exception))
        self.assertIsInstance(ctx.exception, ImportError)

    def test_is_reduced_build_env(self):
        """Tests that is_reduced_build reflects the ONNX_LIGHT_REDUCED env var."""
        backend = _load_backend()
        os.environ.pop("ONNX_LIGHT_REDUCED", None)
        self.assertFalse(backend.is_reduced_build())
        for value in ("1", "on", "TRUE", "yes"):
            os.environ["ONNX_LIGHT_REDUCED"] = value
            self.assertTrue(backend.is_reduced_build(), msg=value)
        os.environ["ONNX_LIGHT_REDUCED"] = "0"
        self.assertFalse(backend.is_reduced_build())

    def test_augment_config_settings_full(self):
        """Tests that the CMake define is not injected for a full build."""
        backend = _load_backend()
        os.environ.pop("ONNX_LIGHT_REDUCED", None)
        self.assertIsNone(backend._augment_config_settings(None))
        self.assertEqual(backend._augment_config_settings({"a": "b"}), {"a": "b"})

    def test_augment_config_settings_reduced(self):
        """Tests that the CMake define is injected for a reduced build."""
        backend = _load_backend()
        os.environ["ONNX_LIGHT_REDUCED"] = "1"
        settings = backend._augment_config_settings({"a": "b"})
        self.assertEqual(settings["a"], "b")
        self.assertEqual(settings["cmake.define.ONNX_LIGHT_BUILD_KERNELS"], "OFF")

    def test_augment_preserves_explicit_define(self):
        """Tests that an explicit kernels define is not overridden."""
        backend = _load_backend()
        os.environ["ONNX_LIGHT_REDUCED"] = "1"
        settings = backend._augment_config_settings(
            {"cmake.define.ONNX_LIGHT_BUILD_KERNELS": "ON"}
        )
        self.assertEqual(settings["cmake.define.ONNX_LIGHT_BUILD_KERNELS"], "ON")

    def test_reduced_distribution_name_restores_pyproject(self):
        """Tests that the reduced build renames and then restores pyproject.toml."""
        backend = _load_backend()
        pyproject = _ROOT / "pyproject.toml"
        original = pyproject.read_text(encoding="utf-8")
        os.environ["ONNX_LIGHT_REDUCED"] = "1"
        with backend._reduced_distribution_name():
            inside = pyproject.read_text(encoding="utf-8")
            self.assertIn(f'name = "{backend.REDUCED_DISTRIBUTION_NAME}"', inside)
            self.assertNotIn(f'name = "{backend.FULL_DISTRIBUTION_NAME}"', inside)
        self.assertEqual(pyproject.read_text(encoding="utf-8"), original)

    def test_reduced_distribution_name_noop_for_full(self):
        """Tests that a full build leaves pyproject.toml unchanged."""
        backend = _load_backend()
        pyproject = _ROOT / "pyproject.toml"
        original = pyproject.read_text(encoding="utf-8")
        os.environ.pop("ONNX_LIGHT_REDUCED", None)
        with backend._reduced_distribution_name():
            self.assertEqual(pyproject.read_text(encoding="utf-8"), original)
        self.assertEqual(pyproject.read_text(encoding="utf-8"), original)

    def test_backend_exposes_pep517_hooks(self):
        """Tests that the in-tree backend exposes the standard PEP 517 hooks."""
        backend = _load_backend()
        for hook in (
            "build_wheel",
            "build_sdist",
            "build_editable",
            "get_requires_for_build_wheel",
            "get_requires_for_build_sdist",
            "get_requires_for_build_editable",
            "prepare_metadata_for_build_wheel",
            "prepare_metadata_for_build_editable",
        ):
            self.assertTrue(callable(getattr(backend, hook)), msg=hook)


if __name__ == "__main__":
    unittest.main(verbosity=2)
