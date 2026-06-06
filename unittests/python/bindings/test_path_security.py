"""Tests for path security utilities (_path_security module).

Covers path traversal detection, symlink escape detection, and canonical
containment validation for external data paths.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

# Import _path_security directly to avoid pulling in compiled C++ extensions.
_spec = importlib.util.spec_from_file_location(
    "onnx_light.onnx_lib._path_security",
    os.path.join(
        os.path.dirname(__file__), "..", "..", "onnx_light", "onnx_lib", "_path_security.py"
    ),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_is_relative_and_contained = _mod._is_relative_and_contained
validate_external_data_path = _mod.validate_external_data_path


class TestIsRelativeAndContained(unittest.TestCase):
    """Tests for _is_relative_and_contained."""

    def test_simple_relative(self):
        self.assertTrue(_is_relative_and_contained("weights.bin"))

    def test_relative_subdirectory(self):
        self.assertTrue(_is_relative_and_contained("subdir/weights.bin"))

    def test_empty_string(self):
        self.assertFalse(_is_relative_and_contained(""))

    def test_parent_traversal(self):
        self.assertFalse(_is_relative_and_contained("../outside.bin"))

    def test_deep_parent_traversal(self):
        self.assertFalse(_is_relative_and_contained("../../etc/passwd"))

    def test_nested_traversal(self):
        # safe/../../outside resolves to ../outside
        self.assertFalse(_is_relative_and_contained("safe/../../outside"))

    def test_absolute_posix(self):
        self.assertFalse(_is_relative_and_contained("/etc/passwd"))

    @unittest.skipUnless(sys.platform == "win32", "Windows-specific test")
    def test_absolute_windows(self):
        self.assertFalse(_is_relative_and_contained("C:\\Windows\\System32\\config"))

    def test_dot_only(self):
        # "." is valid (refers to base_dir itself)
        self.assertTrue(_is_relative_and_contained("./weights.bin"))


class TestValidateExternalDataPath(unittest.TestCase):
    """Tests for validate_external_data_path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a valid weights file
        self.weights_file = os.path.join(self.tmpdir, "weights.bin")
        with open(self.weights_file, "wb") as f:
            f.write(b"\x00" * 16)
        # Create a subdirectory with a file
        self.subdir = os.path.join(self.tmpdir, "data")
        os.makedirs(self.subdir, exist_ok=True)
        self.sub_weights = os.path.join(self.subdir, "model.data")
        with open(self.sub_weights, "wb") as f:
            f.write(b"\x01" * 16)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_simple_location(self):
        result = validate_external_data_path("weights.bin", self.tmpdir)
        self.assertEqual(result, os.path.realpath(self.weights_file))

    def test_valid_subdirectory_location(self):
        result = validate_external_data_path("data/model.data", self.tmpdir)
        self.assertEqual(result, os.path.realpath(self.sub_weights))

    def test_rejects_parent_traversal(self):
        with self.assertRaises(ValueError) as ctx:
            validate_external_data_path("../../../etc/passwd", self.tmpdir)
        self.assertIn("relative path", str(ctx.exception))

    def test_rejects_absolute_by_default(self):
        with self.assertRaises(ValueError) as ctx:
            validate_external_data_path("/etc/passwd", self.tmpdir)
        self.assertIn("relative path", str(ctx.exception))

    def test_rejects_empty_location(self):
        with self.assertRaises(ValueError):
            validate_external_data_path("", self.tmpdir)

    def test_allows_absolute_with_flag(self):
        # Allow absolute but still check containment
        result = validate_external_data_path(self.weights_file, self.tmpdir, allow_absolute=True)
        self.assertEqual(result, os.path.realpath(self.weights_file))

    def test_rejects_absolute_outside_base(self):
        outside = tempfile.mktemp(suffix=".bin")
        try:
            with open(outside, "wb") as f:
                f.write(b"\x00")
            with self.assertRaises(ValueError) as ctx:
                validate_external_data_path(outside, self.tmpdir, allow_absolute=True)
            self.assertIn("outside", str(ctx.exception))
        finally:
            if os.path.exists(outside):
                os.unlink(outside)

    @unittest.skipUnless(
        hasattr(os, "symlink") and sys.platform != "win32",
        "Symlinks require POSIX or elevated privileges on Windows",
    )
    def test_rejects_symlink_escape(self):
        # Create a symlink inside tmpdir that points outside
        outside_dir = tempfile.mkdtemp()
        outside_file = os.path.join(outside_dir, "secret.bin")
        with open(outside_file, "wb") as f:
            f.write(b"secret")
        link_path = os.path.join(self.tmpdir, "escape_link")
        os.symlink(outside_dir, link_path)
        try:
            with self.assertRaises(ValueError) as ctx:
                validate_external_data_path("escape_link/secret.bin", self.tmpdir)
            self.assertIn("outside", str(ctx.exception))
        finally:
            os.unlink(link_path)
            import shutil

            shutil.rmtree(outside_dir, ignore_errors=True)

    @unittest.skipUnless(
        hasattr(os, "symlink") and sys.platform != "win32",
        "Symlinks require POSIX or elevated privileges on Windows",
    )
    def test_rejects_parent_dir_symlink(self):
        # Symlink in a parent component that points outside
        outside_dir = tempfile.mkdtemp()
        outside_file = os.path.join(outside_dir, "data.bin")
        with open(outside_file, "wb") as f:
            f.write(b"data")
        # Create subdir/link -> outside_dir
        subdir = os.path.join(self.tmpdir, "models")
        os.makedirs(subdir, exist_ok=True)
        link_in_sub = os.path.join(subdir, "link")
        os.symlink(outside_dir, link_in_sub)
        try:
            with self.assertRaises(ValueError):
                validate_external_data_path("models/link/data.bin", self.tmpdir)
        finally:
            os.unlink(link_in_sub)
            import shutil

            shutil.rmtree(outside_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
