import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestLogicalImport(ExtTestCase):
    def test_logical_files_imported(self):
        """Verifies that the logical defs files are vendored."""
        root = Path(__file__).resolve().parents[2]
        logical = root / "onnx_light" / "onnx" / "defs" / "logical"

        expected = {"defs.cc", "old.cc"}
        present = {path.name for path in logical.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_logical_uses_light_namespace(self):
        """Verifies that the logical files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        logical = root / "onnx_light" / "onnx" / "defs" / "logical"

        for name in ("defs.cc", "old.cc"):
            content = (logical / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
