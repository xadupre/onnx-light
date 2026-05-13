import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestCheckerImport(ExtTestCase):
    def test_checker_files_imported(self):
        """Verifies that checker files are vendored."""
        onnx_dir = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx"

        expected = {"checker.cc", "checker.h"}
        present = {path.name for path in onnx_dir.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_checker_files_use_light_namespace(self):
        """Verifies that checker files use ONNX_LIGHT_NAMESPACE."""
        onnx_dir = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx"

        for name in ("checker.cc", "checker.h"):
            content = (onnx_dir / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
