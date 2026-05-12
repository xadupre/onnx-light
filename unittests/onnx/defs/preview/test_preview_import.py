import unittest
from pathlib import Path


class TestPreviewImport(unittest.TestCase):
    def test_preview_files_imported(self):
        """Verifies that preview defs files are vendored."""
        root = Path(__file__).resolve().parents[4]
        preview = root / "onnx_light" / "onnx" / "defs" / "preview"

        expected = {"defs.cc"}
        present = {path.name for path in preview.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_preview_uses_light_namespace(self):
        """Verifies that preview files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[4]
        preview = root / "onnx_light" / "onnx" / "defs" / "preview"

        for name in ("defs.cc",):
            content = (preview / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
