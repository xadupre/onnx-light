import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestTextImport(ExtTestCase):
    def test_text_files_imported(self):
        """Verifies that the text defs file is vendored."""
        root = Path(__file__).resolve().parents[2]
        text = root / "onnx_light" / "onnx" / "defs" / "text"

        expected = {"defs.cc"}
        present = {path.name for path in text.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_text_uses_light_namespace(self):
        """Verifies that the text defs file uses ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        text = root / "onnx_light" / "onnx" / "defs" / "text"

        for name in ("defs.cc",):
            content = (text / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
