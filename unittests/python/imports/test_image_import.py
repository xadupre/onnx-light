import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestImageImport(ExtTestCase):
    def test_image_files_imported(self):
        """Verifies that image defs files are vendored."""
        root = Path(__file__).resolve().parents[2]
        image = root / "onnx_light" / "onnx_lib" / "defs" / "image"

        expected = {"defs.cc"}
        present = {path.name for path in image.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_image_uses_light_namespace(self):
        """Verifies that image files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        image = root / "onnx_light" / "onnx_lib" / "defs" / "image"

        for name in ("defs.cc",):
            content = (image / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
