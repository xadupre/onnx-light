import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestObjectDetectionImport(ExtTestCase):
    def test_object_detection_files_imported(self):
        """Verifies that object_detection defs files are vendored."""
        root = Path(__file__).resolve().parents[2]
        object_detection = root / "onnx_light" / "onnx_lib" / "defs" / "object_detection"

        expected = {"defs.cc"}
        present = {path.name for path in object_detection.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_object_detection_uses_light_namespace(self):
        """Verifies that object_detection files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        object_detection = root / "onnx_light" / "onnx_lib" / "defs" / "object_detection"

        for name in ("defs.cc",):
            content = (object_detection / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
