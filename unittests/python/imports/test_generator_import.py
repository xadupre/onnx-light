import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestGeneratorImport(ExtTestCase):
    def test_generator_files_imported(self):
        """Verifies that generator defs files are vendored."""
        root = Path(__file__).resolve().parents[2]
        generator = root / "onnx_light" / "onnx_lib" / "defs" / "generator"

        expected = {"defs.cc", "old.cc", "utils.cc", "utils.h"}
        present = {path.name for path in generator.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_generator_uses_light_namespace(self):
        """Verifies that generator files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        generator = root / "onnx_light" / "onnx_lib" / "defs" / "generator"

        for name in ("defs.cc", "old.cc", "utils.cc", "utils.h"):
            content = (generator / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
