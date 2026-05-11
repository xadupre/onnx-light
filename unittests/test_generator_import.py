import unittest
from pathlib import Path


class TestGeneratorImport(unittest.TestCase):
    def test_generator_files_imported(self):
        """Checks that generator defs files are vendored."""
        root = Path(__file__).resolve().parents[1]
        generator = root / "onnx_light" / "onnx" / "defs" / "generator"

        expected = {"defs.cc", "old.cc", "utils.cc", "utils.h"}
        present = {path.name for path in generator.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_generator_uses_light_namespace(self):
        """Checks that generator files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[1]
        generator = root / "onnx_light" / "onnx" / "defs" / "generator"

        for name in ("defs.cc", "old.cc", "utils.cc", "utils.h"):
            content = (generator / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
