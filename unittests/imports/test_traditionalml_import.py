import unittest
from pathlib import Path


class TestTraditionalmlImport(unittest.TestCase):
    def test_traditionalml_files_imported(self):
        """Verifies that traditionalml defs files are vendored."""
        root = Path(__file__).resolve().parents[2]
        traditionalml = root / "onnx_light" / "onnx" / "defs" / "traditionalml"

        expected = {"defs.cc", "old.cc", "utils.h"}
        present = {path.name for path in traditionalml.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_traditionalml_uses_light_namespace(self):
        """Verifies that traditionalml files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        traditionalml = root / "onnx_light" / "onnx" / "defs" / "traditionalml"

        for name in ("defs.cc", "old.cc", "utils.h"):
            content = (traditionalml / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
