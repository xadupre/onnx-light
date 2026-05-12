import unittest
from pathlib import Path


class TestOptionalImport(unittest.TestCase):
    def test_optional_files_imported(self):
        """Checks that optional defs files are vendored."""
        root = Path(__file__).resolve().parents[1]
        optional = root / "onnx_light" / "onnx" / "defs" / "optional"

        expected = {"defs.cc", "old.cc"}
        present = {path.name for path in optional.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_optional_uses_light_namespace(self):
        """Checks that optional files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[1]
        optional = root / "onnx_light" / "onnx" / "defs" / "optional"

        for name in ("defs.cc", "old.cc"):
            content = (optional / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
