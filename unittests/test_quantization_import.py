import unittest
from pathlib import Path


class TestQuantizationImport(unittest.TestCase):
    def test_quantization_files_imported(self):
        """Verifies that quantization defs files are vendored."""
        root = Path(__file__).resolve().parents[1]
        quantization = root / "onnx_light" / "onnx" / "defs" / "quantization"

        expected = {"defs.cc", "old.cc"}
        present = {path.name for path in quantization.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_quantization_uses_light_namespace(self):
        """Verifies that quantization files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[1]
        quantization = root / "onnx_light" / "onnx" / "defs" / "quantization"

        for name in ("defs.cc", "old.cc"):
            content = (quantization / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
