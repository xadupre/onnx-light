import unittest
from pathlib import Path


class TestFunctionImport(unittest.TestCase):
    def test_function_files_imported(self):
        """Checks that function defs files are vendored."""
        defs = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx" / "defs"

        expected = {"function.cc", "function.h"}
        present = {path.name for path in defs.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_function_files_use_light_namespace(self):
        """Checks that function defs files use ONNX_LIGHT_NAMESPACE."""
        defs = Path(__file__).resolve().parents[2] / "onnx_light" / "onnx" / "defs"

        for name in ("function.cc", "function.h"):
            content = (defs / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
