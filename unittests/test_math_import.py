import unittest
from pathlib import Path


class TestMathImport(unittest.TestCase):
    def test_math_files_imported(self):
        """Verifies that math defs files are vendored."""
        root = Path(__file__).resolve().parents[1]
        math = root / "onnx_light" / "onnx" / "defs" / "math"

        expected = {"defs.cc", "old.cc", "utils.cc", "utils.h"}
        present = {path.name for path in math.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_math_uses_light_namespace(self):
        """Verifies that math files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[1]
        math = root / "onnx_light" / "onnx" / "defs" / "math"

        for name in ("defs.cc", "old.cc", "utils.cc", "utils.h"):
            content = (math / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
