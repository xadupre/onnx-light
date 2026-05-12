import unittest
from pathlib import Path


class TestReductionImport(unittest.TestCase):
    def test_reduction_files_imported(self):
        """Verifies that reduction defs files are vendored."""
        root = Path(__file__).resolve().parents[4]
        reduction = root / "onnx_light" / "onnx" / "defs" / "reduction"

        expected = {"defs.cc", "old.cc", "utils.cc", "utils.h"}
        present = {path.name for path in reduction.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_reduction_uses_light_namespace(self):
        """Verifies that reduction files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[4]
        reduction = root / "onnx_light" / "onnx" / "defs" / "reduction"

        for name in ("defs.cc", "old.cc", "utils.cc", "utils.h"):
            content = (reduction / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
