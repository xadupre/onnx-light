import unittest
from pathlib import Path


class TestSequenceImport(unittest.TestCase):
    def test_sequence_files_imported(self):
        """Verifies that sequence defs files are vendored."""
        root = Path(__file__).resolve().parents[4]
        sequence = root / "onnx_light" / "onnx" / "defs" / "sequence"

        expected = {"defs.cc", "utils.cc", "utils.h"}
        present = {path.name for path in sequence.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_sequence_uses_light_namespace(self):
        """Verifies that sequence files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[4]
        sequence = root / "onnx_light" / "onnx" / "defs" / "sequence"

        for name in ("defs.cc", "utils.cc", "utils.h"):
            content = (sequence / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
