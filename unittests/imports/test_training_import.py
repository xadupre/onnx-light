import unittest
from pathlib import Path


class TestTrainingImport(unittest.TestCase):
    def test_training_files_imported(self):
        """Verifies that training defs files are vendored."""
        root = Path(__file__).resolve().parents[2]
        training = root / "onnx_light" / "onnx" / "defs" / "training"

        expected = {"defs.cc"}
        present = {path.name for path in training.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_training_uses_light_namespace(self):
        """Verifies that training files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        training = root / "onnx_light" / "onnx" / "defs" / "training"

        for name in ("defs.cc",):
            content = (training / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
