import unittest
from pathlib import Path


class TestNnImport(unittest.TestCase):
    def test_nn_files_imported(self):
        """Verifies that nn defs files are vendored."""
        root = Path(__file__).resolve().parents[4]
        nn = root / "onnx_light" / "onnx" / "defs" / "nn"

        expected = {"defs.cc", "old.cc", "utils.cc", "utils.h"}
        present = {path.name for path in nn.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_nn_uses_light_namespace(self):
        """Verifies that nn files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[4]
        nn = root / "onnx_light" / "onnx" / "defs" / "nn"

        for name in ("defs.cc", "old.cc", "utils.cc", "utils.h"):
            content = (nn / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
