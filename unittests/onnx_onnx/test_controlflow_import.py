import unittest
from pathlib import Path


class TestControlflowImport(unittest.TestCase):
    def test_controlflow_files_imported(self):
        """Verifies that controlflow defs files are vendored."""
        root = Path(__file__).resolve().parents[1]
        controlflow = root / "onnx_light" / "onnx" / "defs" / "controlflow"

        expected = {"defs.cc", "old.cc", "utils.cc", "utils.h"}
        present = {path.name for path in controlflow.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_controlflow_uses_light_namespace(self):
        """Verifies that controlflow files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[1]
        controlflow = root / "onnx_light" / "onnx" / "defs" / "controlflow"

        for name in ("defs.cc", "old.cc", "utils.cc", "utils.h"):
            content = (controlflow / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
