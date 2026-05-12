import unittest
from pathlib import Path


class TestInlinerImport(unittest.TestCase):
    def test_inliner_files_imported(self):
        """Verifies that inliner files are vendored."""
        root = Path(__file__).resolve().parents[2]
        inliner = root / "onnx_light" / "onnx" / "inliner"

        expected = {"inliner.cc", "inliner.h"}
        present = {path.name for path in inliner.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_inliner_uses_light_namespace(self):
        """Verifies that inliner files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[2]
        inliner = root / "onnx_light" / "onnx" / "inliner"

        for name in ("inliner.cc", "inliner.h"):
            content = (inliner / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
