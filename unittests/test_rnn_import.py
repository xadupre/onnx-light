import unittest
from pathlib import Path


class TestRnnImport(unittest.TestCase):
    def test_rnn_files_imported(self):
        """Verifies that RNN defs files are vendored."""
        root = Path(__file__).resolve().parents[1]
        rnn = root / "onnx_light" / "onnx" / "defs" / "rnn"

        expected = {"defs.cc", "old.cc"}
        present = {path.name for path in rnn.glob("*") if path.is_file()}
        self.assertTrue(expected.issubset(present), msg=f"missing={sorted(expected - present)}")

    def test_rnn_uses_light_namespace(self):
        """Verifies that RNN files use ONNX_LIGHT_NAMESPACE."""
        root = Path(__file__).resolve().parents[1]
        rnn = root / "onnx_light" / "onnx" / "defs" / "rnn"

        for name in ("defs.cc", "old.cc"):
            content = (rnn / name).read_text(encoding="utf-8")
            self.assertIn("ONNX_LIGHT_NAMESPACE", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
