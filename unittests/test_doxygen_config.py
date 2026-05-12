import unittest
from pathlib import Path


class TestDoxygenConfig(unittest.TestCase):
    def test_doxygen_input_includes_onnx_defs(self):
        """Verifies that Doxygen indexes ONNX defs headers used by docs."""
        doxygen_path = Path(__file__).resolve().parents[1] / "docs" / "Doxyfile"
        content = doxygen_path.read_text(encoding="utf-8")
        self.assertIn("../onnx_light/onnx/defs", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
