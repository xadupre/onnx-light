import unittest
from pathlib import Path


class TestDoxygenConfig(unittest.TestCase):
    def test_doxygen_input_includes_onnx_tree(self):
        """Tests that Doxygen indexes the ONNX tree used by C++ docs pages."""
        doxygen_path = Path(__file__).resolve().parents[1] / "docs" / "Doxyfile"
        content = doxygen_path.read_text(encoding="utf-8")
        self.assertIn("../onnx_light/onnx", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
