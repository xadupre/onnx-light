import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestPythonApiDocs(ExtTestCase):
    def test_python_api_index_lists_core_modules(self):
        """Tests that Python API index references the documented core submodules."""
        index_path = Path(__file__).resolve().parents[2] / "docs" / "api" / "onnx" / "index.rst"
        content = index_path.read_text(encoding="utf-8")
        for page in [
            "compose",
            "defs",
            "helper",
            "io_helper",
            "numpy_helper",
            "parser",
            "pychecker",
            "shape_inference",
            "protos",
        ]:
            self.assertIn(f"    {page}", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
