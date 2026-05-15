import unittest
from pathlib import Path


class TestCheckerCppDocs(unittest.TestCase):
    def test_checker_header_has_file_level_doxygen_documentation(self):
        repo = Path(__file__).resolve().parents[2]
        checker_header = repo / "onnx_light" / "onnx" / "checker.h"
        content = checker_header.read_text(encoding="utf-8")
        self.assertIn("@file checker.h", content)
        self.assertIn("@brief Declares ONNX model and graph validation entry points.", content)

    def test_checker_cpp_page_has_intro(self):
        repo = Path(__file__).resolve().parents[2]
        checker_page = repo / "docs" / "api" / "cpp" / "onnx" / "checker.rst"
        content = checker_page.read_text(encoding="utf-8")
        self.assertIn("Validation utilities for ONNX graphs and models", content)
        self.assertIn(":cpp:func:`onnx::checker::check_model`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
