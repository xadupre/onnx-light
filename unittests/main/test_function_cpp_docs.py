import unittest
from pathlib import Path


class TestFunctionCppDocs(unittest.TestCase):
    def test_function_header_has_file_level_doxygen_documentation(self):
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "function.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("@file function.h", content)
        self.assertIn(
            "@brief Declares helpers for constructing and expanding ONNX FunctionProto bodies.",
            content,
        )
        self.assertIn("Provides utility types and helpers to build function-body nodes.", content)
        self.assertIn(
            "Fluent builder for FunctionProto definitions used by operator schemas.", content
        )

    def test_function_cpp_page_has_intro(self):
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "function.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("Helpers for constructing and expanding", content)
        self.assertIn(":cpp:class:`onnx::FunctionBodyHelper`", content)
        self.assertIn(":cpp:class:`onnx::FunctionBuilder`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
