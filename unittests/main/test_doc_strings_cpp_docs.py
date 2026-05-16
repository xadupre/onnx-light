import unittest
from pathlib import Path


class TestDocStringsCppDocs(unittest.TestCase):
    def test_header_has_file_level_docs(self):
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "doc_strings.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("/**", content)
        self.assertIn("@file doc_strings.h", content)
        self.assertIn(
            "@brief Declares exported ONNX operator documentation string constants.", content
        )
        self.assertIn("*/", content)
        self.assertIn("@name Operator documentation strings", content)

    def test_rst_page_has_intro(self):
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "doc_strings.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("Operator documentation strings for ONNX schemas", content)
        self.assertIn(":cpp:var:`kDoc_Relu_ver6`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
