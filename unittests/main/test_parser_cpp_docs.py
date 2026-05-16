import unittest
from pathlib import Path


class TestParserCppDocs(unittest.TestCase):
    def test_parser_header_has_file_level_doxygen_documentation(self):
        """Verifies that parser.h defines file-level Doxygen documentation."""
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "parser.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("@file parser.h", content)
        self.assertIn(
            "@brief Declares the ONNX text-format parser and related helpers.",
            content,
        )
        self.assertIn("cursor-based tokenizer", content)
        self.assertIn("builds protobuf structures from text", content)
        self.assertIn("Maps ONNX primitive type name strings", content)
        self.assertIn("Maps ONNX attribute type name strings", content)
        self.assertIn("Singleton map from keyword identifier strings", content)
        self.assertIn("Cursor-based tokenizer that drives parsing", content)
        self.assertIn(
            "High-level ONNX text-format parser that builds protobuf structures.",
            content,
        )

    def test_parser_cpp_page_has_intro(self):
        """Verifies that the C++ API page documents the parser header."""
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "parser.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("ONNX text-format parser declarations", content)
        self.assertIn(":cpp:class:`onnx::OnnxParser`", content)
        self.assertIn(":cpp:class:`onnx::ParserBase`", content)
        self.assertIn(":cpp:class:`onnx::PrimitiveTypeNameMap`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
