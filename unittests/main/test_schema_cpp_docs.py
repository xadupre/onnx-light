import unittest
from pathlib import Path


class TestSchemaCppDocs(unittest.TestCase):
    def test_schema_header_has_file_level_doxygen_documentation(self):
        repo = Path(__file__).resolve().parents[2]
        schema_header = repo / "onnx_light" / "onnx" / "defs" / "schema.h"
        content = schema_header.read_text(encoding="utf-8")
        self.assertIn("@file schema.h", content)
        self.assertIn(
            "@brief Declares ONNX operator schema types and registration helpers.", content
        )

    def test_schema_cpp_page_has_intro(self):
        repo = Path(__file__).resolve().parents[2]
        schema_page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "schema.rst"
        content = schema_page.read_text(encoding="utf-8")
        self.assertIn("Operator schema definitions and registration utilities", content)
        self.assertIn(":cpp:class:`onnx::OpSchema`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
