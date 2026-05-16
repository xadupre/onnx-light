import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestSchemaCppDocs(ExtTestCase):
    def test_schema_header_has_file_and_api_documentation(self):
        """Validates that schema.h exposes key file-level and API-level Doxygen docs."""
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "schema.h"
        content = header.read_text(encoding="utf-8")

        self.assertIn("@file schema.h", content)
        self.assertIn(
            "@brief Declares ONNX operator schema types and registration helpers.", content
        )
        for snippet in (
            "/// Returns the schema support level.",
            "/// Adds a pre-built attribute declaration.",
            "/// Declares a formal input parameter by index.",
            "/// Returns all currently loaded schemas including historical versions.",
        ):
            self.assertIn(snippet, content)

    def test_schema_rst_page_has_intro_and_doxygen_directive(self):
        """Validates that schema.rst includes intro text and a doxygenfile directive."""
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "schema.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("Operator schema definitions and registration utilities", content)
        self.assertIn(".. doxygenfile:: schema.h", content)

    def test_defs_index_lists_schema_page(self):
        """Validates that the defs C++ docs index includes the schema page."""
        repo = Path(__file__).resolve().parents[2]
        index = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "index.rst"
        content = index.read_text(encoding="utf-8")
        self.assertIn("schema", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
