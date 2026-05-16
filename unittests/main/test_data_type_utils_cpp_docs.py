import unittest
from pathlib import Path


class TestDataTypeUtilsCppDocs(unittest.TestCase):
    def test_header_has_file_level_doxygen_docs(self):
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "data_type_utils.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("@file data_type_utils.h", content)
        self.assertIn(
            "@brief Declares helpers that convert between ONNX type descriptors.", content
        )
        self.assertIn(
            "Converts ONNX type spellings into canonical DataType and TypeProto forms", content
        )
        self.assertIn(
            "This function converts a type string into a canonical DataType identifier", content
        )

    def test_cpp_page_has_intro(self):
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "data_type_utils.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("Utilities for converting ONNX type spellings and protobuf", content)
        self.assertIn(":cpp:class:`onnx::Utils::DataTypeUtils`", content)
        self.assertIn(":cpp:type:`onnx::DataType`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
