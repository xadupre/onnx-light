import unittest
from pathlib import Path


class TestShapeInferenceCppDocs(unittest.TestCase):
    def test_shape_inference_header_has_file_level_doxygen_documentation(self):
        """Verifies that shape_inference.h defines file-level Doxygen metadata."""
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "shape_inference.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("@file shape_inference.h", content)
        self.assertIn(
            "@brief Declares interfaces and helper utilities for operator shape inference.",
            content,
        )

    def test_shape_inference_cpp_page_has_intro(self):
        """Verifies that the C++ API page documents the shape inference header."""
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "shape_inference.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("Core declarations for operator type-and-shape inference", content)
        self.assertIn(":cpp:class:`onnx::ShapeInferenceOptions`", content)
        self.assertIn(":cpp:func:`onnx::propagateElemTypeFromInputToOutput`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
