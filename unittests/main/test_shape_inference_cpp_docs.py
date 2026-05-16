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
        self.assertIn("Stores runtime options controlling schema-level shape inference.", content)
        self.assertIn(
            "Provides graph-level inference for attributes containing subgraphs.", content
        )
        self.assertIn(
            "Supplies inputs, outputs, and attributes to operator type-and-shape inferencers.",
            content,
        )
        self.assertIn("Supplies tensor-shape constants to data-propagation functions.", content)
        self.assertIn("@name ONNX-compatible helper APIs", content)

    def test_param_return_tags_present(self):
        """Verifies that key methods in shape_inference.h carry @param and @return tags."""
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "shape_inference.h"
        content = header.read_text(encoding="utf-8")
        # ShapeInferenceOptions constructor is documented
        self.assertIn("@param check_type_val", content)
        self.assertIn("@param strict_mode_val", content)
        self.assertIn("@param data_prop_val", content)
        # GraphInferencer::doInferencing is documented
        self.assertIn("@param input_types", content)
        self.assertIn("@param input_data", content)
        # InferenceError members are documented
        self.assertIn("@param message", content)
        self.assertIn("@param context", content)
        # InferenceContext methods carry @param / @return
        self.assertIn("@param name", content)
        self.assertIn("@param index", content)
        self.assertIn("@return Pointer to the AttributeProto", content)
        self.assertIn("@return Count of inputs", content)
        self.assertIn("@return Count of outputs", content)
        self.assertIn("@return Mutable pointer to the TypeProto", content)
        self.assertIn("@return A vector of pointers to inferred output TypeProto values", content)
        # DataPropagationContext::addOutputData is documented
        self.assertIn("@param tp", content)
        # Helper functions carry @param tags
        self.assertIn("@param inputIndex", content)
        self.assertIn("@param outputIndex", content)
        self.assertIn("@param elemType", content)
        self.assertIn("@param attributeName", content)
        self.assertIn("@param defaultValue", content)
        self.assertIn("@param shape", content)

    def test_macro_docs_present(self):
        """Verifies that the inference-error macros carry usage examples in their Doxygen."""
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "shape_inference.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("@def fail_type_inference", content)
        self.assertIn("@def fail_shape_inference", content)
        self.assertIn("[TypeInferenceError]", content)
        self.assertIn("[ShapeInferenceError]", content)

    def test_broadcast_helper_documented(self):
        """Verifies that bidirectionalBroadcastShapeInference is documented."""
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "shape_inference.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("NumPy-style broadcasting semantics", content)
        self.assertIn("@param shape1", content)
        self.assertIn("@param shape2", content)
        self.assertIn("@param output_shape", content)

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
