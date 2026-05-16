import unittest
from pathlib import Path
from onnx_light.ext_test_case import ExtTestCase


class TestDefsUtilsCppDocs(ExtTestCase):
    def test_defs_utils_headers_have_file_level_doxygen_documentation(self):
        """Validates that defs subfolder utils.h headers define file-level Doxygen metadata."""
        repo = Path(__file__).resolve().parents[2]
        defs_dir = repo / "onnx_light" / "onnx" / "defs"
        expectations = {
            "controlflow/utils.h": (
                "@brief Declares control-flow shape-inference helpers shared by If, Loop, "
                "and Scan."
            ),
            "generator/utils.h": (
                "@brief Declares shape-inference helpers for generator-style operators."
            ),
            "math/utils.h": (
                "@brief Declares reusable math-operator schema and inference helpers."
            ),
            "nn/utils.h": "@brief Declares shared neural-network operator helpers.",
            "reduction/utils.h": (
                "@brief Declares shared reduction-operator schema generator helpers."
            ),
            "sequence/utils.h": "@brief Declares reusable schema helpers for sequence operators.",
            "tensor/utils.h": (
                "@brief Declares tensor-operator schema and shape-inference helpers."
            ),
            "traditionalml/utils.h": (
                "@brief Declares validation helpers shared by traditional-ML operator schemas."
            ),
        }

        for relpath, brief_line in expectations.items():
            content = (defs_dir / relpath).read_text(encoding="utf-8")
            self.assertIn(f"@file {relpath}", content)
            self.assertIn(brief_line, content)

    def test_defs_utils_rst_pages_have_intro_content(self):
        """Validates that defs utils.h C++ API pages include intro snippets and directives."""
        repo = Path(__file__).resolve().parents[2]
        docs_dir = repo / "docs" / "api" / "cpp" / "onnx" / "defs"
        expectations = {
            "controlflow_utils.rst": (
                "IfInferenceFunction",
                ".. doxygenfile:: controlflow/utils.h",
            ),
            "generator_utils.rst": ("ConstantOpInference", ".. doxygenfile:: generator/utils.h"),
            "math_utils.rst": ("TopKOpGenerator", ".. doxygenfile:: math/utils.h"),
            "nn_utils.rst": ("getConvPoolStrides", ".. doxygenfile:: nn/utils.h"),
            "reduction_utils.rst": ("ReduceOpGenerator", ".. doxygenfile:: reduction/utils.h"),
            "sequence_utils.rst": (
                "SplitToSequenceOpGenerator",
                ".. doxygenfile:: sequence/utils.h",
            ),
            "tensor_utils.rst": ("resizeShapeInference", ".. doxygenfile:: tensor/utils.h"),
            "traditionalml_utils.rst": (
                "AssertAttributeProtoTypeAndLength",
                ".. doxygenfile:: traditionalml/utils.h",
            ),
        }

        for filename, snippets in expectations.items():
            content = (docs_dir / filename).read_text(encoding="utf-8")
            for snippet in snippets:
                self.assertIn(snippet, content)

    def test_defs_index_lists_utils_pages(self):
        """Validates that the defs C++ docs index links the new utils pages."""
        repo = Path(__file__).resolve().parents[2]
        index = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "index.rst"
        content = index.read_text(encoding="utf-8")
        for entry in (
            "controlflow_utils",
            "generator_utils",
            "math_utils",
            "nn_utils",
            "reduction_utils",
            "sequence_utils",
            "tensor_utils",
            "traditionalml_utils",
        ):
            self.assertIn(entry, content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
