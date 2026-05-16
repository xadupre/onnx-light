import unittest
from pathlib import Path


class TestTensorUtilCppDocs(unittest.TestCase):
    def test_tensor_util_header_has_file_level_doxygen_documentation(self):
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "tensor_util.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("@file tensor_util.h", content)
        self.assertIn(
            "@brief Declares tensor conversion helpers used by ONNX schema definitions.", content
        )
        self.assertIn("Extracts typed element data from a Tensor wrapper.", content)
        self.assertIn("Creates a scalar TensorProto from a C++ value.", content)
        self.assertIn("Creates a 1D TensorProto from a vector of values.", content)

    def test_tensor_util_cpp_page_has_intro(self):
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "tensor_util.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("Tensor conversion helpers used by schema builders", content)
        self.assertIn(":cpp:func:`onnx::ParseData`", content)
        self.assertIn(":cpp:func:`onnx::ToTensor`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
