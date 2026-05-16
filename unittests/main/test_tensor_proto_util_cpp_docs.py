import unittest
from pathlib import Path


class TestTensorProtoUtilCppDocs(unittest.TestCase):
    def test_tensor_proto_util_header_has_file_level_doxygen_documentation(self):
        """Verifies that tensor_proto_util.h defines file-level Doxygen metadata."""
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "tensor_proto_util.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("@file tensor_proto_util.h", content)
        self.assertIn(
            "@brief Compatibility stub that redirects to onnx/defs/tensor_util.h.",
            content,
        )
        self.assertIn("ParseData", content)
        self.assertIn("ToTensor", content)

    def test_tensor_proto_util_cpp_page_has_intro(self):
        """Verifies that the C++ API page documents tensor_proto_util.h."""
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "tensor_proto_util.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("tensor_util", content)
        self.assertIn(":cpp:func:`onnx::ParseData`", content)
        self.assertIn(":cpp:func:`onnx::ToTensor`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
