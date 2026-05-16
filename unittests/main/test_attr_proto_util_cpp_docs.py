import unittest
from pathlib import Path


class TestAttrProtoUtilCppDocs(unittest.TestCase):
    def test_header_has_doxygen_comments(self):
        repo = Path(__file__).resolve().parents[2]
        header = repo / "onnx_light" / "onnx" / "defs" / "attr_proto_util.h"
        content = header.read_text(encoding="utf-8")
        self.assertIn("@file attr_proto_util.h", content)
        self.assertIn(
            "@brief Declares helpers to construct ONNX AttributeProto instances.", content
        )
        self.assertIn("Creates a FLOAT attribute with the provided value.", content)
        self.assertIn(
            "Creates a reference attribute with distinct function-body and source names.", content
        )

    def test_rst_page_has_intro(self):
        repo = Path(__file__).resolve().parents[2]
        page = repo / "docs" / "api" / "cpp" / "onnx" / "defs" / "attr_proto_util.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("Attribute construction helpers for ONNX nodes", content)
        self.assertIn(":cpp:func:`onnx::MakeAttribute`", content)
        self.assertIn(":cpp:func:`onnx::MakeRefAttribute`", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
