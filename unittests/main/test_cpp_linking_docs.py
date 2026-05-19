import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestCppLinkingDocs(unittest.TestCase):
    def test_cpp_linking_mentions_proto_only_target(self):
        """Verifies that the C++ linking guide documents the proto-only target."""
        page = ROOT / "docs" / "design" / "cplusplus_linking.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("onnx_light::lib_onnx_proto", content)
        self.assertIn("does not need any notion of operators", content)

    def test_differences_page_mentions_operator_aware_split(self):
        """Verifies when the onnx comparison page says the full library is needed."""
        page = ROOT / "docs" / "design" / "differences.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("onnx_light::lib_onnx_proto", content)
        self.assertIn("operator notions", content)

    def test_load_example_docs_match_proto_only_target(self):
        """Verifies that the load example docs reference the proto-only CMake target."""
        page = ROOT / "docs" / "examples_cc" / "load_onnx_light_time_example.rst"
        content = page.read_text(encoding="utf-8")
        self.assertIn("onnx_light::lib_onnx_proto", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
