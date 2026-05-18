import unittest
from pathlib import Path


class TestCppTestDocs(unittest.TestCase):
    def test_readme_documents_how_to_build_cpp_tests(self):
        readme = Path(__file__).resolve().parents[2] / "README.md"
        content = readme.read_text(encoding="utf-8")

        self.assertIn("cmake.define.ONNX_LIGHT_BUILD_TESTS=ON", content)
        self.assertIn(
            "python setup.py build_ext --inplace --build-temp build --cpp-tests", content
        )
        self.assertIn("ctest --test-dir build --output-on-failure", content)

    def test_docs_index_documents_how_to_build_cpp_tests(self):
        index = Path(__file__).resolve().parents[2] / "docs" / "index.rst"
        content = index.read_text(encoding="utf-8")

        self.assertIn("cmake.define.ONNX_LIGHT_BUILD_TESTS=ON", content)
        self.assertIn(
            "python setup.py build_ext --inplace --build-temp build --cpp-tests", content
        )
        self.assertIn("ctest --test-dir build --output-on-failure", content)

    def test_print_proto_debug_page_linked_and_contains_function(self):
        examples_index = (
            Path(__file__).resolve().parents[2] / "docs" / "design" / "examples" / "index.rst"
        )
        content = examples_index.read_text(encoding="utf-8")
        self.assertIn("print_proto_debug_example", content)

        debug_page = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "design"
            / "examples"
            / "print_proto_debug_example.rst"
        )
        self.assertTrue(debug_page.exists())
        page_content = debug_page.read_text(encoding="utf-8")
        self.assertIn("PrintToVectorString", page_content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
