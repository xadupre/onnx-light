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


if __name__ == "__main__":
    unittest.main(verbosity=2)
