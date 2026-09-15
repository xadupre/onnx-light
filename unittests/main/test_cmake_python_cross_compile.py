import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestCMakePythonCrossCompile(ExtTestCase):
    def test_cross_compile_python_package_ordering(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        start = content.index("if(CMAKE_CROSSCOMPILING)")
        end = content.index("else()", start)
        cross_block = content[start:end]

        expected_ordered_snippets = [
            (
                "find_package(Python3 3.12 REQUIRED COMPONENTS Development.Module\n"
                "      OPTIONAL_COMPONENTS Development.SABIModule)"
            ),
            "find_package(Python 3.12 REQUIRED COMPONENTS Interpreter)",
            "if(NOT TARGET Python::Module)",
            "add_library(Python::Module ALIAS Python3::Module)",
            "if(TARGET Python3::SABIModule AND NOT TARGET Python::SABIModule)",
            "add_library(Python::SABIModule ALIAS Python3::SABIModule)",
        ]
        cursor = 0
        for snippet in expected_ordered_snippets:
            with self.subTest(snippet=snippet):
                offset = cross_block.find(snippet, cursor)
                self.assertGreaterEqual(offset, 0, f"Missing snippet: {snippet}")
                cursor = offset + len(snippet)


if __name__ == "__main__":
    unittest.main(verbosity=2)
