import unittest

from pathlib import Path


class TestCMakeGlobs(unittest.TestCase):
    def test_root_cmake_uses_globs_for_repeated_source_lists(self):
        """Verifies that repeated source collections are discovered with CMake globs."""
        cmake_path = Path(__file__).resolve().parents[2] / "CMakeLists.txt"
        content = cmake_path.read_text(encoding="utf-8")

        expected_fragments = [
            "file(GLOB ONNX_LIGHT_SOURCES CONFIGURE_DEPENDS",
            "file(GLOB ONNX_COMMON_SOURCES CONFIGURE_DEPENDS",
            "file(GLOB ONNX_DEFS_SOURCES CONFIGURE_DEPENDS",
            "file(GLOB ONNX_VERSION_CONVERTER_SOURCES CONFIGURE_DEPENDS",
            "file(GLOB_RECURSE ONNX_CPP_TEST_SOURCES CONFIGURE_DEPENDS",
            "file(GLOB_RECURSE ONNX_LIGHT_BENCHMARK_SOURCES CONFIGURE_DEPENDS",
        ]

        for expected in expected_fragments:
            self.assertIn(expected, content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
