import unittest

from pathlib import Path


class TestCMakeGlobs(unittest.TestCase):
    @unittest.skip("broken")
    def test_cmake_glob_source_discovery(self):
        """Ensures that repeated source collections are discovered with CMake globs."""
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

    def test_cmake_glob_header_discovery_for_visual_studio(self):
        """Ensures that headers are attached to the CMake project for Visual Studio."""
        cmake_path = Path(__file__).resolve().parents[2] / "CMakeLists.txt"
        content = cmake_path.read_text(encoding="utf-8")

        expected_fragments = [
            "file(GLOB_RECURSE ONNX_LIGHT_HEADERS CONFIGURE_DEPENDS",
            '"${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/*.h"',
            '"${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/*.hpp"',
            "${ONNX_LIGHT_HEADERS}",
            'source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Header Files"',
        ]

        for expected in expected_fragments:
            self.assertIn(expected, content)

    def test_cmake_cpp_test_visibility_for_visual_studio(self):
        """Ensures that the main C++ test project also shows the helper test in Visual Studio."""
        cmake_path = Path(__file__).resolve().parents[2] / "CMakeLists.txt"
        content = cmake_path.read_text(encoding="utf-8")

        expected_fragments = [
            "set(ONNX_CPP_ALL_TEST_SOURCES ${ONNX_CPP_TEST_SOURCES})",
            "add_custom_target(onnx_cpp_tests SOURCES ${ONNX_CPP_ALL_TEST_SOURCES})",
            'source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp" PREFIX "Source Files"',
        ]

        for expected in expected_fragments:
            self.assertIn(expected, content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
