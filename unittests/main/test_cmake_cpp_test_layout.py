import subprocess
import tempfile
import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestCMakeCppTestLayout(ExtTestCase):
    @staticmethod
    def _write_gtest_package(directory):
        """Creates a scoped imported-target package for configure-only checks."""
        package = directory / "gtest"
        package.mkdir()
        (package / "GTestConfig.cmake").write_text(
            "add_library(GTest::gtest INTERFACE IMPORTED)\n"
            "add_library(GTest::gtest_main INTERFACE IMPORTED)\n"
            "set_property(TARGET GTest::gtest_main PROPERTY "
            "INTERFACE_LINK_LIBRARIES GTest::gtest)\n",
            encoding="utf-8",
        )
        return package

    def test_installed_gtest_is_visible_to_parent_tests(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            package = self._write_gtest_package(directory)
            result = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(root),
                    "-B",
                    str(directory / "build"),
                    f"-DGTest_DIR={package}",
                    "-DONNX_LIGHT_BUILD_PYTHON=OFF",
                    "-DONNX_LIGHT_BUILD_KERNELS=OFF",
                    "-DONNX_LIGHT_BUILD_TESTS=ON",
                    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_windows_proto_runtime_outputs_share_a_directory(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            package = self._write_gtest_package(directory)
            (directory / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.15)\n"
                "project(ProtoRuntimeLayout LANGUAGES C CXX)\n"
                "set(WIN32 TRUE)\n"
                "set(ONNX_PROTO_BUILD_SHARED ON)\n"
                "set(ONNX_PROTO_BUILD_TESTS ON)\n"
                "if(CUSTOM_RUNTIME)\n"
                '  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/runtime")\n'
                '  set(expected "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")\n'
                "else()\n"
                '  set(expected "${CMAKE_BINARY_DIR}/proto")\n'
                "endif()\n"
                f'add_subdirectory("{(root / "onnx_proto").as_posix()}" proto)\n'
                "foreach(target IN ITEMS lib_onnx_proto test_onnx_proto "
                "test_stream_class_print test_onnx_dlpack test_onnx_light_helpers "
                "proto_consumer)\n"
                "  get_target_property(output ${target} RUNTIME_OUTPUT_DIRECTORY)\n"
                "  if(NOT output STREQUAL expected)\n"
                '    message(FATAL_ERROR "${target}: ${output} != ${expected}")\n'
                "  endif()\n"
                "endforeach()\n",
                encoding="utf-8",
            )
            for custom in ("OFF", "ON"):
                with self.subTest(custom_runtime=custom):
                    result = subprocess.run(
                        [
                            "cmake",
                            "-S",
                            str(directory),
                            "-B",
                            str(directory / custom),
                            f"-DGTest_DIR={package}",
                            f"-DCUSTOM_RUNTIME={custom}",
                            "-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=TRUE",
                            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_benchmarks_configure_without_openmp_lookup(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "FindOpenMP.cmake").write_text(
                'message(FATAL_ERROR "onnx-light must not look for OpenMP")\n', encoding="utf-8"
            )
            result = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(root),
                    "-B",
                    str(directory / "build"),
                    f"-DCMAKE_MODULE_PATH={directory.as_posix()}",
                    "-DONNX_LIGHT_BUILD_PYTHON=OFF",
                    "-DONNX_LIGHT_BUILD_KERNELS=OFF",
                    "-DONNX_LIGHT_BUILD_TESTS=OFF",
                    "-DONNX_LIGHT_BUILD_BENCHMARKS=ON",
                    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_helper_test_belongs_to_standalone_proto_project(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        proto_tests = (root / "onnx_proto/tests/CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("set(ONNX_PROTO_BUILD_TESTS ${ONNX_LIGHT_BUILD_TESTS})", content)
        self.assertIn(
            "target_link_libraries(test_onnx_light_helpers PRIVATE lib_onnx_proto)", proto_tests
        )
        self.assertNotIn("unittests/cc/onnx_proto", content)
        self.assertNotIn("unittests/cpp-exe/test_onnx_light_helpers.cc", content)
        self.assertTrue((root / "onnx_proto/tests/test_onnx_light_helpers.cc").exists())
        self.assertFalse((root / "unittests" / "cpp-exe").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
