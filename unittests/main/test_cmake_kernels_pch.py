import subprocess
import tempfile
import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestCMakeKernelsPch(ExtTestCase):
    def test_runtime_pch_is_private_and_optional(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 4.0)\n"
                "project(KernelsPch LANGUAGES CXX)\n"
                f'add_subdirectory("{root.as_posix()}" onnx-light)\n'
                'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/pch.txt" CONTENT\n'
                '  "$<TARGET_PROPERTY:lib_onnx_kernels,PRECOMPILE_HEADERS>")\n'
                'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/interface-pch.txt" CONTENT\n'
                '  "$<TARGET_PROPERTY:lib_onnx_kernels,INTERFACE_PRECOMPILE_HEADERS>")\n',
                encoding="utf-8",
            )
            for enabled in ("ON", "OFF"):
                with self.subTest(pch=enabled):
                    build = directory / enabled
                    result = subprocess.run(
                        [
                            "cmake",
                            "-S",
                            str(directory),
                            "-B",
                            str(build),
                            "-DONNX_LIGHT_BUILD_PYTHON=OFF",
                            "-DONNX_LIGHT_BUILD_KERNELS=ON",
                            "-DONNX_LIGHT_BUILD_TESTS=OFF",
                            f"-DONNX_LIGHT_USE_PCH={enabled}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    headers = (build / "pch.txt").read_text(encoding="utf-8")
                    self.assertEqual(
                        (build / "interface-pch.txt").read_text(encoding="utf-8"), ""
                    )
                    if enabled == "ON":
                        self.assertIn("onnx_core/backend_test/test_case.h", headers)
                        self.assertIn("onnx_core/runtime/kernels/node_helpers.h", headers)
                    else:
                        self.assertEqual(headers, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
