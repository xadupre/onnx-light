import subprocess
import tempfile
import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestCMakePch(ExtTestCase):
    def test_extension_pch_policy(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 4.0)\n"
                "project(PchPolicy LANGUAGES C CXX)\n"
                "set(CMAKE_CXX_COMPILER_LAUNCHER ${TEST_LAUNCHER})\n"
                "set(CMAKE_C_COMPILER_LAUNCHER ${TEST_C_LAUNCHER})\n"
                f'add_subdirectory("{root.as_posix()}" onnx_light)\n'
                "foreach(target IN ITEMS lib_onnx_shape lib_onnx_patterns)\n"
                "  if(target STREQUAL lib_onnx_shape)\n"
                '    set(header "onnx_core/shapes/shapes_context.h")\n'
                "  else()\n"
                '    set(header "onnx_core/builder/graph_graph.h")\n'
                "  endif()\n"
                "  get_target_property(pch ${target} PRECOMPILE_HEADERS)\n"
                "  if(EXPECT_PCH)\n"
                f'    set(expected "{root.as_posix()}/onnx_light/${{header}}")\n'
                "    if(NOT pch STREQUAL expected)\n"
                '      message(FATAL_ERROR "${target}: ${pch} != ${expected}")\n'
                "    endif()\n"
                "  elseif(pch)\n"
                '    message(FATAL_ERROR "${target}: unexpected PCH ${pch}")\n'
                "  endif()\n"
                "  get_target_property(interface_pch ${target} INTERFACE_PRECOMPILE_HEADERS)\n"
                "  if(interface_pch)\n"
                '    message(FATAL_ERROR "${target}: PCH must remain private")\n'
                "  endif()\n"
                "endforeach()\n",
                encoding="utf-8",
            )
            cases = (
                ("default", "", "", None, True),
                ("disabled", "", "", "OFF", False),
                ("ccache", "ccache", "", None, False),
                ("sccache", "sccache", "", None, False),
                ("forced_ccache", "ccache", "", "ON", True),
                ("forced_sccache", "sccache", "", "ON", True),
                ("c_ccache", "", "ccache", None, False),
                ("c_sccache", "", "sccache", None, False),
                ("forced_c_ccache", "", "ccache", "ON", True),
                ("forced_c_sccache", "", "sccache", "ON", True),
            )
            for name, launcher, c_launcher, use_pch, expected in cases:
                with self.subTest(case=name):
                    command = [
                        "cmake",
                        "-S",
                        str(directory),
                        "-B",
                        str(directory / name),
                        "-DONNX_LIGHT_BUILD_PYTHON=OFF",
                        "-DONNX_LIGHT_BUILD_KERNELS=OFF",
                        "-DONNX_LIGHT_BUILD_TESTS=OFF",
                        f"-DTEST_LAUNCHER={launcher}",
                        f"-DTEST_C_LAUNCHER={c_launcher}",
                        f"-DEXPECT_PCH={'ON' if expected else 'OFF'}",
                    ]
                    if use_pch is not None:
                        command.append(f"-DONNX_LIGHT_USE_PCH={use_pch}")
                    result = subprocess.run(
                        command, capture_output=True, text=True, timeout=120, check=False
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
