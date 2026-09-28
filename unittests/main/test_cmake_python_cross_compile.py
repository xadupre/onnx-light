import shutil
import subprocess
import tempfile
import unittest

from pathlib import Path


class TestCMakePythonCrossCompile(unittest.TestCase):
    def test_cross_compile_python_package_ordering(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        start = content.index("find_package(Python3 3.12 REQUIRED COMPONENTS")
        end = content.index("  # Find nanobind", start)
        cross_block = content[start:end]

        expected_ordered_snippets = [
            "find_package(Python3 3.12 REQUIRED COMPONENTS ${python_dev_component})",
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

    def configure_discovery(self, settings, succeeds=True):
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake is unavailable")
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        start = content.index("if(ONNX_LIGHT_BUILD_PYTHON)\n")
        end = content.index("  # Find nanobind", start)
        discovery = content[start:end] + "endif()\n"
        # Exercise the production CMake decisions without needing every Python
        # ABI/platform installed. Imported targets still use real CMake aliases.
        project = (
            """cmake_minimum_required(VERSION 3.15)
project(PythonDiscovery NONE)
set(ONNX_LIGHT_BUILD_PYTHON ON)
set(CMAKE_VERSION 4.1)
set(WIN32 OFF)
set(test_gil_disabled 0)
macro(execute_process)
  cmake_parse_arguments(query "" "OUTPUT_VARIABLE" "" ${ARGV})
  set(${query_OUTPUT_VARIABLE} ${test_gil_disabled})
endmacro()
macro(find_package package)
  message(STATUS "test_find=${ARGV}")
  foreach(component ${ARGN})
    if(component STREQUAL "Development.Module")
      add_library(${package}::Module INTERFACE IMPORTED)
    elseif(component STREQUAL "Development.SABIModule")
      if(test_sabi_missing)
        message(FATAL_ERROR "Missing Development.SABIModule")
      endif()
      add_library(${package}::SABIModule INTERFACE IMPORTED)
    endif()
  endforeach()
endmacro()
"""
            + settings
            + "\n"
            + discovery
            + """
message(STATUS "test_abi=${Python_FIND_ABI}|${Python3_FIND_ABI}")
message(STATUS "test_cache_abi=$CACHE{Python_FIND_ABI}|$CACHE{Python3_FIND_ABI}")
"""
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "CMakeLists.txt").write_text(project, encoding="utf-8")
            result = subprocess.run(
                [cmake, "-S", str(source), "-B", str(source / "build")],
                capture_output=True,
                text=True,
                check=False,
            )
        output = result.stdout + result.stderr
        if not succeeds:
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("Missing Development.SABIModule", output)
            return None
        self.assertEqual(result.returncode, 0, output)
        return {
            key: [
                line.split("=", 1)[1]
                for line in result.stdout.splitlines()
                if line.startswith(f"-- test_{key}=")
            ]
            for key in ("find", "abi", "cache_abi")
        }

    def test_native_free_threaded_abi(self):
        abi = "ANY;ANY;ANY;ON"
        for version, windows, expected in [
            ("3.29", False, ""),
            ("3.30", False, ""),
            ("3.31", False, ""),
            ("4.0", False, ""),
            ("4.1", False, abi),
            ("3.30", True, abi),
            ("4.1", True, abi),
        ]:
            for cached in (False, True):
                with self.subTest(version=version, windows=windows, cached=cached):
                    settings = (
                        f"set(CMAKE_VERSION {version})\n"
                        f"set(WIN32 {'ON' if windows else 'OFF'})\n"
                        "set(test_gil_disabled 1)\n"
                    )
                    if cached:
                        settings += (
                            f'set(Python_FIND_ABI "{abi}" CACHE STRING "")\n'
                            f'set(Python3_FIND_ABI "{abi}" CACHE STRING "")\n'
                        )
                    result = self.configure_discovery(settings)
                    expected_abi = abi if cached and version == "3.29" else expected
                    self.assertEqual(result["abi"], [f"{expected_abi}|{expected_abi}"])
                    expected_cache = expected_abi if cached else ""
                    self.assertEqual(result["cache_abi"], [f"{expected_cache}|{expected_cache}"])
                    self.assertEqual(
                        result["find"],
                        [
                            "Python;3.12;REQUIRED;COMPONENTS;Interpreter",
                            "Python;3.12;REQUIRED;COMPONENTS;Interpreter;Development.Module",
                        ],
                    )

    def test_explicit_native_abi_is_preserved(self):
        abi = "OFF;OFF;ON;ON"
        result = self.configure_discovery(
            f'set(test_gil_disabled 1)\nset(Python_FIND_ABI "{abi}")\n'
            f'set(Python3_FIND_ABI "{abi}")'
        )
        self.assertEqual(result["abi"], [f"{abi}|{abi}"])

    def test_stable_abi_component_selection(self):
        for cross in (False, True):
            for gil_disabled in (False, True):
                for skbuild in (None, "", "Development.SABIModule"):
                    with self.subTest(cross=cross, gil=gil_disabled, skbuild=skbuild):
                        settings = (
                            f"set(CMAKE_CROSSCOMPILING {'ON' if cross else 'OFF'})\n"
                            f"set(test_gil_disabled {int(gil_disabled)})\n"
                        )
                        if cross and gil_disabled:
                            settings += 'set(Python3_FIND_ABI "ANY;ANY;ANY;ON")\n'
                        if skbuild is not None:
                            settings += f'set(SKBUILD_SABI_COMPONENT "{skbuild}")\n'
                        result = self.configure_discovery(settings)
                        stable_abi = not gil_disabled if skbuild is None else bool(skbuild)
                        components = "Development.Module"
                        if stable_abi:
                            components += ";Development.SABIModule"
                        expected = (
                            [
                                f"Python3;3.12;REQUIRED;COMPONENTS;{components}",
                                "Python;3.12;REQUIRED;COMPONENTS;Interpreter",
                            ]
                            if cross
                            else [
                                "Python;3.12;REQUIRED;COMPONENTS;Interpreter",
                                f"Python;3.12;REQUIRED;COMPONENTS;Interpreter;{components}",
                            ]
                        )
                        self.assertEqual(result["find"], expected)

    def test_cross_compile_target_abi(self):
        for abi in ("ANY;ANY;ANY", "ANY;ANY;ANY;ANY", "ANY;ANY;ANY;OFF", "ANY;ANY;ANY;on"):
            with self.subTest(abi=abi):
                result = self.configure_discovery(
                    f'set(CMAKE_CROSSCOMPILING ON)\nset(Python3_FIND_ABI "{abi}")'
                )
                self.assertEqual(result["abi"], [f"|{abi}"])
                self.assertEqual(
                    "Development.SABIModule" in result["find"][0], not abi.endswith(";on")
                )

    def test_old_cmake_without_stable_abi_support(self):
        result = self.configure_discovery("set(CMAKE_VERSION 3.25)")
        self.assertNotIn("Development.SABIModule", result["find"][-1])

    def test_missing_required_stable_abi_fails(self):
        for cross in (False, True):
            for skbuild in ("", "set(SKBUILD_SABI_COMPONENT Development.SABIModule)"):
                with self.subTest(cross=cross, skbuild=skbuild):
                    self.configure_discovery(
                        f"set(CMAKE_CROSSCOMPILING {'ON' if cross else 'OFF'})\n"
                        f"set(test_sabi_missing ON)\n{skbuild}",
                        succeeds=False,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
