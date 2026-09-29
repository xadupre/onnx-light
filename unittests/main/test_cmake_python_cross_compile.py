import shutil
import subprocess
import tempfile

from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestCMakePythonCrossCompile(ExtTestCase):
    def _configure(self, abi):
        """Runs the production Python discovery block in cross-compiling mode."""
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake is unavailable")

        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        start = content.index("if(ONNX_LIGHT_BUILD_PYTHON)\n")
        end = content.index("  # Find nanobind", start)
        discovery = content[start:end] + "endif()\n"
        project = (
            """cmake_minimum_required(VERSION 3.15)
project(PythonDiscovery NONE)
set(ONNX_LIGHT_BUILD_PYTHON ON)
set(CMAKE_CROSSCOMPILING ON)
set(Python3_FIND_ABI "@ABI@")
macro(find_package package)
  message(STATUS "test_find=${ARGV}")
  foreach(component ${ARGN})
    if(component STREQUAL "Development.Module")
      add_library(${package}::Module INTERFACE IMPORTED)
    elseif(component STREQUAL "Development.SABIModule")
      add_library(${package}::SABIModule INTERFACE IMPORTED)
    endif()
  endforeach()
endmacro()
""".replace("@ABI@", abi)
            + discovery
            + """
get_target_property(module_alias Python::Module ALIASED_TARGET)
if(TARGET Python::SABIModule)
  get_target_property(sabi_alias Python::SABIModule ALIASED_TARGET)
else()
  set(sabi_alias "")
endif()
message(STATUS "test_module_alias=${module_alias}")
message(STATUS "test_sabi_alias=${sabi_alias}")
"""
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "CMakeLists.txt").write_text(project, encoding="utf-8")
            result = subprocess.run(
                [cmake, "-S", str(source), "-B", str(source / "build")],
                capture_output=True,
                check=False,
                text=True,
            )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        values = {}
        for line in result.stdout.splitlines():
            if line.startswith("-- test_"):
                key, value = line.removeprefix("-- test_").split("=", 1)
                values.setdefault(key, []).append(value)
        return values

    def test_cross_compile_target_abi_components_and_aliases(self):
        for gil_disabled in (False, True):
            abi = f"ANY;ANY;ANY;{'ON' if gil_disabled else 'OFF'}"
            with self.subTest(abi=abi):
                values = self._configure(abi)
                components = "Development.Module"
                if not gil_disabled:
                    components += ";Development.SABIModule"
                self.assertEqual(
                    values["find"],
                    [
                        f"Python3;3.12;REQUIRED;COMPONENTS;{components}",
                        "Python;3.12;REQUIRED;COMPONENTS;Interpreter",
                    ],
                )
                self.assertEqual(values["module_alias"], ["Python3::Module"])
                self.assertEqual(
                    values["sabi_alias"], ["" if gil_disabled else "Python3::SABIModule"]
                )


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
