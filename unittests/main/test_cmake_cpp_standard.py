import unittest
from pathlib import Path


class TestCMakeCppStandard(unittest.TestCase):
    def test_project_cmake_uses_cpp20(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("set(CMAKE_CXX_STANDARD 20)", content)
        self.assertIn("target_compile_features(lib_onnx_proto PUBLIC cxx_std_20)", content)
        self.assertIn("target_compile_features(lib_onnx_lib PUBLIC cxx_std_20)", content)
        self.assertRegex(
            content,
            (
                r"(?s)if\(MSVC\)\s*"
                r"if\(NOT CMAKE_GENERATOR MATCHES \"Ninja\"\)\s*"
                r"add_compile_options\(\$<\$<COMPILE_LANGUAGE:C,CXX>:/MP>\)\s*"
                r"endif\(\)\s*"
                r"if\(ONNX_LIGHT_WERROR\)\s*"
                r"add_compile_options\(\$<\$<COMPILE_LANGUAGE:C,CXX>:/WX>\)\s*"
                r"endif\(\)\s*"
                r"endif\(\)"
            ),
            msg=(
                "MSVC /MP must be guarded by a Ninja check; /WX must remain "
                "in the outer MSVC block, guarded by ONNX_LIGHT_WERROR."
            ),
        )

    def test_project_cmake_treats_warnings_as_errors(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('option(ONNX_LIGHT_WERROR\n', content)
        self.assertRegex(
            content,
            (
                r"(?s)if\(NOT MSVC\).*?"
                r"add_compile_options\(\$<\$<COMPILE_LANGUAGE:C,CXX>:-Wall>\)\s*"
                r"add_compile_options\(\$<\$<COMPILE_LANGUAGE:C,CXX>:-Wextra>\)\s*"
                r"if\(ONNX_LIGHT_WERROR\)\s*"
                r"add_compile_options\(\$<\$<COMPILE_LANGUAGE:C,CXX>:-Werror>\)\s*"
                r"endif\(\)\s*"
                r"endif\(\)"
            ),
            msg="GCC/Clang builds must add -Werror when ONNX_LIGHT_WERROR is ON.",
        )
        self.assertIn("function(onnx_light_disable_werror)", content)
        self.assertIn("onnx_light_disable_werror(gtest gtest_main gmock gmock_main)", content)

    def test_examples_cmake_use_cpp20(self):
        root = Path(__file__).resolve().parents[2]
        example_cmake_files = [
            root / "examples" / "load_onnx_time" / "CMakeLists.txt",
            root / "examples" / "load_onnx_light_time" / "CMakeLists.txt",
            root / "examples" / "save_onnx_light_time" / "CMakeLists.txt",
            root / "examples" / "build_save_load_onnx_proto" / "CMakeLists.txt",
        ]
        for cmake_file in example_cmake_files:
            with self.subTest(cmake_file=cmake_file):
                content = cmake_file.read_text(encoding="utf-8")
                self.assertIn("set(CMAKE_CXX_STANDARD 20)", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
