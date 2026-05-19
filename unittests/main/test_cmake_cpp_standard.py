import unittest
from pathlib import Path


class TestCMakeCppStandard(unittest.TestCase):
    def test_project_cmake_uses_cpp20(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("set(CMAKE_CXX_STANDARD 20)", content)
        self.assertIn("target_compile_features(lib_onnx_proto PUBLIC cxx_std_20)", content)
        self.assertIn("target_compile_features(lib_onnx_lib PUBLIC cxx_std_20)", content)

    def test_examples_cmake_use_cpp20(self):
        root = Path(__file__).resolve().parents[2]
        example_cmake_files = [
            root / "examples" / "load_onnx_time" / "CMakeLists.txt",
            root / "examples" / "load_onnx_light_time" / "CMakeLists.txt",
            root / "examples" / "save_onnx_light_time" / "CMakeLists.txt",
        ]
        for cmake_file in example_cmake_files:
            with self.subTest(cmake_file=cmake_file):
                content = cmake_file.read_text(encoding="utf-8")
                self.assertIn("set(CMAKE_CXX_STANDARD 20)", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
