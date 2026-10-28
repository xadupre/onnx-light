import re
import tomllib
import unittest
from pathlib import Path


class TestCMakeVersion(unittest.TestCase):
    root = Path(__file__).resolve().parents[2]

    def test_build_entry_points_require_cmake_4(self):
        paths = [
            self.root / "CMakeLists.txt",
            *sorted((self.root / "examples").glob("*/CMakeLists.txt")),
            self.root / "onnx_proto/CMakeLists.txt",
            self.root / "onnx_proto/tests/consumer/CMakeLists.txt",
            self.root / "onnx_threads/CMakeLists.txt",
            self.root / "onnx_threads/tests/consumer/CMakeLists.txt",
            self.root / "examples/load_onnx_time/build.sh",
        ]
        for path in paths:
            with self.subTest(path=path):
                content = path.read_text(encoding="utf-8")
                self.assertIn("cmake_minimum_required(VERSION 4.0)", content)

    def test_packaging_requires_cmake_4(self):
        project = tomllib.loads((self.root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["tool"]["scikit-build"]["cmake"]["version"], ">=4.0")
        self.assertIn("cmake>=4.0", project["project"]["optional-dependencies"]["dev-no-onnx"])
        pixi = tomllib.loads((self.root / "pixi.toml").read_text(encoding="utf-8"))
        self.assertEqual(pixi["dependencies"]["cmake"], ">=4.0")

    def test_ci_only_builds_with_cmake_4_or_newer(self):
        workflows = self.root / ".github/workflows"
        core = (workflows / "ci_core.yml").read_text(encoding="utf-8")
        matrix = re.search(r"cmake: \[([^\]]+)\]", core)
        self.assertIsNotNone(matrix)
        versions = re.findall(r'"(\d+)\.\d+\.\d+"', matrix[1])
        self.assertTrue(versions)
        self.assertTrue(all(int(major) >= 4 for major in versions))
        for name in (
            "ci_core.yml",
            "build_release_cpp.yml",
            "build_standalone_example.yml",
            "clang_tidy.yml",
            "cq_fuzz.yml",
            "kernel_reports.yml",
        ):
            with self.subTest(workflow=name):
                self.assertIn('"cmake>=4.0"', (workflows / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
