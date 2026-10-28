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

    def test_ci_installs_persistent_cmake_before_package_builds(self):
        workflows = self.root / ".github/workflows"
        for name in (
            "ci_core.yml",
            "ci_debug.yml",
            "coverage.yml",
            "cq_asan_ubsan.yml",
            "cq_hardening.yml",
        ):
            content = (workflows / name).read_text(encoding="utf-8")
            jobs = re.split(r"^  [\w-]+:\s*$", content, flags=re.MULTILINE)[1:]
            builds_checked = 0
            for job in jobs:
                steps = re.split(r"^      - ", job, flags=re.MULTILINE)[1:]
                cmake_installed = False
                for step in steps:
                    if 'run: python -m pip install "cmake>=4.0"' in step:
                        cmake_installed = True
                    if "pip install" in step and "-e .[dev]" in step:
                        with self.subTest(workflow=name, step=step.splitlines()[0]):
                            self.assertTrue(
                                cmake_installed,
                                "CTest discovery must not depend on pip's temporary CMake.",
                            )
                        builds_checked += 1
            self.assertGreater(builds_checked, 0, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
