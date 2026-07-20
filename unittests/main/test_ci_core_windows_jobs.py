import unittest
from pathlib import Path

import yaml


class TestCiCoreWindowsJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.content = (cls.root / ".github" / "workflows" / "ci_core.yml").read_text(
            encoding="utf-8"
        )

    def test_windows_x86_build_job_exists(self):
        """Verifies that the dedicated Windows x86 build job remains present."""
        self.assertIn("windows_x86_build:", self.content)
        self.assertIn("name: core (windows-latest, x86 build)", self.content)

    def test_windows_64_build_uses_msvc_ninja_with_sccache(self):
        """Verifies that the 64-bit Windows build uses MSVC-backed Ninja with sccache."""
        self.assertIn("- name: Set up MSVC for Ninja (Windows)", self.content)
        self.assertIn("uses: ilammy/msvc-dev-cmd@v1.13.0", self.content)
        self.assertIn("arch: x64", self.content)
        self.assertRegex(
            self.content,
            r"(?s)- name: Set up MSVC for Ninja \(Windows\).*?if: runner\.os == 'Windows'",
        )
        self.assertIn("- name: Build and Install package (Windows)", self.content)
        self.assertIn("CMAKE_GENERATOR: Ninja", self.content)
        self.assertIn('SCCACHE_GHA_ENABLED: "true"', self.content)
        self.assertIn("cmake.define.CMAKE_C_COMPILER_LAUNCHER=sccache", self.content)
        self.assertIn("cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=sccache", self.content)

    def test_windows_cpp_tests_are_enabled(self):
        """Verifies that the 64-bit Windows job still enables and runs C++ tests."""
        self.assertRegex(
            self.content,
            (
                r"(?s)- name: Build and Install package \(Windows\).*?"
                r"cmake\.define\.ONNX_LIGHT_BUILD_TESTS=ON.*?"
                r"- name: Run C\+\+ unit tests \(Windows\)\s+"
                r"if: runner\.os == 'Windows'\s+"
                r"run: ctest --test-dir build --output-on-failure -C Release --timeout 120"
            ),
        )

    def test_reduced_and_no_onnx_preflights_run_in_parallel(self):
        """Verifies that the reduced and no-onnx preflights have no interdependency."""
        jobs = yaml.safe_load(self.content)["jobs"]
        self.assertIsNone(jobs["reduced_tests_ubuntu"].get("needs"))
        self.assertIsNone(jobs["no_onnx_tests_ubuntu"].get("needs"))

    def test_downstream_jobs_gate_on_both_preflights(self):
        """Verifies that every downstream build job waits on both preflights."""
        jobs = yaml.safe_load(self.content)["jobs"]
        for job_name in ("core_tests_ubuntu", "core_tests", "windows_x86_build"):
            needs = jobs[job_name].get("needs")
            self.assertIn("reduced_tests_ubuntu", needs, job_name)
            self.assertIn("no_onnx_tests_ubuntu", needs, job_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
