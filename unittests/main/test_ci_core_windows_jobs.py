import re
import unittest

from pathlib import Path


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

    def test_windows_x86_build_uses_msvc_ninja_with_sccache(self):
        """Verifies that the 32-bit Windows build uses MSVC-backed Ninja with sccache."""
        match = re.search(r"(?ms)^  windows_x86_build:\s*$(.*?)(?=^  \w+:\s*$|\Z)", self.content)
        self.assertIsNotNone(match)
        job = match.group(1)
        self.assertIn("- name: Set up sccache", job)
        self.assertIn("- name: Set up x86 MSVC for Ninja", job)
        self.assertIn("uses: ilammy/msvc-dev-cmd@v1.13.0", job)
        self.assertIn("arch: x86", job)
        self.assertIn("-G Ninja", job)
        self.assertIn("-DCMAKE_C_COMPILER_LAUNCHER=sccache", job)
        self.assertIn("-DCMAKE_CXX_COMPILER_LAUNCHER=sccache", job)
        self.assertNotIn("-A Win32", job)

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
                r"run: ctest --test-dir build --output-on-failure -C Release --timeout 240"
            ),
        )

    def test_windows_ninja_is_installed_and_verified_before_configuration(self):
        """Ensures both Windows builds provide their selected generator explicitly."""
        for job_name, setup_name, build_name in (
            ("core_tests", "Install Ninja (Windows)", "Build and Install package (Windows)"),
            ("windows_x86_build", "Install CMake and Ninja", "Configure 32-bit CMake build"),
        ):
            with self.subTest(job=job_name):
                match = re.search(
                    rf"(?ms)^  {job_name}:\s*$(.*?)(?=^  \w+:\s*$|\Z)", self.content
                )
                self.assertIsNotNone(match)
                job = match.group(1)
                setup_marker = f"- name: {setup_name}\n"
                self.assertIn(setup_marker, job)
                setup = job.split(setup_marker, 1)[1].split("\n      - ", 1)[0]
                self.assertRegex(setup, r"python -m pip install[^\n]*\bninja\b")
                self.assertIn("ninja --version", setup)
                self.assertLess(
                    setup.index("python -m pip install"), setup.index("ninja --version")
                )
                self.assertLess(job.index(setup_marker), job.index(f"- name: {build_name}\n"))
                if job_name == "core_tests":
                    self.assertIn("if: runner.os == 'Windows'", setup)

    def _job_needs(self, job_name):
        """Returns the list of job names in the ``needs`` clause of ``job_name``."""
        match = re.search(
            rf"(?m)^  {re.escape(job_name)}:\s*$(.*?)(?=^  \w+:\s*$|\Z)", self.content, re.DOTALL
        )
        self.assertIsNotNone(match, job_name)
        needs_match = re.search(r"(?m)^    needs:\s*(.+)$", match.group(1))
        if needs_match is None:
            return None
        return re.findall(r"\w+", needs_match.group(1))

    def test_reduced_and_no_onnx_preflights_run_in_parallel(self):
        """Verifies that the reduced and no-onnx preflights have no interdependency."""
        self.assertIsNone(self._job_needs("reduced_tests_ubuntu"))
        self.assertIsNone(self._job_needs("no_onnx_tests_ubuntu"))

    def test_downstream_jobs_gate_on_both_preflights(self):
        """Verifies that every downstream build job waits on both preflights."""
        for job_name in ("free_threaded", "core_tests_ubuntu", "core_tests", "windows_x86_build"):
            needs = self._job_needs(job_name)
            self.assertIsNotNone(needs, job_name)
            self.assertIn("reduced_tests_ubuntu", needs, job_name)
            self.assertIn("no_onnx_tests_ubuntu", needs, job_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
