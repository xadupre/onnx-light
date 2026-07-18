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

    def test_windows_64_build_sccache_no_ninja(self):
        """Verifies that the 64-bit Windows build uses sccache without forcing Ninja."""
        self.assertRegex(
            self.content,
            (
                r"(?s)- name: Build and Install package \(Windows\)\s+"
                r"if: runner\.os == 'Windows'\s+env:\s+SCCACHE_GHA_ENABLED: \"true\"\s+"
                r"run: pip install "
            ),
        )
        self.assertNotIn("CMAKE_GENERATOR: Ninja", self.content)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
