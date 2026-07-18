import unittest
from pathlib import Path


class TestCiCoreWindowsJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.content = (cls.root / ".github" / "workflows" / "ci_core.yml").read_text(
            encoding="utf-8"
        )

    def test_windows_x86_build_job_is_kept(self):
        self.assertIn("windows_x86_build:", self.content)
        self.assertIn("name: core (windows-latest, x86 build)", self.content)

    def test_windows_64_build_uses_ninja_and_sccache(self):
        self.assertRegex(
            self.content,
            (
                r"(?s)name: core \(\$\{\{ matrix\.os \}\}, py\$\{\{ matrix\.python-version \}\}\)"
                r".*?env:\s+SCCACHE_GHA_ENABLED: \"true\""
                r".*?- name: Build and Install package \(Windows\)\s+"
                r"if: runner\.os == 'Windows'\s+env:\s+CMAKE_GENERATOR: Ninja\s+"
                r"run: pip install .*?cmake\.define\.CMAKE_C_COMPILER_LAUNCHER=sccache"
                r".*?cmake\.define\.CMAKE_CXX_COMPILER_LAUNCHER=sccache"
            ),
        )

    def test_windows_cpp_tests_remain_enabled(self):
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
