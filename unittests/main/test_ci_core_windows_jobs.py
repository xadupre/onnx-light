import unittest
from pathlib import Path


class TestCiCoreWindowsJobs(unittest.TestCase):
    def test_windows_core_job_includes_cpp_tests(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / ".github" / "workflows" / "ci_core.yml").read_text(encoding="utf-8")
        self.assertIn("Run C++ unit tests (Windows)", content)
        self.assertIn("cmake.define.ONNX_LIGHT_BUILD_TESTS=ON", content)

    def test_windows_x86_build_job_is_removed(self):
        root = Path(__file__).resolve().parents[2]
        content = (root / ".github" / "workflows" / "ci_core.yml").read_text(encoding="utf-8")
        self.assertNotIn("windows_x86_build:", content)
        self.assertNotIn("name: core (windows-latest, x86 build)", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
