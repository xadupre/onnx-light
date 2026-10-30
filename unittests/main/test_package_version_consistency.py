import re
import unittest

from pathlib import Path


class TestPackageVersionConsistency(unittest.TestCase):
    """Checks package version consistency across release metadata files."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]

    @staticmethod
    def _extract(pattern: str, content: str, source: str) -> str:
        match = re.search(pattern, content, flags=re.MULTILINE)
        if match is None:
            raise AssertionError(f"Unable to extract version from {source!r}.")
        return match.group(1)

    def test_documentation_homepage_displays_release(self):
        """Checks that the homepage displays the release from the package version."""
        homepage = (self.root / "docs" / "index.rst").read_text(encoding="utf-8")
        config = (self.root / "docs" / "conf.py").read_text(encoding="utf-8")

        self.assertIn("\nVersion: |release|\n", homepage)
        self.assertRegex(config, r"(?m)^release\s*=\s*onnx_light\.__version__\s*$")

    def test_package_versions_are_synchronized(self):
        """Verifies Python and C++ package version declarations remain aligned."""
        pyproject = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        package_init = (self.root / "onnx_light" / "__init__.py").read_text(encoding="utf-8")
        setup_py = (self.root / "setup.py").read_text(encoding="utf-8")
        cpp_version = (self.root / "onnx_light" / "_version.cc").read_text(encoding="utf-8")
        cmake = (self.root / "CMakeLists.txt").read_text(encoding="utf-8")
        proto_cmake = (self.root / "onnx_proto" / "CMakeLists.txt").read_text(encoding="utf-8")
        changelog = (self.root / "CHANGELOGS.md").read_text(encoding="utf-8")

        pyproject_version = self._extract(
            r'^version\s*=\s*"([^"]+)"', pyproject, "pyproject.toml"
        )
        package_init_version = self._extract(
            r'^__version__\s*=\s*"([^"]+)"', package_init, "onnx_light/__init__.py"
        )
        setup_py_version = self._extract(r'version\s*=\s*"([^"]+)"', setup_py, "setup.py")
        cpp_version_value = self._extract(
            r'kVersion\s*=\s*"([^"]+)"', cpp_version, "onnx_light/_version.cc"
        )
        cpp_documented_version = self._extract(
            r'@return Version string "([^"]+)"', cpp_version, "onnx_light/_version.cc"
        )
        cmake_version = self._extract(
            r"project\(onnx_light VERSION ([0-9]+\.[0-9]+\.[0-9]+)", cmake, "CMakeLists.txt"
        )
        proto_cmake_version = self._extract(
            r"project\(onnx_proto VERSION ([0-9]+\.[0-9]+\.[0-9]+)",
            proto_cmake,
            "onnx_proto/CMakeLists.txt",
        )
        changelog_versions = re.findall(r"^## \[([^\]]+)\]", changelog, flags=re.MULTILINE)

        self.assertEqual(pyproject_version, package_init_version)
        self.assertEqual(pyproject_version, setup_py_version)
        self.assertEqual(pyproject_version, cpp_version_value)
        self.assertEqual(pyproject_version, cpp_documented_version)
        self.assertEqual(pyproject_version, cmake_version)
        self.assertEqual(pyproject_version, proto_cmake_version)
        self.assertIn(pyproject_version, changelog_versions)

    def test_changelog_documents_current_version_bump(self):
        """Verifies the current version's changelog entry documents its version bump."""
        pyproject = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        version = self._extract(r'^version\s*=\s*"([^"]+)"', pyproject, "pyproject.toml")
        changelog = (self.root / "CHANGELOGS.md").read_text(encoding="utf-8")
        entries = re.split(r"^## ", changelog, flags=re.MULTILINE)[1:]
        current_entry = next(entry for entry in entries if entry.startswith(f"[{version}]"))
        self.assertIn(f"Bumped the release version to `{version}`.", current_entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
