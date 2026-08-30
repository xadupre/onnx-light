import re
import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestPackageVersionConsistency(ExtTestCase):
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

    def test_package_versions_are_synchronized(self):
        """Verifies Python and C++ package version declarations remain aligned."""
        pyproject = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        package_init = (self.root / "onnx_light" / "__init__.py").read_text(encoding="utf-8")
        setup_py = (self.root / "setup.py").read_text(encoding="utf-8")
        cpp_version = (self.root / "onnx_light" / "_version.cc").read_text(encoding="utf-8")
        cmake = (self.root / "CMakeLists.txt").read_text(encoding="utf-8")

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
        cmake_version = self._extract(
            r"project\(onnx_light VERSION ([0-9]+\.[0-9]+\.[0-9]+)", cmake, "CMakeLists.txt"
        )

        self.assertEqual(pyproject_version, package_init_version)
        self.assertEqual(pyproject_version, setup_py_version)
        self.assertEqual(pyproject_version, cpp_version_value)
        self.assertEqual(pyproject_version, cmake_version)


if __name__ == "__main__":
    unittest.main(verbosity=2)
