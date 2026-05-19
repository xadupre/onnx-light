"""Tests that the Python API documentation RST files reflect the current subfolder hierarchy."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = ROOT / "docs" / "api" / "python"
PROTOS_RST = ROOT / "docs" / "api" / "protos.rst"


class TestPythonApiDocHierarchy(unittest.TestCase):
    def _rst_files(self):
        """Returns all .rst files under docs/api/python/ (recursively)."""
        return list(PYTHON_API_DIR.rglob("*.rst"))

    def test_no_rst_references_onnx_shim(self):
        """Verifies that no Python API RST file uses the onnx backward-compat shim namespace."""
        shim_prefix = "onnx_light.onnx."
        violations = []
        for path in self._rst_files():
            content = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(content.splitlines(), start=1):
                if shim_prefix in line and "automodule" in line or (
                    shim_prefix in line and "autoclass" in line
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
        self.assertFalse(
            violations,
            "The following RST lines still reference the backward-compat shim "
            "'onnx_light.onnx.*' instead of 'onnx_light.onnx_lib.*':\n"
            + "\n".join(violations),
        )

    def test_index_references_onnx_lib(self):
        """Verifies that the Python API index automodule points to onnx_light.onnx_lib."""
        content = (PYTHON_API_DIR / "index.rst").read_text(encoding="utf-8")
        self.assertIn("automodule:: onnx_light.onnx_lib", content)
        self.assertNotIn("automodule:: onnx_light.onnx\n", content)

    def test_inliner_rst_exists(self):
        """Verifies that a dedicated RST page exists for onnx_light.onnx_lib.inliner."""
        inliner_rst = PYTHON_API_DIR / "inliner.rst"
        self.assertTrue(inliner_rst.exists(), "docs/api/python/inliner.rst must exist")
        content = inliner_rst.read_text(encoding="utf-8")
        self.assertIn("onnx_light.onnx_lib.inliner", content)

    def test_version_converter_rst_exists(self):
        """Verifies that a dedicated RST page exists for onnx_light.onnx_lib.version_converter."""
        vc_rst = PYTHON_API_DIR / "version_converter.rst"
        self.assertTrue(vc_rst.exists(), "docs/api/python/version_converter.rst must exist")
        content = vc_rst.read_text(encoding="utf-8")
        self.assertIn("onnx_light.onnx_lib.version_converter", content)

    def test_index_toctree_includes_new_modules(self):
        """Verifies that inliner and version_converter are listed in the Python API index toctree."""
        content = (PYTHON_API_DIR / "index.rst").read_text(encoding="utf-8")
        self.assertIn("inliner", content)
        self.assertIn("version_converter", content)

    def test_protos_rst_references_onnx_lib(self):
        """Verifies that protos.rst uses onnx_light.onnx_lib (not the shim) for autoclass directives."""
        content = PROTOS_RST.read_text(encoding="utf-8")
        self.assertIn("onnx_light.onnx_lib.", content)
        self.assertNotIn("autoclass:: onnx_light.onnx.", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
