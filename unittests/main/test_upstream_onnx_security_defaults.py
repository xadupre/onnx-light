import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestUpstreamOnnxSecurityDefaults(ExtTestCase):
    """Checks secure upstream ONNX defaults after the Gemm advisory."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]

    def test_pyproject_requires_patched_onnx(self):
        """Requires patched upstream onnx in optional dependency sets."""
        content = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertEqual(content.count('"onnx>=1.21.0"'), 2)
        self.assertNotIn('"onnx>=1.17"', content)

    def test_load_onnx_time_defaults_to_patched_tag(self):
        """Uses ONNX 1.21.0 as the fallback tag in the standalone example."""
        content = (self.root / "examples" / "load_onnx_time" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn('set(ONNX_GIT_TAG "v1.21.0")', content)
        self.assertNotIn("v1.17.0", content)

    def test_load_onnx_time_docs_and_scripts_reference_patched_tag(self):
        """Keeps helper docs and scripts aligned on the patched ONNX tag."""
        for relative_path in (
            "examples/load_onnx_time/build.sh",
            "examples/load_onnx_time/build.bat",
            "docs/examples_cc/load_onnx_time_example.rst",
        ):
            with self.subTest(relative_path=relative_path):
                content = (self.root / relative_path).read_text(encoding="utf-8")
                self.assertIn("v1.21.0", content)
                self.assertNotIn("v1.17.0", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
