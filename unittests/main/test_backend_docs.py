import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestBackendDocs(ExtTestCase):
    def test_backend_tests_design_page_exists(self):
        design_page = (
            Path(__file__).resolve().parents[2] / "docs" / "design" / "backend_tests.rst"
        )
        self.assertTrue(design_page.exists())
        content = design_page.read_text(encoding="utf-8")
        self.assertIn("make_test_class", content)
        self.assertIn("my_runtime", content)
        self.assertIn("include_regex", content)
        self.assertIn("exclude_regex", content)

    def test_backend_tests_in_design_index(self):
        design_index = Path(__file__).resolve().parents[2] / "docs" / "design" / "index.rst"
        content = design_index.read_text(encoding="utf-8")
        self.assertIn("backend_tests", content)

    def test_backend_api_label_exists(self):
        docs_root = Path(__file__).resolve().parents[2] / "docs"
        backend_api_index = docs_root / "api" / "python" / "backend" / "index.rst"
        backend_design_page = docs_root / "design" / "backend_tests.rst"

        backend_api_content = backend_api_index.read_text(encoding="utf-8")
        backend_design_content = backend_design_page.read_text(encoding="utf-8")

        self.assertIn(".. _l-api-backend:", backend_api_content)
        self.assertIn(":ref:`l-api-backend`", backend_design_content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
