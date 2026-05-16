import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestBackendDocs(ExtTestCase):
    def test_api_index_references_backend(self):
        """Validates that the API index links the backend documentation."""
        api_index = Path(__file__).resolve().parents[2] / "docs" / "api" / "index.rst"
        content = api_index.read_text(encoding="utf-8")
        self.assertIn("backend/index", content)

    def test_backend_index_lists_random_module(self):
        """Validates that the backend API index lists the random submodule."""
        backend_index = (
            Path(__file__).resolve().parents[2] / "docs" / "api" / "backend" / "index.rst"
        )
        content = backend_index.read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.backend", content)
        self.assertIn("random", content)

    def test_backend_random_page_exists(self):
        """Validates that the backend random API page exists and is wired to automodule."""
        random_page = (
            Path(__file__).resolve().parents[2] / "docs" / "api" / "backend" / "random.rst"
        )
        content = random_page.read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.backend.random", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
