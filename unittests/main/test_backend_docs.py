import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestBackendDocs(ExtTestCase):
    def test_api_index_references_backend(self):
        api_index = Path(__file__).resolve().parents[2] / "docs" / "api" / "index.rst"
        content = api_index.read_text(encoding="utf-8")
        self.assertIn("backend/index", content)

    def test_backend_index_lists_random_module(self):
        backend_index = (
            Path(__file__).resolve().parents[2] / "docs" / "api" / "backend" / "index.rst"
        )
        content = backend_index.read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.backend", content)
        self.assertIn("random", content)
        self.assertIn("test_case_node", content)

    def test_backend_random_page_exists(self):
        random_page = (
            Path(__file__).resolve().parents[2] / "docs" / "api" / "backend" / "random.rst"
        )
        content = random_page.read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.backend.random", content)

    def test_backend_test_case_node_page_exists(self):
        node_page = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "api"
            / "backend"
            / "test_case_node.rst"
        )
        content = node_page.read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.backend.test.case.node", content)
        self.assertIn("test_case_node/abs", content)
        self.assertIn("test_case_node/blackmanwindow", content)

    def test_backend_test_case_node_abs_page_exists(self):
        abs_page = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "api"
            / "backend"
            / "test_case_node"
            / "abs.rst"
        )
        self.assertTrue(abs_page.exists())
        content = abs_page.read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.backend.test.case.node.abs", content)

    def test_backend_test_case_node_blackmanwindow_page_exists(self):
        blackmanwindow_page = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "api"
            / "backend"
            / "test_case_node"
            / "blackmanwindow.rst"
        )
        self.assertTrue(blackmanwindow_page.exists())
        content = blackmanwindow_page.read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.backend.test.case.node.blackmanwindow", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
