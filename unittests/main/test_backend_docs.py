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
        self.assertIn("test_case_node/index", content)

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
            / "test_case_node"
            / "index.rst"
        )
        content = node_page.read_text(encoding="utf-8")
        self.assertIn(".. automodule:: onnx_light.backend.test.case.node", content)
        self.assertIn("abs", content)
        self.assertIn("blackmanwindow", content)

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
