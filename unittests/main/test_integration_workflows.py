import unittest
from pathlib import Path


class TestIntegrationWorkflows(unittest.TestCase):
    def test_ir_py_checks_removed(self):
        """Keeps ir-py as an mbext dependency without running its test suite."""
        root = Path(__file__).resolve().parents[2]
        workflows = root / ".github" / "workflows"
        self.assertFalse((workflows / "int_ir_py.yml").exists())
        for workflow in workflows.glob("*.yml"):
            with self.subTest(workflow=workflow.name):
                self.assertNotIn(
                    "working-directory: ir-py\n        run: pytest", workflow.read_text()
                )

        mbext = (workflows / "int_mbext.yml").read_text(encoding="utf-8")
        self.assertIn("repository: onnx/ir-py", mbext)
        self.assertIn("Install ir-py (without pulling in onnx)", mbext)
        self.assertIn(
            "working-directory: mbext\n        run: pytest --durations=10 tests/fast", mbext
        )
        self.assertIn('USE_ONNX_LIGHT: "1"', mbext)
        self.assertIn("contents: read", mbext)

    def test_removed_workflow_badges(self):
        """Omits badges linking to the removed ir-py workflow."""
        root = Path(__file__).resolve().parents[2]
        for filename in ("README.md", "docs/index.rst"):
            with self.subTest(filename=filename):
                self.assertNotIn("int_ir_py.yml", (root / filename).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
