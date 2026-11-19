import unittest
from pathlib import Path


class TestReleaseCppWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.workflow = (root / ".github/workflows/build_release_cpp.yml").read_text(
            encoding="utf-8"
        )
        cls.build_job = cls.workflow.split("\n  build_cpp_artifacts:", 1)[1].split(
            "\n  publish_cpp_release:", 1
        )[0]
        cls.publish_job = cls.workflow.split("\n  publish_cpp_release:", 1)[1]

    def test_build_uses_read_only_permissions(self):
        self.assertIn("\npermissions:\n  contents: read\n", self.workflow)
        self.assertNotIn("contents: write", self.build_job)

    def test_release_publication_uses_a_separate_write_scoped_job(self):
        self.assertIn("needs: build_cpp_artifacts", self.publish_job)
        self.assertIn("if: ${{ github.event_name == 'release' }}", self.publish_job)
        self.assertIn("contents: write", self.publish_job)
        self.assertIn("actions/download-artifact@v7", self.publish_job)
        self.assertIn('gh release upload "$TAG" release-artifacts/*', self.publish_job)


if __name__ == "__main__":
    unittest.main()
