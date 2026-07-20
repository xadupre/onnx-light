import unittest
from pathlib import Path


class TestCodeqlWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.path = cls.root / ".github" / "workflows" / "codeql.yml"

    def test_codeql_workflow_exists(self):
        """Verifies that the advanced CodeQL workflow is present."""
        self.assertTrue(self.path.exists(), "codeql.yml workflow is missing")

    def test_cpp_uses_build_mode_none(self):
        """Verifies C/C++ is analyzed with build-mode none to avoid the slow dynamic build."""
        import yaml

        data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        includes = data["jobs"]["analyze"]["strategy"]["matrix"]["include"]
        by_language = {entry["language"]: entry for entry in includes}
        self.assertIn("c-cpp", by_language)
        self.assertEqual(by_language["c-cpp"]["build-mode"], "none")

    def test_uses_codeql_action_v3(self):
        """Verifies the workflow drives CodeQL via the official action init/analyze steps."""
        content = self.path.read_text(encoding="utf-8")
        self.assertIn("github/codeql-action/init@v3", content)
        self.assertIn("github/codeql-action/analyze@v3", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
