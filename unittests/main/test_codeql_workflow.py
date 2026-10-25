import unittest
from pathlib import Path


class TestCodeQLWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.content = (root / ".github" / "workflows" / "codeql.yml").read_text(encoding="utf-8")

    def test_defers_drafts_and_restarts_when_ready(self):
        """Defers PR analysis until readiness without suppressing main or weekly scans."""
        triggers = self.content.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
        self.assertIn('push:\n    branches: ["main"]', triggers)
        self.assertIn('pull_request:\n    branches: ["main"]', triggers)
        self.assertIn(
            "types: [opened, reopened, synchronize, ready_for_review, converted_to_draft]",
            triggers,
        )
        self.assertIn(
            "if: github.event_name != 'pull_request' || !github.event.pull_request.draft",
            self.content.split("\n  analyze:", 1)[1].split("\n    steps:", 1)[0],
        )
        self.assertIn("workflow_dispatch:", triggers)
        cron = triggers.split('cron: "', 1)[1].split('"', 1)[0].split()
        self.assertEqual(len(cron), 5)
        self.assertEqual(cron[2:4], ["*", "*"])
        self.assertTrue(0 <= int(cron[4]) <= 6)
        self.assertNotIn("pull_request_target:", triggers)
        self.assertIn("group: ${{ github.workflow }}-${{ github.ref }}", self.content)
        self.assertIn("cancel-in-progress: true", self.content)

    def test_preserves_security_alert_uploads_and_language_coverage(self):
        """Keeps all existing languages and CodeQL security-alert publication enabled."""
        self.assertIn("language: [actions, c-cpp, python]", self.content)
        self.assertIn("languages: ${{ matrix.language }}", self.content)
        self.assertIn("build-mode: none", self.content)
        self.assertIn("security-events: write", self.content)
        self.assertIn("github/codeql-action/init@v4", self.content)
        self.assertIn("github/codeql-action/analyze@v4", self.content)
        self.assertIn("category: /language:${{ matrix.language }}", self.content)
        self.assertNotIn("upload: false", self.content)
        self.assertNotIn("upload: never", self.content)
        self.assertNotIn("continue-on-error:", self.content)


if __name__ == "__main__":
    unittest.main()
