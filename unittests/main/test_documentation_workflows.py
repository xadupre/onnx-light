import unittest
from pathlib import Path


class TestDocumentationWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.workflows = {
            filename: (root / ".github" / "workflows" / filename).read_text(encoding="utf-8")
            for filename in ("docs.yml", "doc_cpp.yml")
        }

    def test_weekly_and_manual_triggers_only(self):
        """Runs documentation weekly or manually, independently of regular CI."""
        for filename, content in self.workflows.items():
            with self.subTest(workflow=filename):
                triggers = content.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
                self.assertEqual(
                    [line.strip() for line in triggers.splitlines() if line.endswith(":")],
                    ["schedule:", "workflow_dispatch:"],
                )
                cron = triggers.split('cron: "', 1)[1].split('"', 1)[0].split()
                self.assertEqual(len(cron), 5)
                self.assertTrue(0 <= int(cron[0]) <= 59)
                self.assertTrue(0 <= int(cron[1]) <= 23)
                self.assertEqual(cron[2:4], ["*", "*"])
                self.assertTrue(0 <= int(cron[4]) <= 6)
                self.assertNotIn("github.event.workflow_run", content)

    def test_reports_only_failed_builds_with_isolated_permissions(self):
        """Reports failed runs without granting issue access to documentation builds."""
        for filename, build_job in (("docs.yml", "docs"), ("doc_cpp.yml", "doc_cpp")):
            with self.subTest(workflow=filename):
                build, report = self.workflows[filename].split("\n  report_failure:", 1)
                self.assertIn("contents: read", build)
                self.assertNotIn("issues: write", build)
                self.assertIn(f"needs: {build_job}", report)
                self.assertIn(
                    f"if: ${{{{ failure() && needs.{build_job}.result == 'failure' }}}}", report
                )
                self.assertIn("issues: write", report)
                self.assertNotIn("actions/checkout", report)
                self.assertIn("GH_TOKEN: ${{ github.token }}", report)
                self.assertIn('gh issue list --repo "$GITHUB_REPOSITORY" --state all', report)
                self.assertIn('--search "${GITHUB_RUN_ID} in:title"', report)
                self.assertIn('if [ "$issue_count" -eq 0 ]; then', report)
                self.assertIn('gh issue create --repo "$GITHUB_REPOSITORY"', report)
                self.assertIn("/actions/runs/${{ github.run_id }}", report)
                self.assertIn('--title "$ISSUE_TITLE"', report)
                self.assertIn("${RUN_URL}", report)


if __name__ == "__main__":
    unittest.main()
