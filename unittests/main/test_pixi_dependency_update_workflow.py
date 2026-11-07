import unittest
from pathlib import Path


class TestPixiDependencyUpdateWorkflow(unittest.TestCase):
    def test_schedules_targeted_lockfile_updates_and_opens_a_pull_request(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github" / "workflows" / "update-pixi-dependencies.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("schedule:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("pixi update --no-install cmake openssl", workflow)
        self.assertIn("git add pixi.lock", workflow)
        self.assertIn("pull-requests: write", workflow)
        self.assertIn("gh pr create", workflow)
        self.assertIn("github.event.repository.default_branch", workflow)


if __name__ == "__main__":
    unittest.main()
