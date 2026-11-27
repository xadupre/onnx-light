import re
import unittest
from pathlib import Path


class TestScorecardWorkflow(unittest.TestCase):
    def test_runs_weekly_and_manually(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github/workflows/cq_scorecard.yml").read_text(encoding="utf-8")
        triggers = workflow.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
        self.assertEqual(
            [line.strip() for line in triggers.splitlines() if line.endswith(":")],
            ["schedule:", "workflow_dispatch:"],
        )
        schedules = re.findall(r'^\s*-\s*cron:\s*"([^"]+)"\s*$', triggers, re.MULTILINE)
        self.assertEqual(len(schedules), 1)
        cron = schedules[0].split()
        self.assertEqual(len(cron), 5)
        self.assertTrue(0 <= int(cron[0]) <= 59)
        self.assertTrue(0 <= int(cron[1]) <= 23)
        self.assertEqual(cron[2:4], ["*", "*"])
        self.assertTrue(0 <= int(cron[4]) <= 6)


if __name__ == "__main__":
    unittest.main()
