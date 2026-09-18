import unittest
from pathlib import Path


class TestKernelReportsWorkflow(unittest.TestCase):
    def test_initializes_cache_from_runner_environment(self):
        """Keeps runner-dependent cache initialization inside a workflow step."""
        root = Path(__file__).resolve().parents[2]
        content = (root / ".github" / "workflows" / "kernel_reports.yml").read_text(
            encoding="utf-8"
        )
        job_configuration, steps = content.split("    steps:", 1)
        self.assertNotIn("${{ runner.", job_configuration)
        self.assertIn(
            'echo "XDG_CACHE_HOME=$RUNNER_TEMP/kernel-reports-cache" >> "$GITHUB_ENV"', steps
        )
        self.assertLess(steps.index("XDG_CACHE_HOME="), steps.index("python setup.py"))


if __name__ == "__main__":
    unittest.main()
