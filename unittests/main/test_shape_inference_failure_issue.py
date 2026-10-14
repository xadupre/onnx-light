import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class TestShapeInferenceFailureIssue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = (
            Path(__file__).resolve().parents[2]
            / ".github"
            / "workflows"
            / "shape_inference_failure_issue.yml"
        ).read_text(encoding="utf-8")
        cls.script = textwrap.dedent(workflow.split("        run: |\n", 1)[1])

    def test_reports_only_new_failures(self):
        """Verifies that completed failures create one issue per run."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gh = root / "gh"
            gh.write_text(
                "#!/bin/bash\n"
                'printf "%s\\n" "$*" >> "$CALLS"\n'
                'if [ "$1" = api ]; then\n'
                '  [ "$API_OK" = 1 ] || exit 1\n'
                '  if [ "$HAS_FAILURE" = 1 ]; then printf "38281279788\\t'
                "https://github.com/xadupre/xadupre.github.io/actions/runs/38281279788\\n"
                '"; fi\n'
                'elif [ "$1" = issue ] && [ "$2" = list ]; then\n'
                '  printf "%s\\n" "$ISSUE_COUNT"\n'
                "fi\n",
                encoding="utf-8",
            )
            gh.chmod(0o755)
            calls = root / "calls"
            for has_failure, issue_count, api_ok, expected_creates in (
                ("0", "0", "1", 0),
                ("1", "0", "1", 1),
                ("1", "1", "1", 0),
                ("1", "0", "0", 0),
            ):
                calls.write_text("", encoding="utf-8")
                result = subprocess.run(
                    ["bash", "-c", self.script],
                    env={
                        **os.environ,
                        "PATH": f"{root}:{os.environ['PATH']}",
                        "GITHUB_REPOSITORY": "xadupre/onnx-light",
                        "CALLS": str(calls),
                        "HAS_FAILURE": has_failure,
                        "ISSUE_COUNT": issue_count,
                        "API_OK": api_ok,
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0 if api_ok == "1" else 1, result.stderr)
                lines = calls.read_text(encoding="utf-8").splitlines()
                self.assertIn("status=failure", lines[0])
                created = [line for line in lines if line.startswith("issue create ")]
                self.assertEqual(len(created), expected_creates)
                if created:
                    self.assertIn("38281279788", created[0])
                    self.assertIn("xadupre/onnx-light", created[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
