import os
import re
import subprocess
from tempfile import TemporaryDirectory
import textwrap
import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestClangTidyWorkflow(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.content = (cls.root / ".github" / "workflows" / "clang_tidy.yml").read_text(
            encoding="utf-8"
        )
        cls.launcher = "${{ steps.sccache.outcome == 'success' && 'sccache' || '' }}"

    def test_uses_sccache(self):
        """Verifies that the clang-tidy build is cached with sccache."""
        self.assertIn("mozilla-actions/sccache-action", self.content)
        self.assertIn(
            "SCCACHE_GHA_ENABLED: ${{ steps.sccache.outcome == 'success' }}", self.content
        )
        self.assertIn(f"-DCMAKE_C_COMPILER_LAUNCHER={self.launcher}", self.content)
        self.assertIn(f"-DCMAKE_CXX_COMPILER_LAUNCHER={self.launcher}", self.content)

    def test_sccache_failure_only_disables_the_cache(self):
        """Verifies that failed cache setup leaves builds and checks mandatory."""
        for workflow in ("ci_core.yml", "clang_tidy.yml"):
            content = (self.root / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
            checked = 0
            for name, job in re.findall(r"(?ms)^  (\w+):\s*$(.*?)(?=^  \w+:\s*$|\Z)", content):
                if "mozilla-actions/sccache-action" not in job:
                    continue
                with self.subTest(workflow=workflow, job=name):
                    checked += 1
                    steps = re.findall(r"(?ms)^      - (.*?)(?=^      - |\Z)", job)
                    setup = next(s for s in steps if "mozilla-actions/sccache-action" in s)
                    self.assertIn("id: sccache\n", setup)
                    self.assertIn("continue-on-error: true\n", setup)
                    self.assertNotRegex(job, r"(?m)^    continue-on-error:")
                    self.assertIn("if: steps.sccache.outcome != 'success'", job)
                    self.assertIn("::warning::", job)
                    for step in steps:
                        if step != setup:
                            self.assertNotIn("continue-on-error:", step)
                        if "COMPILER_LAUNCHER" not in step:
                            continue
                        for language in ("C", "CXX"):
                            self.assertIn(
                                f"CMAKE_{language}_COMPILER_LAUNCHER={self.launcher}", step
                            )
                        self.assertNotIn("COMPILER_LAUNCHER=sccache", step)
                        self.assertNotRegex(step, r"(?m)^\s+if:.*steps\.sccache")
            self.assertGreater(checked, 0, workflow)

    def test_pull_request_scans_only_changed_files(self):
        """Verifies that pull requests only run clang-tidy on changed sources."""
        self.assertIn("git diff --name-only", self.content)
        self.assertIn("run-clang-tidy", self.content)

    @unittest.skipIf(os.name == "nt", "Executes the Linux workflow shell.")
    def test_workflow_only_change_does_not_fail_cpp_file_selection(self):
        """Accepts an empty C++ diff without hiding git failures."""
        step = self.content.split("      - name: Collect files for clang-tidy\n", 1)[1]
        script = textwrap.dedent(
            step.split("\n      - name:", 1)[0].split("        run: |\n", 1)[1]
        )
        for event in ("pull_request", "push"):
            for base in ("HEAD", "not-a-valid-git-reference"):
                with self.subTest(event=event, base=base), TemporaryDirectory() as folder:
                    rendered = script
                    for expression, value in (
                        ("github.event_name", event),
                        ("github.event.pull_request.base.sha", base),
                        ("github.event.before", "HEAD"),
                        ("github.sha", base if event == "push" else "HEAD"),
                    ):
                        rendered = rendered.replace("${{ " + expression + " }}", value)
                    output = Path(folder) / "output"
                    process = subprocess.run(
                        ["bash", "-e", "-o", "pipefail", "-c", rendered],
                        cwd=self.root,
                        env={**os.environ, "GITHUB_OUTPUT": str(output)},
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    if base == "HEAD":
                        self.assertEqual(process.returncode, 0, process.stderr)
                        self.assertEqual(output.read_text(), "files=\n")
                    else:
                        self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
