import unittest
from pathlib import Path

import yaml


class TestCiOrchestrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.path = cls.root / ".github" / "workflows" / "ci_orchestrator.yml"
        cls.content = cls.path.read_text(encoding="utf-8")
        cls.data = yaml.safe_load(cls.content)

    # ── basic structure ──────────────────────────────────────────────────────

    def test_file_exists(self):
        """Verifies that the orchestrator workflow file exists."""
        self.assertTrue(self.path.exists())

    def test_workflow_name(self):
        """Verifies that the workflow is named ci-orchestrator."""
        self.assertEqual(self.data["name"], "ci-orchestrator")

    def test_triggers_on_push_and_pull_request(self):
        """Verifies push/pull_request triggers on main.

        PyYAML parses the bare 'on' YAML key as the boolean True.
        """
        on = self.data[True]
        self.assertIn("push", on)
        self.assertIn("pull_request", on)
        self.assertIn("main", on["push"]["branches"])
        self.assertIn("main", on["pull_request"]["branches"])

    # ── Phase 1: fast checks call existing reusable workflows ────────────────

    def test_fast_phase_jobs_exist(self):
        """Verifies that all fast-phase jobs are present in the orchestrator."""
        jobs = self.data["jobs"]
        for job in (
            "fast_clang_format",
            "fast_style",
            "fast_typing",
            "fast_spelling",
            "fast_codeql",
        ):
            self.assertIn(job, jobs, f"Fast-phase job '{job}' is missing")

    def test_fast_phase_jobs_use_existing_workflows(self):
        """Verifies that fast-phase jobs call existing reusable workflows via uses:."""
        jobs = self.data["jobs"]
        expected = {
            "fast_clang_format": "./.github/workflows/clang_format.yml",
            "fast_style": "./.github/workflows/style.yml",
            "fast_typing": "./.github/workflows/typing.yml",
            "fast_spelling": "./.github/workflows/spelling.yml",
            "fast_codeql": "./.github/workflows/codeql.yml",
        }
        for job, expected_uses in expected.items():
            self.assertEqual(
                jobs[job]["uses"],
                expected_uses,
                f"Job '{job}' must use '{expected_uses}'",
            )

    def test_fast_phase_jobs_have_no_needs(self):
        """Verifies that fast-phase jobs have no dependencies so they run first."""
        jobs = self.data["jobs"]
        for job in (
            "fast_clang_format",
            "fast_style",
            "fast_typing",
            "fast_spelling",
            "fast_codeql",
        ):
            self.assertNotIn(
                "needs",
                jobs[job],
                f"Fast-phase job '{job}' must not have 'needs'",
            )

    def test_codeql_grants_security_events_write(self):
        """Verifies that the fast_codeql job grants security-events: write permission."""
        perms = self.data["jobs"]["fast_codeql"].get("permissions", {})
        self.assertEqual(
            perms.get("security-events"),
            "write",
            "fast_codeql must grant 'security-events: write'",
        )

    # ── Gate 1 ───────────────────────────────────────────────────────────────

    def test_fast_gate_needs_all_fast_jobs(self):
        """Verifies that fast_gate depends on all five fast-phase jobs."""
        gate = self.data["jobs"]["fast_gate"]
        needs = gate["needs"]
        for job in (
            "fast_clang_format",
            "fast_style",
            "fast_typing",
            "fast_spelling",
            "fast_codeql",
        ):
            self.assertIn(job, needs, f"fast_gate must need '{job}'")

    # ── Phase 2+3: ci-1-core called as reusable workflow ─────────────────────

    def test_call_ci_core_job_exists(self):
        """Verifies that the orchestrator has a job that calls ci-1-core."""
        self.assertIn("call_ci_core", self.data["jobs"])

    def test_call_ci_core_uses_ci_core_workflow(self):
        """Verifies that call_ci_core calls ci_core.yml via uses:."""
        job = self.data["jobs"]["call_ci_core"]
        self.assertEqual(job["uses"], "./.github/workflows/ci_core.yml")

    def test_call_ci_core_needs_fast_gate(self):
        """Verifies that call_ci_core is blocked by the fast gate."""
        needs = self.data["jobs"]["call_ci_core"]["needs"]
        self.assertIn("fast_gate", needs)

    # ── Final job ─────────────────────────────────────────────────────────────

    def test_ci_pass_job_exists(self):
        """Verifies that the ci_pass final gate job exists."""
        self.assertIn("ci_pass", self.data["jobs"])

    def test_ci_pass_needs_call_ci_core(self):
        """Verifies that ci_pass depends on the ci-1-core call."""
        needs = self.data["jobs"]["ci_pass"]["needs"]
        self.assertIn("call_ci_core", needs)

    def test_ci_pass_runs_if_always(self):
        """Verifies that ci_pass uses if: always() so it always reports a result."""
        self.assertEqual(self.data["jobs"]["ci_pass"].get("if"), "always()")

    # ── Pipeline order ────────────────────────────────────────────────────────

    def test_pipeline_order_is_enforced(self):
        """Verifies the full dependency chain: fast → gate → core → ci_pass."""
        jobs = self.data["jobs"]

        def needs(job):
            return jobs[job].get("needs", [])

        # Gate 1 needs all fast jobs
        self.assertIn("fast_clang_format", needs("fast_gate"))
        self.assertIn("fast_style", needs("fast_gate"))
        self.assertIn("fast_typing", needs("fast_gate"))
        self.assertIn("fast_spelling", needs("fast_gate"))
        self.assertIn("fast_codeql", needs("fast_gate"))
        # call_ci_core needs gate 1
        self.assertIn("fast_gate", needs("call_ci_core"))
        # ci_pass needs call_ci_core
        self.assertIn("call_ci_core", needs("ci_pass"))

    # ── Reusable workflows exist ──────────────────────────────────────────────

    def test_reusable_workflows_have_workflow_call_trigger(self):
        """Verifies that every called workflow file supports workflow_call:."""
        wf_dir = self.root / ".github" / "workflows"
        called = [
            "clang_format.yml",
            "style.yml",
            "typing.yml",
            "spelling.yml",
            "codeql.yml",
            "ci_core.yml",
        ]
        for fname in called:
            data = yaml.safe_load((wf_dir / fname).read_text(encoding="utf-8"))
            # PyYAML parses bare 'on' as True
            on = data.get(True, data.get("on", {}))
            self.assertIn(
                "workflow_call",
                on,
                f"{fname} must declare 'workflow_call:' trigger",
            )

    # ── Downstream workflows updated ──────────────────────────────────────────

    def test_downstream_workflows_listen_to_ci_orchestrator(self):
        """Verifies that workflows downstream of ci-1-core now listen to ci-orchestrator."""
        wf_dir = self.root / ".github" / "workflows"
        downstream = ["docs.yml", "ci_debug.yml", "build_standalone_example.yml"]
        for fname in downstream:
            data = yaml.safe_load((wf_dir / fname).read_text(encoding="utf-8"))
            on = data.get(True, data.get("on", {}))
            wr = on.get("workflow_run", {})
            workflows = wr.get("workflows", [])
            self.assertIn(
                "ci-orchestrator",
                workflows,
                f"{fname} must listen to 'ci-orchestrator' instead of 'ci-1-core'",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
