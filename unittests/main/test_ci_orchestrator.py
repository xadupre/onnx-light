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
        self.assertEqual(self.data["name"], "ci-orchestrator")

    def test_triggers(self):
        """Verifies push/pull_request triggers on main."""
        # PyYAML parses the bare 'on' key as the boolean True.
        on = self.data[True]
        self.assertIn("push", on)
        self.assertIn("pull_request", on)
        self.assertIn("main", on["push"]["branches"])
        self.assertIn("main", on["pull_request"]["branches"])

    # ── Phase 1: fast checks ─────────────────────────────────────────────────

    def test_fast_phase_jobs_exist(self):
        jobs = self.data["jobs"]
        for job in (
            "fast_clang_format",
            "fast_style",
            "fast_typing",
            "fast_spelling",
            "fast_codeql",
        ):
            self.assertIn(job, jobs, f"Fast-phase job '{job}' is missing")

    def test_fast_phase_jobs_have_no_needs(self):
        """Fast phase jobs must not depend on any other job (run first)."""
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

    def test_fast_codeql_uses_codeql_action(self):
        steps = self.data["jobs"]["fast_codeql"]["steps"]
        action_names = [
            s.get("uses", "")
            for s in steps
            if s.get("uses", "").startswith("github/codeql-action/")
        ]
        self.assertTrue(
            len(action_names) >= 2,
            "fast_codeql must use at least github/codeql-action/init and "
            "github/codeql-action/analyze",
        )

    # ── Gate 1 ───────────────────────────────────────────────────────────────

    def test_fast_gate_needs_all_fast_jobs(self):
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

    # ── Phase 2: preflight ────────────────────────────────────────────────────

    def test_reduced_tests_needs_fast_gate(self):
        needs = self.data["jobs"]["reduced_tests_ubuntu"]["needs"]
        self.assertIn("fast_gate", needs)

    def test_no_onnx_needs_reduced(self):
        needs = self.data["jobs"]["no_onnx_tests_ubuntu"]["needs"]
        self.assertIn("reduced_tests_ubuntu", needs)

    # ── Gate 2 ────────────────────────────────────────────────────────────────

    def test_preflight_gate_needs_no_onnx(self):
        needs = self.data["jobs"]["preflight_gate"]["needs"]
        self.assertIn("no_onnx_tests_ubuntu", needs)

    # ── Phase 3: heavy builds ─────────────────────────────────────────────────

    def test_heavy_builds_need_preflight_gate(self):
        jobs = self.data["jobs"]
        for job in ("core_tests_ubuntu", "core_tests", "windows_x86_build"):
            needs = jobs[job]["needs"]
            self.assertIn(
                "preflight_gate",
                needs,
                f"Phase-3 job '{job}' must need 'preflight_gate'",
            )

    def test_core_tests_matrix_includes_windows_and_macos(self):
        matrix = self.data["jobs"]["core_tests"]["strategy"]["matrix"]
        self.assertIn("windows-latest", matrix["os"])
        self.assertIn("macos-latest", matrix["os"])

    def test_windows_build_uses_msvc_ninja_with_sccache(self):
        """Verifies 64-bit Windows build in Phase 3 uses MSVC-backed Ninja with sccache."""
        self.assertIn("- name: Set up MSVC for Ninja (Windows)", self.content)
        self.assertIn("uses: ilammy/msvc-dev-cmd@v1.13.0", self.content)
        self.assertIn("arch: x64", self.content)
        self.assertIn("- name: Build and Install package (Windows)", self.content)
        self.assertIn("CMAKE_GENERATOR: Ninja", self.content)
        self.assertIn('SCCACHE_GHA_ENABLED: "true"', self.content)
        self.assertIn("cmake.define.CMAKE_C_COMPILER_LAUNCHER=sccache", self.content)
        self.assertIn("cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=sccache", self.content)

    def test_windows_x86_build_job_exists(self):
        """Verifies that the dedicated Windows x86 build job is present."""
        self.assertIn("windows_x86_build:", self.content)
        self.assertIn("name: core (windows-latest, x86 build)", self.content)

    # ── Final job ─────────────────────────────────────────────────────────────

    def test_ci_pass_job_exists(self):
        self.assertIn("ci_pass", self.data["jobs"])

    def test_ci_pass_needs_all_phase3_jobs(self):
        needs = self.data["jobs"]["ci_pass"]["needs"]
        for job in ("core_tests_ubuntu", "core_tests", "windows_x86_build"):
            self.assertIn(job, needs, f"ci_pass must need '{job}'")

    def test_ci_pass_runs_if_always(self):
        """ci_pass must use if: always() so it reports failure even when others fail."""
        self.assertEqual(self.data["jobs"]["ci_pass"].get("if"), "always()")

    def test_pipeline_order_is_enforced(self):
        """Smoke-check that the dependency chain fast→preflight→heavy→pass is intact."""
        jobs = self.data["jobs"]

        def needs(job):
            return jobs[job].get("needs", [])

        # Gate 1 needs all fast jobs
        self.assertIn("fast_clang_format", needs("fast_gate"))
        # Reduced needs gate 1
        self.assertIn("fast_gate", needs("reduced_tests_ubuntu"))
        # no-onnx needs reduced
        self.assertIn("reduced_tests_ubuntu", needs("no_onnx_tests_ubuntu"))
        # Gate 2 needs no-onnx
        self.assertIn("no_onnx_tests_ubuntu", needs("preflight_gate"))
        # Phase 3 needs gate 2
        self.assertIn("preflight_gate", needs("core_tests_ubuntu"))
        # ci_pass needs phase 3
        self.assertIn("core_tests_ubuntu", needs("ci_pass"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
