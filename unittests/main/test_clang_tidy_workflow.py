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

    def test_uses_sccache(self):
        """Verifies that the clang-tidy build is cached with sccache."""
        self.assertIn("mozilla-actions/sccache-action", self.content)
        self.assertIn('SCCACHE_GHA_ENABLED: "true"', self.content)
        self.assertIn("-DCMAKE_C_COMPILER_LAUNCHER=sccache", self.content)
        self.assertIn("-DCMAKE_CXX_COMPILER_LAUNCHER=sccache", self.content)

    def test_pull_request_scans_only_changed_files(self):
        """Verifies that pull requests only run clang-tidy on changed sources."""
        self.assertIn("git diff --name-only", self.content)
        self.assertIn("run-clang-tidy", self.content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
