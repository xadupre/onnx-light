import subprocess
import tempfile
import unittest
from pathlib import Path


class TestClean(unittest.TestCase):
    def test_build_artifacts_only_removed_with_all(self):
        script = Path(__file__).resolve().parents[2] / "clean.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = (
                "build-copilot",
                "build_copilot",
                "build2",
                "build_tests",
                "_skbuild",
                "htmlcov",
                "cover",
            )
            for name in artifacts:
                (root / name).mkdir()
                (root / name / "generated.txt").write_text("generated", encoding="utf-8")
            for name in (".coverage", ".coverage.copilot"):
                (root / name).write_text("generated", encoding="utf-8")
            (root / "source.txt").write_text("source", encoding="utf-8")

            subprocess.run(["bash", str(script)], cwd=root, check=True, capture_output=True)
            for name in (*artifacts, ".coverage", ".coverage.copilot"):
                self.assertTrue((root / name).exists(), name)

            result = subprocess.run(
                ["bash", str(script), "--all"], cwd=root, check=True, capture_output=True
            )
            self.assertEqual(result.stderr, b"")
            for name in (*artifacts, ".coverage", ".coverage.copilot"):
                self.assertFalse((root / name).exists(), name)
            self.assertEqual((root / "source.txt").read_text(encoding="utf-8"), "source")


if __name__ == "__main__":
    unittest.main()
