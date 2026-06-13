import subprocess
import unittest
from pathlib import Path


class TestNoSymlinks(unittest.TestCase):
    """Ensures no symlink is allowed anywhere in the repository."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]

    def test_no_tracked_symlinks(self):
        """Checks that git does not track any symlink (mode 120000)."""
        result = subprocess.run(
            ["git", "ls-files", "-s"], cwd=self.root, capture_output=True, text=True, check=True
        )
        symlinks = [
            line.split("\t", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("120000 ")
        ]
        self.assertEqual(
            symlinks, [], f"The repository must not contain any symlink, found: {symlinks}"
        )

    def test_no_symlinks_on_disk(self):
        """Checks that no symlink exists on disk outside the .git directory."""
        symlinks = [
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if ".git" not in path.parts and path.is_symlink()
        ]
        self.assertEqual(
            symlinks, [], f"The repository must not contain any symlink, found: {symlinks}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
