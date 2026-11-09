import re
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestFuzzWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.content = (cls.root / ".github/workflows/cq_fuzz.yml").read_text(encoding="utf-8")

    def test_sanitizers_cover_libraries_and_harnesses(self):
        """Verifies that sanitizer instrumentation reaches the fuzzed libraries."""
        self.assertIn(
            "CFLAGS: -fno-omit-frame-pointer -fsanitize=address,undefined", self.content
        )
        self.assertIn(
            "CXXFLAGS: -fno-omit-frame-pointer -fsanitize=address,undefined", self.content
        )
        self.assertIn("LDFLAGS: -fsanitize=address,undefined", self.content)
        self.assertIn("-DONNX_LIGHT_FUZZER_SANITIZERS=address,undefined", self.content)
        self.assertIn("-DCMAKE_BUILD_TYPE=RelWithDebInfo", self.content)
        self.assertIn("apt-get install -y clang llvm", self.content)
        self.assertRegex(self.content, r"ASAN_OPTIONS: .*symbolize=1.*abort_on_error=1")
        self.assertIn(
            "UBSAN_OPTIONS: print_stacktrace=1:halt_on_error=1:abort_on_error=1", self.content
        )

    def test_each_harness_has_a_crash_directory(self):
        """Verifies that every smoke harness saves artifacts in its own directory."""
        targets = {path.stem for path in (self.root / "fuzz").glob("fuzz_*.cc")}
        commands = re.findall(r"run: \./build-fuzz/(fuzz_\w+) ([^\n]+)", self.content)
        self.assertEqual({target for target, _ in commands}, targets)
        directories = re.search(r"mkdir -p crashes/\{([^}]+)\}", self.content).group(1).split(",")
        self.assertEqual({f"fuzz_{directory}" for directory in directories}, targets)
        for target, arguments in commands:
            with self.subTest(target=target):
                self.assertIn(f"-artifact_prefix=crashes/{target[5:]}/", arguments)
                self.assertIn("-runs=2000", arguments)
                self.assertIn("-timeout=25", arguments)

    def test_failure_diagnostics_preserve_artifacts(self):
        """Verifies that bounded crash replay and artifact upload run on failure."""
        self.assertRegex(self.content, r"name: Print backtraces for crashes\n\s+if: failure\(\)")
        self.assertRegex(self.content, r"name: Upload crash artifacts\n\s+if: failure\(\)")
        self.assertIn('"$harness" -timeout=25 "$crash" || true', self.content)
        self.assertIn("name: fuzz-crashes\n          path: crashes/", self.content)
        self.assertIn("if-no-files-found: ignore", self.content)

    def test_embedded_shell_syntax(self):
        """Checks the workflow's multiline shell commands with Bash."""
        blocks = re.findall(r"        run: \|\n((?:          .*\n|\n)+)", self.content)
        self.assertTrue(blocks)
        for block in blocks:
            subprocess.run(
                ["bash", "-n"], input=block, text=True, check=True, capture_output=True
            )

    def test_replay_without_crashes(self):
        """Verifies that failure diagnostics also handle builds without crash inputs."""
        replay = self.content.split("      - name: Print backtraces for crashes\n", 1)[1]
        block = re.search(r"        run: \|\n((?:          .*\n|\n)+)", replay).group(1)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", "-e", "-c", block],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
