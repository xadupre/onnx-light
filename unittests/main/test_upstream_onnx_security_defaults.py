import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


class TestUpstreamOnnxSecurityDefaults(unittest.TestCase):
    """Checks that upstream ONNX defaults stay on a patched release."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]

    def test_python_dependency_minimums_use_patched_onnx(self):
        pyproject = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        pixi = (self.root / "pixi.toml").read_text(encoding="utf-8")

        self.assertEqual(pyproject.count('"onnx>=1.21.0"'), 2)
        self.assertEqual(pixi.count('onnx = ">=1.21.0"'), 2)

    def test_load_onnx_time_example_uses_patched_default_tag(self):
        cmake = (self.root / "examples" / "load_onnx_time" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        build_sh = (self.root / "examples" / "load_onnx_time" / "build.sh").read_text(
            encoding="utf-8"
        )
        build_bat = (self.root / "examples" / "load_onnx_time" / "build.bat").read_text(
            encoding="utf-8"
        )
        docs = (self.root / "docs" / "examples_cc" / "load_onnx_time_example.rst").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("v1.17.0", cmake)
        self.assertNotIn("v1.17.0", build_sh)
        self.assertNotIn("v1.17.0", build_bat)
        self.assertNotIn("v1.17.0", docs)
        self.assertIn('set(ONNX_GIT_TAG "v1.21.0")', cmake)
        self.assertIn('ONNX_DEFAULT_GIT_TAG="${ONNX_DEFAULT_GIT_TAG:-v1.21.0}"', build_sh)
        self.assertIn('set "ONNX_DEFAULT_GIT_TAG=v1.21.0"', build_bat)
        self.assertIn("ONNX_GIT_TAG=v1.21.0 bash examples/load_onnx_time/build.sh", docs)
        self.assertIn("set ONNX_GIT_TAG=v1.21.0", docs)

    def test_build_sh_clamps_installed_onnx_to_patched_release_floor(self):
        build_sh = (self.root / "examples" / "load_onnx_time" / "build.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('re.match(r"(\\d+)\\.(\\d+)\\.(\\d+)", onnx.__version__)', build_sh)
        self.assertIn(
            'print("v" + ".".join(groups) if parts >= (1, 21, 0) else "v1.21.0")', build_sh
        )
        self.assertIn("falling back to v1.21.0", build_sh)

    def _get_build_sh_python_snippet(self):
        build_sh = (self.root / "examples" / "load_onnx_time" / "build.sh").read_text(
            encoding="utf-8"
        )
        match = re.search(r"<<'PY'\n(.*?)\nPY", build_sh, re.DOTALL)
        self.assertIsNotNone(match, "Unable to find embedded Python in build.sh")
        return match.group(1)

    def _run_build_sh_python_snippet(self, version: str):
        snippet = self._get_build_sh_python_snippet()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "onnx.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = (
                tmp if not env.get("PYTHONPATH") else tmp + os.pathsep + env["PYTHONPATH"]
            )
            proc = subprocess.run(
                [sys.executable, "-c", snippet],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}")
        return proc.stdout.strip(), proc.stderr.strip()

    def test_build_sh_version_clamp_runtime(self):
        stdout, stderr = self._run_build_sh_python_snippet("1.17.0")
        self.assertEqual(stdout, "v1.21.0")
        self.assertEqual(stderr, "")

        stdout, stderr = self._run_build_sh_python_snippet("1.21.0")
        self.assertEqual(stdout, "v1.21.0")
        self.assertEqual(stderr, "")

        stdout, stderr = self._run_build_sh_python_snippet("1.22.0.dev20260615")
        self.assertEqual(stdout, "v1.22.0")
        self.assertEqual(stderr, "")

        stdout, stderr = self._run_build_sh_python_snippet("invalid")
        self.assertEqual(stdout, "v1.21.0")
        self.assertIn("falling back to v1.21.0", stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
