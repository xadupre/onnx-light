import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestSetupBuildExt(unittest.TestCase):
    def test_setup_build_ext_inplace_dry_run(self):
        """Verifies setup.py build_ext --inplace --dry-run execution."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "setup.py", "build_ext", "--inplace", "--dry-run"]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("running build_ext", f"{proc.stdout}\n{proc.stderr}")

    def test_setup_build_ext_inplace_dry_run_honors_cmake_args(self):
        """Verifies setup.py build_ext forwards CMAKE_ARGS to CMake."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "setup.py", "build_ext", "--inplace", "--dry-run"]
        env = dict(os.environ)
        env["CMAKE_ARGS"] = "-DONNX_LIGHT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug"
        proc = subprocess.run(
            command, cwd=root, env=env, check=False, capture_output=True, text=True
        )

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("-DONNX_LIGHT_BUILD_TESTS=ON", f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("-DCMAKE_BUILD_TYPE=Debug", f"{proc.stdout}\n{proc.stderr}")

    def test_setup_build_ext_inplace_dry_run_without_setuptools(self):
        """Verifies setup.py build_ext --inplace without setuptools."""
        root = Path(__file__).resolve().parents[2]
        # -S avoids importing site, which excludes setuptools from module lookup.
        command = [sys.executable, "-S", "setup.py", "build_ext", "--inplace", "--dry-run"]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("running build_ext", f"{proc.stdout}\n{proc.stderr}")

    def test_setup_build_ext_inplace_dry_run_without_setuptools_honors_cmake_args(self):
        """Verifies setup.py build_ext forwards CMAKE_ARGS without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-S", "setup.py", "build_ext", "--inplace", "--dry-run"]
        env = dict(os.environ)
        env["CMAKE_ARGS"] = "-DONNX_LIGHT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug"
        proc = subprocess.run(
            command, cwd=root, env=env, check=False, capture_output=True, text=True
        )

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("-DONNX_LIGHT_BUILD_TESTS=ON", f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("-DCMAKE_BUILD_TYPE=Debug", f"{proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
