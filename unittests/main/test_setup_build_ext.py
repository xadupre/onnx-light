import os
import subprocess
import sys
import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


def has_setuptool():
    try:
        import setuptools  # noqa: F401

        return True
    except ImportError:
        return False


skip_test = not has_setuptool()


class TestSetupBuildExt(ExtTestCase):
    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_inplace_dry_run(self):
        """Verifies setup.py build_ext --inplace --dry-run execution."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "setup.py", "build_ext", "--inplace", "--dry-run"]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("running build_ext", output)
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", output)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
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

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_inplace_dry_run_cpp_tests_flag(self):
        """Tests that setup.py build_ext enables C++ tests with --cpp-tests."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--cpp-tests",
        ]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("-DONNX_LIGHT_BUILD_TESTS=ON", f"{proc.stdout}\n{proc.stderr}")

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_cpp_tests_flag_overrides_cmake_args(self):
        """Tests that --cpp-tests overrides ONNX_LIGHT_BUILD_TESTS from CMAKE_ARGS."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--cpp-tests",
        ]
        env = dict(os.environ)
        env["CMAKE_ARGS"] = "-DONNX_LIGHT_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Debug"
        proc = subprocess.run(
            command, cwd=root, env=env, check=False, capture_output=True, text=True
        )

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("-DONNX_LIGHT_BUILD_TESTS=ON", output)
        self.assertNotIn("-DONNX_LIGHT_BUILD_TESTS=OFF", output)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_inplace_dry_run_without_setuptools(self):
        """Verifies setup.py build_ext --inplace without setuptools."""
        root = Path(__file__).resolve().parents[2]
        # -S avoids importing site, which excludes setuptools from module lookup.
        command = [sys.executable, "-S", "setup.py", "build_ext", "--inplace", "--dry-run"]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("running build_ext", output)
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", output)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
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

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_inplace_dry_run_without_setuptools_cpp_tests_flag(self):
        """Tests that setup.py build_ext enables C++ tests with --cpp-tests without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "-S",
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--cpp-tests",
        ]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("-DONNX_LIGHT_BUILD_TESTS=ON", f"{proc.stdout}\n{proc.stderr}")

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_without_setuptools_cpp_tests_flag_overrides_cmake_args(self):
        """Tests that --cpp-tests overrides ONNX_LIGHT_BUILD_TESTS without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "-S",
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--cpp-tests",
        ]
        env = dict(os.environ)
        env["CMAKE_ARGS"] = "-DONNX_LIGHT_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Debug"
        proc = subprocess.run(
            command, cwd=root, env=env, check=False, capture_output=True, text=True
        )

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("-DONNX_LIGHT_BUILD_TESTS=ON", output)
        self.assertNotIn("-DONNX_LIGHT_BUILD_TESTS=OFF", output)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_parallel_flag(self):
        """Tests that setup.py build_ext passes --parallel N to cmake --build."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--parallel=4",
        ]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("--parallel 4", f"{proc.stdout}\n{proc.stderr}")

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_without_setuptools_parallel_flag(self):
        """Tests that build_ext passes --parallel N to cmake --build without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "-S",
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--parallel",
            "4",
        ]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("--parallel 4", f"{proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
