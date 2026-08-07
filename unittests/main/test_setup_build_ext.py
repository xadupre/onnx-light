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


skip_test = has_setuptool()


class TestSetupBuildExt(ExtTestCase):
    def _line_index(self, lines, predicate, description):
        """Returns the index of the first line matching predicate, failing clearly otherwise."""
        for i, line in enumerate(lines):
            if predicate(line):
                return i
        self.fail(f"No line matching {description} in output:\n" + "\n".join(lines))
        return -1  # pragma: no cover - self.fail raises

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
        """Tests that setup.py build_ext builds and runs C++ tests with --cpp-tests."""
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
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("-DONNX_LIGHT_BUILD_TESTS=ON", output)
        self.assertIn("ctest", output)
        self.assertIn("--output-on-failure", output)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_cpp_tests_installs_python_before_ctest(self):
        """Tests that --cpp-tests installs the Python package inplace before running ctest."""
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
        output = f"{proc.stdout}\n{proc.stderr}"
        lines = output.splitlines()
        python_build_index = self._line_index(
            lines, lambda line: "--target _onnxpyprotoop" in line, "the Python extension build"
        )
        install_index = self._line_index(
            lines, lambda line: "cmake --install" in line, "cmake --install"
        )
        ctest_index = self._line_index(lines, lambda line: line.startswith("ctest "), "ctest")
        self.assertLess(python_build_index, install_index)
        self.assertLess(install_index, ctest_index)

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
        """Tests that build_ext builds and runs C++ tests with --cpp-tests without setuptools."""
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
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("-DONNX_LIGHT_BUILD_TESTS=ON", output)
        self.assertIn("ctest", output)
        self.assertIn("--output-on-failure", output)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_without_setuptools_installs_python_before_ctest(self):
        """Tests --cpp-tests installs the Python package inplace before ctest (no setuptools)."""
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
        output = f"{proc.stdout}\n{proc.stderr}"
        lines = output.splitlines()
        python_build_index = self._line_index(
            lines, lambda line: "--target _onnxpyprotoop" in line, "the Python extension build"
        )
        install_index = self._line_index(
            lines, lambda line: "cmake --install" in line, "cmake --install"
        )
        ctest_index = self._line_index(lines, lambda line: line.startswith("ctest "), "ctest")
        self.assertLess(python_build_index, install_index)
        self.assertLess(install_index, ctest_index)

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
    def test_setup_build_ext_default_parallel(self):
        """Verifies setup.py build_ext defaults to parallel CMake builds."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "setup.py", "build_ext", "--inplace", "--dry-run"]
        env = dict(os.environ)
        env.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)
        proc = subprocess.run(
            command, cwd=root, env=env, check=False, capture_output=True, text=True
        )

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("--parallel", f"{proc.stdout}\n{proc.stderr}")

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

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_without_setuptools_default_parallel(self):
        """Verifies setup.py build_ext defaults to parallel builds without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-S", "setup.py", "build_ext", "--inplace", "--dry-run"]
        env = dict(os.environ)
        env.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)
        proc = subprocess.run(
            command, cwd=root, env=env, check=False, capture_output=True, text=True
        )

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("--parallel", f"{proc.stdout}\n{proc.stderr}")

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_cpp_tests_disables_upstream_onnx(self):
        """Tests that --cpp-tests forces ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=OFF."""
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
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("-DONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=OFF", output)
        self.assertNotIn("-DONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=ON", output)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_without_setuptools_cpp_tests_disables_upstream_onnx(self):
        """
        Tests that --cpp-tests forces ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=OFF
        without setuptools.
        """
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
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("-DONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=OFF", output)
        self.assertNotIn("-DONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=ON", output)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_no_kernels_flag(self):
        """Tests that --no-kernels sets ONNX_LIGHT_BUILD_KERNELS=OFF."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--no-kernels",
        ]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("-DONNX_LIGHT_BUILD_KERNELS=OFF", f"{proc.stdout}\n{proc.stderr}")

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_without_setuptools_no_kernels_flag(self):
        """Tests that --no-kernels sets ONNX_LIGHT_BUILD_KERNELS=OFF without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "-S",
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--no-kernels",
        ]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("-DONNX_LIGHT_BUILD_KERNELS=OFF", f"{proc.stdout}\n{proc.stderr}")

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_run_cpp_tests_flag_removed(self):
        """Tests that the removed --run-cpp-tests flag is rejected."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--run-cpp-tests",
        ]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertNotEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("--run-cpp-tests", f"{proc.stdout}\n{proc.stderr}")

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_without_setuptools_run_cpp_tests_flag_removed(self):
        """Tests that the removed --run-cpp-tests flag is rejected without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            "-S",
            "setup.py",
            "build_ext",
            "--inplace",
            "--dry-run",
            "--run-cpp-tests",
        ]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertNotEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("--run-cpp-tests", f"{proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
