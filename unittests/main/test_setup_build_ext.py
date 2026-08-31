import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import onnx_light
from onnx_light import get_cpp_build_info
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx_py import _onnxpyprotoop


def has_setuptool():
    try:
        import setuptools  # noqa: F401

        return True
    except ImportError:
        return False


skip_test = has_setuptool()


class TestSetupBuildExt(ExtTestCase):
    def test_setup_build_ext_inplace_rejects_editable_install(self):
        """Verifies that an existing editable hook blocks an inplace build."""
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            hook = Path(temporary_directory) / "_editable_skbc_onnx_light.pth"
            hook.touch()
            env = dict(os.environ)
            python_path = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                f"{temporary_directory}{os.pathsep}{python_path}"
                if python_path
                else temporary_directory
            )
            proc = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(proc.returncode, 0)
        output = f"{proc.stdout}\n{proc.stderr}"
        self.assertIn("editable onnx-light installation is active", output)
        self.assertIn(str(hook), output)
        self.assertNotIn("cmake -S", output)

    def test_cpp_build_info_uses_python_runtime_library(self):
        info = get_cpp_build_info()
        library_dir = Path(_onnxpyprotoop.__file__).resolve().parent
        self.assertEqual(Path(info["include_dir"]), Path(onnx_light.__file__).resolve().parent)
        self.assertEqual(Path(info["library_dir"]), library_dir)
        # lib_onnx_proto is built shared on every platform when the Python
        # extensions are built, so it is always reported here.
        self.assertIn("proto_library", info)
        for key in ("core_library", "proto_library"):
            if key not in info:
                continue
            library = Path(info[key])
            self.assertEqual(library.parent, library_dir)
            self.assertExists(library)

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
    def test_setup_build_ext_cpp_tests_uses_one_build_before_ctest(self):
        """Tests that --cpp-tests uses one global build before install and ctest."""
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
        build_lines = [i for i, line in enumerate(lines) if line.startswith("cmake --build ")]
        self.assertEqual(
            len(build_lines), 1, msg="Expected one global CMake build:\n" + "\n".join(lines)
        )
        build_index = build_lines[0]
        self.assertNotIn("--target", lines[build_index])
        install_index = self._line_index(
            lines, lambda line: "cmake --install" in line, "cmake --install"
        )
        ctest_index = self._line_index(lines, lambda line: line.startswith("ctest "), "ctest")
        self.assertLess(build_index, install_index)
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
    def test_setup_build_ext_without_setuptools_uses_one_build_before_ctest(self):
        """Tests --cpp-tests uses one global build before ctest without setuptools."""
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
        build_lines = [i for i, line in enumerate(lines) if line.startswith("cmake --build ")]
        self.assertEqual(
            len(build_lines), 1, msg="Expected one global CMake build:\n" + "\n".join(lines)
        )
        build_index = build_lines[0]
        self.assertNotIn("--target", lines[build_index])
        install_index = self._line_index(
            lines, lambda line: "cmake --install" in line, "cmake --install"
        )
        ctest_index = self._line_index(lines, lambda line: line.startswith("ctest "), "ctest")
        self.assertLess(build_index, install_index)
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
    def test_setup_build_ext_honors_cmake_build_parallel_level(self):
        """Verifies setup.py build_ext forwards CMAKE_BUILD_PARALLEL_LEVEL to --parallel."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "setup.py", "build_ext", "--inplace", "--dry-run"]
        env = dict(os.environ)
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = "3"
        proc = subprocess.run(
            command, cwd=root, env=env, check=False, capture_output=True, text=True
        )

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("--parallel 3", f"{proc.stdout}\n{proc.stderr}")

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
    def test_setup_build_ext_without_setuptools_honors_cmake_build_parallel_level(self):
        """Verifies build_ext forwards CMAKE_BUILD_PARALLEL_LEVEL without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-S", "setup.py", "build_ext", "--inplace", "--dry-run"]
        env = dict(os.environ)
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = "3"
        proc = subprocess.run(
            command, cwd=root, env=env, check=False, capture_output=True, text=True
        )

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        self.assertIn("--parallel 3", f"{proc.stdout}\n{proc.stderr}")

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

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_install_uses_release_config(self):
        """Verifies setup.py build_ext installs with --config Release, never Debug."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "setup.py", "build_ext", "--inplace", "--dry-run"]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        output = f"{proc.stdout}\n{proc.stderr}"
        lines = output.splitlines()
        install_index = self._line_index(
            lines, lambda line: "cmake --install" in line, "cmake --install"
        )
        install_line = lines[install_index]
        self.assertIn("--config Release", install_line)
        self.assertNotIn("Debug", install_line)

    @unittest.skipIf(skip_test, "test add by copilot but unused in real life")
    def test_setup_build_ext_without_setuptools_install_uses_release_config(self):
        """Verifies build_ext installs with --config Release without setuptools."""
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-S", "setup.py", "build_ext", "--inplace", "--dry-run"]
        proc = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        output = f"{proc.stdout}\n{proc.stderr}"
        lines = output.splitlines()
        install_index = self._line_index(
            lines, lambda line: "cmake --install" in line, "cmake --install"
        )
        install_line = lines[install_index]
        self.assertIn("--config Release", install_line)
        self.assertNotIn("Debug", install_line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
