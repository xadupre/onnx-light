import ast
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


def _load_find_standalone_executable():
    root = pathlib.Path(__file__).resolve().parents[1]
    source_path = root / "onnx_light" / "onnx" / "doc.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "find_standalone_executable"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {"os": os, "pathlib": pathlib, "shutil": shutil}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["find_standalone_executable"]


def _load_find_load_onnx_time_executable():
    root = pathlib.Path(__file__).resolve().parents[1]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    windows_build_configs_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "WINDOWS_BUILD_CONFIGS"
            for target in node.targets
        )
    )
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_find_load_onnx_time_executable"
    )
    module = ast.Module(body=[windows_build_configs_node, function_node], type_ignores=[])
    namespace = {
        "os": os,
        "pathlib": pathlib,
        "shutil": shutil,
        "find_standalone_executable": _load_find_standalone_executable(),
    }
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_find_load_onnx_time_executable"], namespace


def _load_find_save_onnx_light_time_executable():
    root = pathlib.Path(__file__).resolve().parents[1]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    windows_build_configs_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "WINDOWS_BUILD_CONFIGS"
            for target in node.targets
        )
    )
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_find_save_onnx_light_time_executable"
    )
    module = ast.Module(body=[windows_build_configs_node, function_node], type_ignores=[])
    namespace = {
        "os": os,
        "pathlib": pathlib,
        "shutil": shutil,
        "find_standalone_executable": _load_find_standalone_executable(),
    }
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_find_save_onnx_light_time_executable"], namespace


def _load_measure_cpp_load_with_example():
    root = pathlib.Path(__file__).resolve().parents[1]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    cpp_pattern_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CPP_LOAD_METRIC_PATTERN"
            for target in node.targets
        )
    )
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_measure_cpp_load_with_example"
    )
    module = ast.Module(body=[cpp_pattern_node, function_node], type_ignores=[])
    namespace = {"subprocess": subprocess, "re": re}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_measure_cpp_load_with_example"], namespace


def _load_measure_cpp_save_with_example():
    root = pathlib.Path(__file__).resolve().parents[1]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    cpp_pattern_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CPP_SAVE_METRIC_PATTERN"
            for target in node.targets
        )
    )
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_measure_cpp_save_with_example"
    )
    module = ast.Module(body=[cpp_pattern_node, function_node], type_ignores=[])
    namespace = {"subprocess": subprocess, "re": re, "shutil": shutil, "tempfile": tempfile}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_measure_cpp_save_with_example"], namespace


class TestPlotOnnxTime(unittest.TestCase):
    def test_find_standalone_executable_returns_none_in_ci_or_without_script_file(self):
        find_executable = _load_find_standalone_executable()
        with patch.dict(os.environ, {"CI": "yes"}, clear=False):
            found = find_executable(
                "load_onnx_time",
                [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                "some_script.py",
            )
        self.assertIsNone(found)

        with (
            patch.dict(os.environ, {"CI": "0"}, clear=False),
            patch.object(shutil, "which") as mocked_which,
        ):
            found = find_executable(
                "load_onnx_time",
                [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                None,
            )
        self.assertIsNone(found)
        mocked_which.assert_not_called()

    def test_find_standalone_executable_falls_back_to_path_lookup(self):
        find_executable = _load_find_standalone_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            script_path.parent.mkdir(parents=True)
            script_path.write_text("", encoding="utf-8")
            with (
                patch.dict(os.environ, {"CI": "0"}, clear=False),
                patch.object(shutil, "which", return_value="/usr/bin/load_onnx_time"),
            ):
                found = find_executable(
                    "load_onnx_time",
                    [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                    str(script_path),
                    windows_build_configs=("Release", "RelWithDebInfo", "Debug", "MinSizeRel"),
                )
            self.assertEqual("/usr/bin/load_onnx_time", found)

    def test_find_standalone_executable_prefers_local_candidate(self):
        find_executable = _load_find_standalone_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            script_path.parent.mkdir(parents=True)
            script_path.write_text("", encoding="utf-8")
            executable = (
                pathlib.Path(tmp) / "build" / "examples" / "load_onnx_time" / "load_onnx_time"
            )
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            with (
                patch.dict(os.environ, {"CI": "0"}, clear=False),
                patch.object(shutil, "which") as mocked_which,
            ):
                found = find_executable(
                    "load_onnx_time",
                    [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                    str(script_path),
                )
            self.assertEqual(executable.resolve(), pathlib.Path(found).resolve())
            mocked_which.assert_not_called()

    def test_find_executable_in_examples_build_location(self):
        find_executable, namespace = _load_find_load_onnx_time_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            executable = (
                pathlib.Path(tmp) / "build" / "examples" / "load_onnx_time" / "load_onnx_time"
            )
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            namespace["__file__"] = str(script_path)

            with patch.dict(namespace["os"].environ, {"CI": "0"}, clear=False):
                found = find_executable()

            self.assertIsNotNone(found)
            self.assertEqual(executable.resolve(), pathlib.Path(found).resolve())

    def test_returns_none_in_ci(self):
        find_executable, namespace = _load_find_load_onnx_time_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            namespace["__file__"] = str(script_path)
            with patch.dict(namespace["os"].environ, {"CI": "true"}, clear=False):
                found = find_executable()
            self.assertIsNone(found)

    def test_falls_back_to_path_lookup(self):
        find_executable, namespace = _load_find_load_onnx_time_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            namespace["__file__"] = str(script_path)
            with (
                patch.dict(namespace["os"].environ, {"CI": "0"}, clear=False),
                patch.object(
                    namespace["shutil"], "which", return_value="/usr/bin/load_onnx_time"
                ),
            ):
                found = find_executable()
            self.assertEqual("/usr/bin/load_onnx_time", found)

    def test_measure_cpp_load_with_example_x4(self):
        measure_cpp, namespace = _load_measure_cpp_load_with_example()
        namespace["_find_load_onnx_time_executable"] = lambda: "/tmp/load_onnx_time"
        stdout = "\n".join(
            ["Average load (ms): 10.0", "Min load (ms): 8.0", "Max load (ms): 12.0"]
        )
        completed = subprocess.CompletedProcess(
            args=["/tmp/load_onnx_time", "model.onnx", "5", "4"], returncode=0, stdout=stdout
        )
        with patch.object(namespace["subprocess"], "run", return_value=completed) as mocked_run:
            got = measure_cpp("model.onnx", n=5, num_threads=4)

        self.assertIsNotNone(got)
        self.assertEqual("load/1filex4/onnxlight-cpp", got["name"])
        self.assertEqual(0.010, got["avg"])
        self.assertEqual(0.008, got["min"])
        self.assertEqual(0.012, got["max"])
        mocked_run.assert_called_once_with(
            ["/tmp/load_onnx_time", "model.onnx", "5", "4"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

    def test_find_save_executable_in_examples_build_location(self):
        find_executable, namespace = _load_find_save_onnx_light_time_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            executable = (
                pathlib.Path(tmp)
                / "build"
                / "examples"
                / "save_onnx_light_time"
                / "save_onnx_light_time"
            )
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            namespace["__file__"] = str(script_path)

            with patch.dict(namespace["os"].environ, {"CI": "0"}, clear=False):
                found = find_executable()

            self.assertIsNotNone(found)
            self.assertEqual(executable.resolve(), pathlib.Path(found).resolve())

    def test_find_save_executable_returns_none_in_ci(self):
        find_executable, namespace = _load_find_save_onnx_light_time_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            namespace["__file__"] = str(script_path)
            with patch.dict(namespace["os"].environ, {"CI": "true"}, clear=False):
                found = find_executable()
            self.assertIsNone(found)

    def test_find_save_executable_falls_back_to_path_lookup(self):
        find_executable, namespace = _load_find_save_onnx_light_time_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            namespace["__file__"] = str(script_path)
            with (
                patch.dict(namespace["os"].environ, {"CI": "0"}, clear=False),
                patch.object(
                    namespace["shutil"], "which", return_value="/usr/bin/save_onnx_light_time"
                ),
            ):
                found = find_executable()
            self.assertEqual("/usr/bin/save_onnx_light_time", found)

    def test_measure_cpp_save_with_example_x1(self):
        measure_cpp, namespace = _load_measure_cpp_save_with_example()
        namespace["_find_save_onnx_light_time_executable"] = lambda: "/tmp/save_onnx_light_time"
        stdout = "\n".join(
            ["Average save (ms): 20.0", "Min save (ms): 18.0", "Max save (ms): 25.0"]
        )
        completed = subprocess.CompletedProcess(
            args=["/tmp/save_onnx_light_time", "model.onnx", "/tmp/out", "5", "1"],
            returncode=0,
            stdout=stdout,
        )
        with (
            patch.object(namespace["subprocess"], "run", return_value=completed),
            patch.object(namespace["tempfile"], "TemporaryDirectory") as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__ = lambda s: "/tmp/out"
            mock_tmpdir.return_value.__exit__ = lambda s, *a: False
            got = measure_cpp("model.onnx", n=5, num_threads=1)

        self.assertIsNotNone(got)
        self.assertEqual("save/2filex1/onnxlight-cpp", got["name"])
        self.assertEqual(0.020, got["avg"])
        self.assertEqual(0.018, got["min"])
        self.assertEqual(0.025, got["max"])

    def test_measure_cpp_save_with_example_x4(self):
        measure_cpp, namespace = _load_measure_cpp_save_with_example()
        namespace["_find_save_onnx_light_time_executable"] = lambda: "/tmp/save_onnx_light_time"
        stdout = "\n".join(
            ["Average save (ms): 10.0", "Min save (ms): 8.0", "Max save (ms): 12.0"]
        )
        completed = subprocess.CompletedProcess(
            args=["/tmp/save_onnx_light_time", "model.onnx", "/tmp/out", "5", "4"],
            returncode=0,
            stdout=stdout,
        )
        with (
            patch.object(namespace["subprocess"], "run", return_value=completed),
            patch.object(namespace["tempfile"], "TemporaryDirectory") as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__ = lambda s: "/tmp/out"
            mock_tmpdir.return_value.__exit__ = lambda s, *a: False
            got = measure_cpp("model.onnx", n=5, num_threads=4)

        self.assertIsNotNone(got)
        self.assertEqual("save/2filex4/onnxlight-cpp", got["name"])
        self.assertEqual(0.010, got["avg"])
        self.assertEqual(0.008, got["min"])
        self.assertEqual(0.012, got["max"])

    def test_measure_cpp_save_returns_none_when_executable_missing(self):
        measure_cpp, namespace = _load_measure_cpp_save_with_example()
        namespace["_find_save_onnx_light_time_executable"] = lambda: None
        got = measure_cpp("model.onnx", n=5, num_threads=1)
        self.assertIsNone(got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
