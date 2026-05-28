import ast
import argparse
import io
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
import onnx_light.onnx_lib.helper as onnxlh

from onnx_light.ext_test_case import ExtTestCase
from unittest.mock import patch


def _load_find_standalone_executable(custom_file: str | None = None):
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "onnx_light" / "doc.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "find_standalone_executable"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {"os": os, "pathlib": pathlib, "shutil": shutil}
    if custom_file is not None:
        namespace["__file__"] = custom_file
    else:
        # Default: use the actual doc.py path so __file__ resolves correctly.
        namespace["__file__"] = str(source_path)
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["find_standalone_executable"]


def _get_measure_cpp_with_example_node():
    """Returns the AST node for ``measure_cpp_with_example`` from doc.py."""
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "onnx_light" / "doc.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "measure_cpp_with_example"
    )


def _load_find_load_onnx_time_executable():
    root = pathlib.Path(__file__).resolve().parents[2]
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


def _load_find_load_onnx_light_time_executable():
    root = pathlib.Path(__file__).resolve().parents[2]
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
        and node.name == "_find_load_onnx_light_time_executable"
    )
    module = ast.Module(body=[windows_build_configs_node, function_node], type_ignores=[])
    namespace = {
        "os": os,
        "pathlib": pathlib,
        "shutil": shutil,
        "find_standalone_executable": _load_find_standalone_executable(),
    }
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_find_load_onnx_light_time_executable"], namespace


def _load_find_save_onnx_light_time_executable():
    root = pathlib.Path(__file__).resolve().parents[2]
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
    root = pathlib.Path(__file__).resolve().parents[2]
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
    measure_cpp_node = _get_measure_cpp_with_example_node()
    module = ast.Module(body=[cpp_pattern_node, measure_cpp_node, function_node], type_ignores=[])
    namespace = {"subprocess": subprocess, "re": re}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_measure_cpp_load_with_example"], namespace


def _load_measure_cpp_save_with_example():
    root = pathlib.Path(__file__).resolve().parents[2]
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
    measure_cpp_node = _get_measure_cpp_with_example_node()
    module = ast.Module(body=[cpp_pattern_node, measure_cpp_node, function_node], type_ignores=[])
    namespace = {"subprocess": subprocess, "re": re, "shutil": shutil, "tempfile": tempfile}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_measure_cpp_save_with_example"], namespace


def _load_parse_benchmark_scenarios():
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    scenarios_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BENCHMARK_SCENARIOS"
            for target in node.targets
        )
    )
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_benchmark_scenarios"
    )
    module = ast.Module(body=[scenarios_node, function_node], type_ignores=[])
    namespace = {"argparse": argparse}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_parse_benchmark_scenarios"]


def _get_measure_call_keywords(result_name: str) -> dict[str, ast.AST]:
    """Returns keyword AST nodes for the ``measure`` call identified by *result_name*."""
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "measure":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value == result_name:
            return {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
    raise AssertionError(f"Unable to find measure call for {result_name!r}")


def _get_measure_call_callable(result_name: str) -> ast.AST:
    """Returns the callable AST node for the ``measure`` call identified by *result_name*."""
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "measure":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value == result_name:
            return node.args[1]
    raise AssertionError(f"Unable to find measure call for {result_name!r}")


def _find_call(function_name: str, first_arg_name: str | None = None) -> ast.Call:
    """Returns the call node for ``function_name`` optionally matching a name arg."""
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != function_name:
            continue
        if first_arg_name is None:
            return node
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        if node.args[0].id == first_arg_name:
            return node
    raise AssertionError(
        f"Unable to find call to {function_name!r}"
        + ("" if first_arg_name is None else f" with first arg {first_arg_name!r}")
    )


def _load_parse_model_path():
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_model_path"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {"argparse": argparse}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_parse_model_path"]


def _load_parse_model_id():
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_model_id"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {"argparse": argparse}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_parse_model_id"]


def _load_download_hf_model():
    import urllib.error
    import urllib.request

    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_download_hf_model"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {"os": os, "urllib": urllib}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_download_hf_model"], namespace


def _load_print_model_stats():
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "core" / "plot_onnx_time.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    tensor_bytes_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tensor_data_bytes"
    )
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "print_model_stats"
    )
    module = ast.Module(body=[tensor_bytes_node, function_node], type_ignores=[])
    namespace = {"onnx": onnxl, "os": os, "math": __import__("math"), "onnxlh": onnxlh}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["print_model_stats"]


class TestPlotOnnxTime(ExtTestCase):
    def test_parse_benchmark_scenarios_default(self):
        parse = _load_parse_benchmark_scenarios()
        got = parse([])
        self.assertEqual({"load", "save", "serialize", "parse", "cpp"}, got)

    def test_parse_benchmark_scenarios_multiple(self):
        parse = _load_parse_benchmark_scenarios()
        got = parse(["--scenario", "load", "--scenario", "save"])
        self.assertEqual({"load", "save"}, got)

    def test_external_onnx_save_benchmark_is_single_shot(self):
        keywords = _get_measure_call_keywords("save/2filex1/onnx")
        self.assertEqual(1, keywords["n"].value)
        self.assertEqual(0, keywords["warmup"].value)

    def test_external_save_benchmarks_use_flush_helpers(self):
        for result_name, helper_name in (
            ("save/2filex1/onnx", "_save_onnx_external_with_flush"),
            ("save/2filex1/onnxlight", "_save_onnxlight_external_with_flush"),
        ):
            with self.subTest(result_name=result_name):
                fn = _get_measure_call_callable(result_name)
                self.assertIsInstance(fn, ast.Name)
                self.assertEqual(helper_name, fn.id)

    def test_mmap_vs_ifstream_load_benchmarks_use_file_load_mode(self):
        for result_name, expected_mode in (
            ("load/1filex1/onnxlight-mmap", "MMAP"),
            ("load/1filex1/onnxlight-ifstream", "IFSTREAM"),
        ):
            with self.subTest(result_name=result_name):
                fn = _get_measure_call_callable(result_name)
                self.assertIsInstance(fn, ast.Lambda)
                self.assertIsInstance(fn.body, ast.Call)
                self.assertIsInstance(fn.body.func, ast.Attribute)
                self.assertEqual("load", fn.body.func.attr)
                keywords = {
                    keyword.arg: keyword.value for keyword in fn.body.keywords if keyword.arg
                }
                self.assertIn("file_load_mode", keywords)
                self.assertIsInstance(keywords["file_load_mode"], ast.Constant)
                self.assertEqual(expected_mode, keywords["file_load_mode"].value)

    def test_external_no_copy_load_benchmark_uses_no_copy_option(self):
        fn = _get_measure_call_callable("load/2filex1/onnxlight-nocopy")
        self.assertIsInstance(fn, ast.Lambda)
        self.assertIsInstance(fn.body, ast.Call)
        self.assertIsInstance(fn.body.func, ast.Attribute)
        self.assertEqual("load", fn.body.func.attr)
        keywords = {keyword.arg: keyword.value for keyword in fn.body.keywords if keyword.arg}
        self.assertIn("location", keywords)
        self.assertIn("no_copy", keywords)
        self.assertIn("touch_raw_data_pages", keywords)
        self.assertIsInstance(keywords["location"], ast.Name)
        self.assertEqual("ext_load_data", keywords["location"].id)
        self.assertIsInstance(keywords["no_copy"], ast.Constant)
        self.assertTrue(keywords["no_copy"].value)
        self.assertIsInstance(keywords["touch_raw_data_pages"], ast.Constant)
        self.assertTrue(keywords["touch_raw_data_pages"].value)

    def test_cpp_external_no_copy_load_benchmark_uses_cpp_example(self):
        call = _find_call("_measure_cpp_load_with_example", "ext_load_onnx")
        keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
        self.assertIn("file_count", keywords)
        self.assertIn("no_copy", keywords)
        self.assertIn("touch_raw_data_pages", keywords)
        self.assertIsInstance(keywords["file_count"], ast.Constant)
        self.assertEqual(2, keywords["file_count"].value)
        self.assertIsInstance(keywords["no_copy"], ast.Constant)
        self.assertTrue(keywords["no_copy"].value)
        self.assertIsInstance(keywords["touch_raw_data_pages"], ast.Constant)
        self.assertTrue(keywords["touch_raw_data_pages"].value)

    def test_find_standalone_executable_returns_none_in_ci_or_without_script_file(self):
        from onnx_light.doc import find_standalone_executable as find_executable

        with patch.dict(os.environ, {"CI": "yes"}, clear=False):
            found = find_executable(
                "load_onnx_time",
                [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                "some_script.py",
            )
        self.assertIsNone(found)

        with patch.dict(os.environ, {"CI": "0"}, clear=False), patch.object(shutil, "which"):
            found = find_executable(
                "load_onnx_time",
                [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                None,
            )
        self.assertIsNotNone(found)

    def test_find_standalone_executable_reason_out_in_ci(self):
        from onnx_light.doc import find_standalone_executable as find_executable

        reasons: list[str] = []
        with patch.dict(os.environ, {"CI": "true"}, clear=False):
            found = find_executable(
                "load_onnx_time",
                [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                "some_script.py",
                reason_out=reasons,
            )
        self.assertIsNone(found)
        self.assertEqual(1, len(reasons))
        self.assertIn("CI environment detected", reasons[0])
        self.assertIn("'true'", reasons[0])

    def test_find_standalone_executable_reason_out_when_missing(self):
        find_executable = _load_find_standalone_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            script_path.parent.mkdir(parents=True)
            script_path.write_text("", encoding="utf-8")
            reasons: list[str] = []
            with (
                patch.dict(os.environ, {"CI": "0"}, clear=False),
                patch.object(shutil, "which", return_value=None),
            ):
                found = find_executable(
                    "load_onnx_time",
                    [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                    str(script_path),
                    reason_out=reasons,
                )
        self.assertIsNone(found)
        self.assertEqual(1, len(reasons))
        self.assertIn("load_onnx_time", reasons[0])
        self.assertIn("not found on PATH", reasons[0])
        self.assertIn("build/examples/load_onnx_time/load_onnx_time", reasons[0])

    def test_find_standalone_executable_no_reason_when_found(self):
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
            reasons: list[str] = []
            with patch.dict(os.environ, {"CI": "0"}, clear=False):
                found = find_executable(
                    "load_onnx_time",
                    [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                    str(script_path),
                    reason_out=reasons,
                )
            self.assertIsNotNone(found)
            self.assertEqual([], reasons)

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

    def test_find_standalone_executable_without_script_file_uses_cwd_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_doc = pathlib.Path(tmp) / "site-packages" / "onnx_light" / "doc.py"
            fake_doc.parent.mkdir(parents=True)
            fake_doc.write_text("", encoding="utf-8")
            repo_root = pathlib.Path(tmp) / "repo"
            docs_dir = repo_root / "docs"
            docs_dir.mkdir(parents=True)
            fake_exe = repo_root / "build" / "examples" / "load_onnx_time" / "load_onnx_time"
            fake_exe.parent.mkdir(parents=True)
            fake_exe.write_text("", encoding="utf-8")
            find_executable = _load_find_standalone_executable(custom_file=str(fake_doc))

            with (
                patch.object(pathlib.Path, "cwd", return_value=docs_dir),
                patch.dict(os.environ, {"CI": "0"}, clear=False),
                patch.object(shutil, "which") as mocked_which,
            ):
                found = find_executable(
                    "load_onnx_time",
                    [pathlib.Path("build/examples/load_onnx_time/load_onnx_time")],
                    None,
                )

            self.assertIsNotNone(found)
            self.assertEqual(fake_exe.resolve(), pathlib.Path(found).resolve())
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
        namespace["_find_load_onnx_time_executable"] = lambda **_kw: "/tmp/load_onnx_time"
        namespace["_find_load_onnx_light_time_executable"] = lambda **_kw: "/tmp/load_onnx_light_time"
        stdout = "\n".join(
            [
                "Average load (ms): 10.0",
                "Median load (ms): 9.0",
                "Min load (ms): 8.0",
                "Max load (ms): 12.0",
                "Std load (ms): 1.5",
            ]
        )
        completed = subprocess.CompletedProcess(
            args=["/tmp/load_onnx_time", "model.onnx", "5", "4"], returncode=0, stdout=stdout
        )
        with patch.object(namespace["subprocess"], "run", return_value=completed) as mocked_run:
            got = measure_cpp("model.onnx", n=5, num_threads=4, executable_name="load_onnx_time")

        self.assertIsNotNone(got)
        self.assertEqual("load/1filex4/onnx-cpp", got["name"])
        self.assertEqual(0.010, got["avg"])
        self.assertEqual(0.009, got["median"])
        self.assertEqual(0.008, got["min"])
        self.assertEqual(0.012, got["max"])
        self.assertEqual(0.0015, got["std"])
        mocked_run.assert_called_once_with(
            ["/tmp/load_onnx_time", "model.onnx", "5", "4"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

    def test_measure_cpp_load_with_example_onnx_light_default(self):
        measure_cpp, namespace = _load_measure_cpp_load_with_example()
        namespace["_find_load_onnx_light_time_executable"] = lambda **_kw: "/tmp/load_onnx_light_time"
        stdout = "\n".join(
            [
                "Average load (ms): 20.0",
                "Median load (ms): 19.5",
                "Min load (ms): 18.0",
                "Max load (ms): 22.0",
                "Std load (ms): 1.25",
            ]
        )
        completed = subprocess.CompletedProcess(
            args=["/tmp/load_onnx_light_time", "model.onnx", "5", "1"],
            returncode=0,
            stdout=stdout,
        )
        with patch.object(namespace["subprocess"], "run", return_value=completed) as mocked_run:
            got = measure_cpp("model.onnx", n=5, num_threads=1)

        self.assertIsNotNone(got)
        self.assertEqual("load/1filex1/onnxlight-cpp", got["name"])
        self.assertEqual(0.020, got["avg"])
        self.assertEqual(0.0195, got["median"])
        self.assertEqual(0.018, got["min"])
        self.assertEqual(0.022, got["max"])
        self.assertEqual(0.00125, got["std"])
        mocked_run.assert_called_once_with(
            ["/tmp/load_onnx_light_time", "model.onnx", "5", "1"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

    def test_measure_cpp_load_with_example_onnx_light_external_no_copy(self):
        measure_cpp, namespace = _load_measure_cpp_load_with_example()
        namespace["_find_load_onnx_light_time_executable"] = lambda **_kw: "/tmp/load_onnx_light_time"
        stdout = "\n".join(
            [
                "Average load (ms): 7.0",
                "Median load (ms): 6.5",
                "Min load (ms): 6.0",
                "Max load (ms): 8.0",
                "Std load (ms): 0.5",
            ]
        )
        completed = subprocess.CompletedProcess(
            args=["/tmp/load_onnx_light_time", "model.onnx", "5", "1", "nocopy_touch"],
            returncode=0,
            stdout=stdout,
        )
        with patch.object(namespace["subprocess"], "run", return_value=completed) as mocked_run:
            got = measure_cpp(
                "model.onnx",
                n=5,
                num_threads=1,
                file_count=2,
                no_copy=True,
                touch_raw_data_pages=True,
            )

        self.assertIsNotNone(got)
        self.assertEqual("load/2filex1/onnxlight-cpp-nocopy", got["name"])
        self.assertEqual(0.007, got["avg"])
        self.assertEqual(0.0065, got["median"])
        self.assertEqual(0.006, got["min"])
        self.assertEqual(0.008, got["max"])
        self.assertEqual(0.0005, got["std"])
        mocked_run.assert_called_once_with(
            ["/tmp/load_onnx_light_time", "model.onnx", "5", "1", "nocopy_touch"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

    def test_measure_cpp_load_with_example_rejects_onnx_no_copy(self):
        measure_cpp, _ = _load_measure_cpp_load_with_example()
        with self.assertRaisesRegex(ValueError, "no_copy is only supported"):
            measure_cpp("model.onnx", executable_name="load_onnx_time", no_copy=True)

    def test_find_load_onnx_light_executable_in_examples_build_location(self):
        find_executable, namespace = _load_find_load_onnx_light_time_executable()
        with tempfile.TemporaryDirectory() as tmp:
            script_path = pathlib.Path(tmp) / "docs" / "examples" / "core" / "plot_onnx_time.py"
            executable = (
                pathlib.Path(tmp)
                / "build"
                / "examples"
                / "load_onnx_light_time"
                / "load_onnx_light_time"
            )
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            namespace["__file__"] = str(script_path)

            with patch.dict(namespace["os"].environ, {"CI": "0"}, clear=False):
                found = find_executable()

            self.assertIsNotNone(found)
            self.assertEqual(executable.resolve(), pathlib.Path(found).resolve())

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
        namespace["_find_save_onnx_light_time_executable"] = lambda **_kw: "/tmp/save_onnx_light_time"
        stdout = "\n".join(
            [
                "Average save (ms): 20.0",
                "Median save (ms): 19.0",
                "Min save (ms): 18.0",
                "Max save (ms): 25.0",
                "Std save (ms): 2.0",
            ]
        )
        completed = subprocess.CompletedProcess(
            args=["/tmp/save_onnx_light_time", "model.onnx", "/tmp/out", "5", "1", "onefile"],
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
        self.assertEqual("save/1filex1/onnxlight-cpp", got["name"])
        self.assertEqual(0.020, got["avg"])
        self.assertEqual(0.019, got["median"])
        self.assertEqual(0.018, got["min"])
        self.assertEqual(0.025, got["max"])
        self.assertEqual(0.002, got["std"])

    def test_measure_cpp_save_with_example_x4(self):
        measure_cpp, namespace = _load_measure_cpp_save_with_example()
        namespace["_find_save_onnx_light_time_executable"] = lambda **_kw: "/tmp/save_onnx_light_time"
        stdout = "\n".join(
            [
                "Average save (ms): 10.0",
                "Median save (ms): 9.0",
                "Min save (ms): 8.0",
                "Max save (ms): 12.0",
                "Std save (ms): 1.0",
            ]
        )
        completed = subprocess.CompletedProcess(
            args=["/tmp/save_onnx_light_time", "model.onnx", "/tmp/out", "5", "4", "onefile"],
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
        self.assertEqual("save/1filex4/onnxlight-cpp", got["name"])
        self.assertEqual(0.010, got["avg"])
        self.assertEqual(0.009, got["median"])
        self.assertEqual(0.008, got["min"])
        self.assertEqual(0.012, got["max"])
        self.assertEqual(0.001, got["std"])

    def test_measure_cpp_save_returns_none_when_executable_missing(self):
        measure_cpp, namespace = _load_measure_cpp_save_with_example()
        namespace["_find_save_onnx_light_time_executable"] = lambda: None
        got = measure_cpp("model.onnx", n=5, num_threads=1)
        self.assertIsNone(got)

    def test_parse_model_path_default(self):
        parse = _load_parse_model_path()
        got = parse([])
        self.assertIsNone(got)

    def test_parse_model_path_with_value(self):
        parse = _load_parse_model_path()
        got = parse(["--model", "/tmp/my_model.onnx"])
        self.assertEqual("/tmp/my_model.onnx", got)

    def test_parse_model_path_ignores_unknown_args(self):
        parse = _load_parse_model_path()
        got = parse(["--scenario", "load", "--model", "some.onnx"])
        self.assertEqual("some.onnx", got)

    def test_parse_model_id_default(self):
        parse = _load_parse_model_id()
        model_id, model_file = parse([])
        self.assertIsNone(model_id)
        self.assertEqual("onnx/model.onnx", model_file)

    def test_parse_model_id_with_value(self):
        parse = _load_parse_model_id()
        model_id, model_file = parse(["--model-id", "fxmarty/onnx-tiny-random-gpt2-with-merge"])
        self.assertEqual("fxmarty/onnx-tiny-random-gpt2-with-merge", model_id)
        self.assertEqual("onnx/model.onnx", model_file)

    def test_parse_model_id_with_custom_file(self):
        parse = _load_parse_model_id()
        model_id, model_file = parse(
            [
                "--model-id",
                "fxmarty/onnx-tiny-random-gpt2-with-merge",
                "--model-file",
                "decoder_model.onnx",
            ]
        )
        self.assertEqual("fxmarty/onnx-tiny-random-gpt2-with-merge", model_id)
        self.assertEqual("decoder_model.onnx", model_file)

    def test_download_hf_model_success(self):
        download, namespace = _load_download_hf_model()
        captured = {}

        def fake_urlretrieve(url, dest):
            captured["url"] = url
            captured["dest"] = dest
            with open(dest, "wb") as f:
                f.write(b"fake-onnx-bytes")

        namespace["urllib"].request.urlretrieve = fake_urlretrieve
        with tempfile.TemporaryDirectory() as tmp:
            got = download("some-org/some-model", "onnx/model.onnx", tmp)
            self.assertIsNotNone(got)
            self.assertTrue(os.path.exists(got))
            self.assertEqual("model.onnx", os.path.basename(got))
        self.assertEqual(
            "https://huggingface.co/some-org/some-model/resolve/main/onnx/model.onnx",
            captured["url"],
        )

    def test_download_hf_model_handles_network_error(self):
        import urllib.error

        download, namespace = _load_download_hf_model()

        def fake_urlretrieve(url, dest):
            raise urllib.error.URLError("connection refused")

        namespace["urllib"].request.urlretrieve = fake_urlretrieve
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                got = download("some-org/some-model", "onnx/model.onnx", tmp)
            self.assertIsNone(got)
            output = buf.getvalue()
            self.assertIn("WARNING", output)
            self.assertIn("Falling back", output)

    def test_download_hf_model_handles_os_error(self):
        download, namespace = _load_download_hf_model()

        def fake_urlretrieve(url, dest):
            raise OSError("disk full")

        namespace["urllib"].request.urlretrieve = fake_urlretrieve
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                got = download("some-org/some-model", "onnx/model.onnx", tmp)
            self.assertIsNone(got)
            output = buf.getvalue()
            self.assertIn("WARNING", output)
            self.assertIn("disk full", output)
            self.assertIn("Falling back", output)

    def test_print_model_stats_basic(self):
        import numpy

        print_stats = _load_print_model_stats()
        w = onh.from_array(numpy.ones((4, 4), dtype="float32"), name="W")
        graph = oh.make_graph(
            [oh.make_node("Relu", ["X"], ["Y"])],
            "g",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [4, 4])],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [4, 4])],
            initializer=[w],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_stats(model)
        output = buf.getvalue()
        self.assertIn("Number of nodes", output)
        self.assertIn(": 1", output)
        self.assertIn("Number of initializers", output)
        self.assertIn("Total initializer size", output)
        self.assertIn("IR version", output)
        self.assertIn("Opset(s)", output)

    def test_print_model_stats_with_file_path(self):
        print_stats = _load_print_model_stats()
        graph = oh.make_graph(
            [],
            "g",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1])],
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1])],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            onnxl.save(model, path)
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                print_stats(model, file_path=path)
            output = buf.getvalue()
        self.assertIn("File size", output)

    def test_print_model_stats_no_file_path(self):
        print_stats = _load_print_model_stats()
        graph = oh.make_graph(
            [],
            "g",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1])],
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1])],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_stats(model)
        output = buf.getvalue()
        self.assertNotIn("File size", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
