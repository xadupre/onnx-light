import ast
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch


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
    namespace = {"os": os, "pathlib": pathlib, "shutil": shutil}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_find_load_onnx_time_executable"], namespace


class TestPlotOnnxTime(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
