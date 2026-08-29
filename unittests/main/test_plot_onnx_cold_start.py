import ast
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh

_EXAMPLE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "proto"
    / "plot_onnx_cold_start.py"
)


def _load_example_helpers():
    """Loads selected helper functions without executing the gallery example."""
    tree = ast.parse(_EXAMPLE_PATH.read_text(encoding="utf-8"), filename=str(_EXAMPLE_PATH))
    wanted = {"_run_sample"}
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ],
        type_ignores=[],
    )
    namespace = {
        "__file__": str(_EXAMPLE_PATH),
        "json": json,
        "pathlib": pathlib,
        "subprocess": subprocess,
        "sys": sys,
    }
    exec(compile(module, filename=str(_EXAMPLE_PATH), mode="exec"), namespace)
    return namespace


class TestPlotOnnxColdStart(unittest.TestCase):
    def test_onnx_light_sample_uses_fresh_process_protocol(self):
        """Verifies a sample reports separate startup and post-import load timing."""
        graph = oh.make_graph(
            [oh.make_node("Identity", ["X"], ["Y"])],
            "cold_start_test",
            [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [1])],
            [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [1])],
            initializer=[onh.from_array(np.array([1], dtype=np.float32), name="weight")],
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory) / "model.onnx"
            onnxl.save(oh.make_model(graph), model_path)
            result = _load_example_helpers()["_run_sample"]("onnx_light", str(model_path))

        self.assertEqual(result["implementation"], "onnx_light")
        self.assertGreater(result["end_to_end_ms"], 0)
        self.assertGreater(result["first_load_after_imports_ms"], 0)
        self.assertIn("peak_rss_kib", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
