import ast
import os
import pathlib
import tempfile
import time
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
    / "plot_onnx_time.py"
)


def _load_example_helpers():
    """Loads the standalone helper functions from plot_onnx_time.py.

    The example script executes benchmark code at import time, so only the
    relevant top-level function definitions are compiled and executed in an
    isolated namespace with the imports they rely on.

    Returns:
        A namespace dict exposing ``make_model``, ``_save_default_model`` and
        ``_model_has_external_data``.
    """
    tree = ast.parse(_EXAMPLE_PATH.read_text(encoding="utf-8"), filename=str(_EXAMPLE_PATH))
    wanted = {"make_model", "_save_default_model", "_model_has_external_data", "measure"}
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "onnxl": onnxl,
        "oh": oh,
        "onh": onh,
        "np": np,
        "os": os,
        "N_INIT": 2,
        "DIM": 4,
        "MAX_MEASURE_DURATION": 2.0,
        "MIN_MEASURE_ITERATIONS": 3,
        "time": time,
    }
    exec(compile(module, filename=str(_EXAMPLE_PATH), mode="exec"), namespace)
    return namespace


class TestPlotOnnxTime(unittest.TestCase):
    def test_measure_collects_minimum_iterations(self):
        """Verifies the duration bound does not produce single-sample results."""
        helpers = _load_example_helpers()
        calls = 0

        def increment():
            nonlocal calls
            calls += 1

        stats = helpers["measure"](
            "increment", increment, n=5, warmup=0, max_duration=0, min_iterations=3
        )
        self.assertEqual(calls, 3)
        self.assertEqual(stats["name"], "increment")

    def test_save_default_model_external(self):
        """Verifies the default synthetic model can be saved with external weights."""
        helpers = _load_example_helpers()
        model = helpers["make_model"](n_init=2, dim=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = helpers["_save_default_model"](model, tmp_dir, external=True)
            self.assertTrue(os.path.exists(onnx_path))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "bench.onnx.data")))

            reloaded = onnxl.load(onnx_path, load_external_data=False)
            self.assertTrue(helpers["_model_has_external_data"](reloaded))

            loaded = onnxl.load(onnx_path, load_external_data=True)
            # With the weights loaded in memory the serialized size grows and
            # ``ByteSize`` reflects the actual model size (unlike a model whose
            # weights are still stored externally).
            self.assertGreater(loaded.ByteSize(), reloaded.ByteSize())
            for init in loaded.graph.initializer:
                self.assertTrue(bool(init.raw_data))

    def test_save_default_model_single_file(self):
        """Verifies the default synthetic model has no external weights by default."""
        helpers = _load_example_helpers()
        model = helpers["make_model"](n_init=2, dim=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = helpers["_save_default_model"](model, tmp_dir, external=False)
            self.assertTrue(os.path.exists(onnx_path))
            self.assertFalse(os.path.exists(os.path.join(tmp_dir, "bench.onnx.data")))
            loaded = onnxl.load(onnx_path)
            self.assertFalse(helpers["_model_has_external_data"](loaded))


if __name__ == "__main__":
    unittest.main(verbosity=2)
