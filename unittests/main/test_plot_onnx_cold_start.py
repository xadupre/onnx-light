import importlib.util
import pathlib
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
    """Loads the gallery example without invoking its main entry point."""
    spec = importlib.util.spec_from_file_location("plot_onnx_cold_start", _EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            result = _load_example_helpers()._run_sample("onnx_light", str(model_path))

        self.assertEqual(result["implementation"], "onnx_light")
        self.assertGreater(result["end_to_end_ms"], 0)
        self.assertGreater(result["first_load_after_imports_ms"], 0)
        self.assertIn("peak_rss_kib", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
