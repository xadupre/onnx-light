import importlib.util
import pathlib
import unittest

from onnx_light.ext_test_case import import_or_skip


def _load_example_module():
    import_or_skip("onnx_light.onnx.backend")
    import_or_skip("onnxruntime")
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "runtime" / "plot_qwen3_init_benchmark.py"
    module_spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    if module_spec is None or module_spec.loader is None:
        raise FileNotFoundError(source_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class TestPlotQwen3InitBenchmark(unittest.TestCase):
    def test_measure_reports_expected_statistics(self):
        module = _load_example_module()
        counter = {"calls": 0}

        def increment():
            counter["calls"] += 1

        stats = module.measure("noop", increment, n=3, warmup=2)

        # 2 warm-up + 3 measured iterations.
        self.assertEqual(counter["calls"], 5)
        self.assertEqual(stats["name"], "noop")
        for key in ("median", "avg", "min", "max", "std"):
            self.assertIn(key, stats)
            self.assertGreaterEqual(stats[key], 0.0)
        self.assertLessEqual(stats["min"], stats["median"])
        self.assertLessEqual(stats["median"], stats["max"])

    def test_materialize_random_weights_fills_only_empty_initializers(self):
        module = _load_example_module()
        import numpy as np
        from onnx_light.onnx.helper import make_graph, make_model, make_opsetid
        from onnx_light.onnx.numpy_helper import from_array, to_array

        constant = from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="constant")
        weight = from_array(np.zeros((2, 3), dtype=np.float16), name="weight")
        # Turn ``weight`` into a metadata-only initializer (shape/dtype only).
        weight.raw_data = b""

        graph = make_graph([], "g", [], [], initializer=[constant, weight])
        model = make_model(graph, opset_imports=[make_opsetid("", 21)])

        self.assertFalse(module.initializer_is_metadata_only(model.graph.initializer[0]))
        self.assertTrue(module.initializer_is_metadata_only(model.graph.initializer[1]))

        count = module.materialize_random_weights(model, seed=0)

        self.assertEqual(count, 1)
        # The constant initializer keeps its original data.
        np.testing.assert_array_equal(
            to_array(model.graph.initializer[0]), np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        # The weight initializer is now materialized with the declared shape.
        materialized = to_array(model.graph.initializer[1])
        self.assertEqual(materialized.shape, (2, 3))
        self.assertFalse(module.initializer_is_metadata_only(model.graph.initializer[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
