import importlib.util
import pathlib
import types
import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip


def _load_example_module():
    import_or_skip("onnx_light.onnx.backend")
    root = pathlib.Path(__file__).resolve().parents[2]
    source_path = root / "docs" / "examples" / "compute" / "plot_qwen3_compute_context_memory.py"
    module_spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    if module_spec is None or module_spec.loader is None:
        raise FileNotFoundError(source_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class TestPlotQwen3ComputeContextMemory(ExtTestCase):
    def test_make_plot_assignments(self):
        module = _load_example_module()
        args = types.SimpleNamespace(batch=3, sequence_length=16, past_sequence_length=8)

        got = module.make_plot_assignments(args)

        self.assertEqual(
            got,
            [
                (
                    "current (past=8, seq=16)",
                    {
                        "batch_size": 3,
                        "sequence_length": 16,
                        "past_sequence_length": 8,
                        "total_sequence_length": 24,
                    },
                ),
                (
                    "past=0, seq=128",
                    {
                        "batch_size": 3,
                        "sequence_length": 128,
                        "past_sequence_length": 0,
                        "total_sequence_length": 128,
                    },
                ),
                (
                    "past=129, seq=1",
                    {
                        "batch_size": 3,
                        "sequence_length": 1,
                        "past_sequence_length": 129,
                        "total_sequence_length": 130,
                    },
                ),
                (
                    "past=256, seq=1",
                    {
                        "batch_size": 3,
                        "sequence_length": 1,
                        "past_sequence_length": 256,
                        "total_sequence_length": 257,
                    },
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
