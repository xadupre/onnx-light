import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("onnxscript")
pytest.importorskip("onnx_light.onnx_core.graph_builder")

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "compute"
    / "plot_graph_builder_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("plot_graph_builder_benchmark", SOURCE)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


@pytest.mark.parametrize("count", (100, 200, 500))
def test_equivalent_models(count):
    example.check_models(count)


def test_timing_modes():
    for build in (example.build_light, example.build_onnxscript):
        assert example.measure(build, 2, False, repeats=1) >= 0
        assert example.measure(build, 2, True, repeats=1) >= 0
