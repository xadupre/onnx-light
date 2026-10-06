import importlib.util
from pathlib import Path

import matplotlib.pyplot
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnx_light.onnx_core.graph_builder")

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "builder"
    / "plot_pattern_fusion_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("plot_pattern_fusion_benchmark", SOURCE)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_fusion_models_and_timings():
    example.check_models(2)
    light = example.build_light(2)
    scripted = example.build_onnxscript(2)
    assert example.measure_construction(example.build_light, 2, repeats=1) >= 0
    assert example.measure_construction(example.build_onnxscript, 2, repeats=1) >= 0
    assert example.measure_light_fusion(light, repeats=1) >= 0
    assert example.measure_onnxscript_fusion(scripted, repeats=1) >= 0


def test_fusion_patterns_include_six_node_rule():
    six_node = next(
        pattern for pattern in example.light_patterns() if pattern.name == "SixNodeFusion"
    )
    assert six_node.op_types == ("Sub", "Abs", "Sqrt", "Neg", "Exp", "Log")


def test_plot_benchmark():
    results = [
        {
            "patterns": count,
            "implementation": implementation,
            "construction": count / 10,
            "fusion": count / 20,
        }
        for count in (100, 200)
        for implementation in ("onnx-light", "onnxscript")
    ]
    figure = example.plot_benchmark(results)
    assert len(figure.axes) == 2
    assert all(len(axis.lines) == 2 for axis in figure.axes)
    matplotlib.pyplot.close(figure)
