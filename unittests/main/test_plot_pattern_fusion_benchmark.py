import importlib.util
from pathlib import Path

import matplotlib.pyplot
import numpy
import pytest

onnx = pytest.importorskip("onnx")
onnx_ir = pytest.importorskip("onnx_ir")
ReferenceEvaluator = pytest.importorskip("onnx.reference").ReferenceEvaluator
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
    assert example.measure_light_fusion(light, repeats=1, parallel=True) >= 0
    assert example.measure_onnxscript_fusion(scripted, repeats=1) >= 0


def test_fusion_patterns_include_six_node_rule():
    six_node = next(
        pattern for pattern in example.light_patterns() if pattern.name == "SixNodeFusion"
    )
    assert six_node.op_types == ("Sub", "Abs", "Sqrt", "Neg", "Exp", "Log")


@pytest.mark.parametrize("block_count", [1, 32])
def test_native_fusion_parallel_matches_sequential_and_onnxscript(block_count):
    model = example.build_transpose_model(block_count)
    original = onnx.load_from_string(model.SerializeToString())
    onnx.checker.check_model(original)
    feeds = {"X": numpy.random.default_rng(0).standard_normal((16, 16, 16)).astype(numpy.float32)}
    expected = ReferenceEvaluator(original).run(None, feeds)[0]
    plans = []
    for parallel in (False, True):
        builder = example.OptimizationGraphBuilder(model)
        plan = example.GraphGraph(builder, [example.TransposeTransposePattern()]).optimize(
            parallel=parallel
        )
        plans.append(plan.SerializeToString())
        assert len(plan.rewrites) == block_count
        optimized = onnx.load_from_string(builder.to_onnx().SerializeToString())
        for opset in optimized.opset_import:
            if opset.domain == "ai.onnx":
                opset.domain = ""
        onnx.checker.check_model(optimized)
        assert len(optimized.graph.node) == 2 * block_count
        numpy.testing.assert_array_equal(
            ReferenceEvaluator(optimized).run(None, feeds)[0], expected
        )
        assert example.measure_light_fusion(model, repeats=1, native=True, parallel=parallel) >= 0
    assert plans[0] == plans[1]
    scripted = onnx_ir.from_proto(original)
    assert example.ONNXSCRIPT_TRANSPOSE_RULES.apply_to_model(scripted) == block_count
    optimized = onnx_ir.to_proto(scripted)
    assert len(optimized.graph.node) == 2 * block_count
    numpy.testing.assert_array_equal(ReferenceEvaluator(optimized).run(None, feeds)[0], expected)
    assert example.measure_onnxscript_fusion(model, repeats=1, native=True) >= 0


@pytest.mark.parametrize("block_count", [0, -1])
def test_transpose_model_requires_positive_block_count(block_count):
    with pytest.raises(ValueError, match="block_count must be positive"):
        example.build_transpose_model(block_count)


def test_plot_benchmark():
    results = [
        {
            "patterns": count,
            "native_patterns": count // 4,
            "implementation": implementation,
            "construction": count / 10,
            "fusion": count / 20,
            "native_fusion": count / 30,
        }
        for count in (100, 200)
        for implementation in ("onnx-light", "onnx-light (parallel)", "onnxscript")
    ]
    figure = example.plot_benchmark(results)
    assert len(figure.axes) == 3
    assert [len(axis.lines) for axis in figure.axes] == [2, 3, 3]
    assert [line.get_label() for line in figure.axes[2].lines] == [
        "onnx-light",
        "onnx-light (parallel)",
        "onnxscript",
    ]
    assert list(figure.axes[2].lines[0].get_xdata()) == [25, 50]
    matplotlib.pyplot.close(figure)
