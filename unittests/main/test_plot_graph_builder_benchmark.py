import gc
import importlib.util
from pathlib import Path

import matplotlib.pyplot
import pytest
import numpy

onnx = pytest.importorskip("onnx")
ReferenceEvaluator = importlib.import_module("onnx.reference").ReferenceEvaluator
pytest.importorskip("onnxscript")
pytest.importorskip("onnx_light.onnx_core.graph_builder")

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "builder"
    / "plot_graph_builder_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("plot_graph_builder_benchmark", SOURCE)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


@pytest.mark.parametrize("count", example.NODE_COUNTS)
def test_equivalent_models(count):
    example.check_models(count)


def test_timing_modes():
    for build in (example.build_light, example.build_onnxscript):
        assert example.measure(build, 20, False, repeats=1) >= 0
        assert example.measure(build, 20, True, repeats=1) >= 0


@pytest.mark.parametrize("build", (example.build_light, example.build_onnxscript))
def test_large_initializers_are_model_outputs(build, monkeypatch):
    monkeypatch.setattr(example, "LARGE_INITIALIZER_BYTES", 16)
    monkeypatch.setattr(example, "_LARGE_INITIALIZER", numpy.zeros(16, dtype=numpy.uint8))
    model = onnx.load_from_string(build(20, large_initializers=True).SerializeToString())
    for opset in model.opset_import:
        if opset.domain == "ai.onnx":
            opset.domain = ""
    onnx.checker.check_model(model)
    assert len(model.graph.initializer) == 5 + example.LARGE_INITIALIZER_COUNT
    assert len(model.graph.output) == 1 + example.LARGE_INITIALIZER_COUNT
    assert all(
        initializer.raw_data == bytes(16)
        for initializer in model.graph.initializer[-example.LARGE_INITIALIZER_COUNT :]
    )


@pytest.mark.parametrize("serialize", (False, True))
def test_timing_releases_each_model_between_samples(serialize):
    alive = [0]
    gc_was_enabled = gc.isenabled()

    class TrackedModel:
        def __init__(self):
            alive[0] += 1
            self.cycle = self

        def __del__(self):
            alive[0] -= 1

        def SerializeToString(self):
            return b"model"

    def build(_):
        assert alive[0] == 0
        assert not gc.isenabled()
        return TrackedModel()

    assert example.measure(build, 20, serialize, repeats=3) >= 0
    assert alive[0] == 0
    assert gc.isenabled() == gc_was_enabled


@pytest.mark.parametrize("build", (example.build_light, example.build_onnxscript))
@pytest.mark.parametrize("shape", example.INPUT_SHAPES)
def test_dynamic_attention(build, shape):
    model = onnx.load_from_string(build(20).SerializeToString())
    for opset in model.opset_import:
        if opset.domain == "ai.onnx":
            opset.domain = ""
    onnx.checker.check_model(model)
    assert [
        dim.dim_param for dim in model.graph.input[0].type.tensor_type.shape.dim
    ] == example.SHAPE
    assert len(model.graph.node) == 20
    assert len(model.graph.initializer) == 5
    x = numpy.random.default_rng(1).standard_normal(shape).astype(numpy.float32)
    query = x.reshape(*shape[:2], 2, shape[2] // 2).transpose(0, 2, 1, 3)
    scores = query @ query.transpose(0, 1, 3, 2) / numpy.sqrt(numpy.float32(shape[2] // 2))
    weights = numpy.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    context = (weights @ query).transpose(0, 2, 1, 3).reshape(shape)
    expected = numpy.maximum((x + context) * 0.5, 0)
    (actual,) = ReferenceEvaluator(model).run(None, {"X": x})
    numpy.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("build", (example.build_light, example.build_onnxscript))
@pytest.mark.parametrize("count", (0, -20, 21))
def test_invalid_node_counts(build, count):
    with pytest.raises(ValueError, match="positive multiple of 20"):
        build(count)


def test_node_type_distribution_table():
    distribution = example.node_type_distribution(example.build_light(20))
    assert distribution == {
        "Add": 1,
        "Cast": 1,
        "Concat": 1,
        "Div": 2,
        "Gather": 3,
        "MatMul": 2,
        "Mul": 1,
        "Relu": 1,
        "Reshape": 2,
        "Shape": 1,
        "Softmax": 1,
        "Sqrt": 1,
        "Transpose": 3,
    }
    table = example.format_node_type_table({20: distribution})
    assert "Gather             3" in table
    assert "Total             20" in table


def test_plot_benchmark():
    results = [
        {"nodes": count, "builder": builder, "model": count / 10, "serialized": count / 5}
        for count in (100, 200)
        for builder in ("onnx-light", "onnxscript")
    ]
    figure = example.plot_benchmark(results)
    assert len(figure.axes) == 2
    assert all(len(axis.lines) == 2 for axis in figure.axes)
    matplotlib.pyplot.close(figure)
