import unittest

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl
import onnx_light.onnx_core.shape_inference as shape_inference

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
make_test_class = import_or_skip("onnx_light.onnx.backend", "make_test_class")
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")

_EXPECTED_METADATA_KEYS = {
    shape_inference.INPLACE_REUSE_METADATA_KEY,
    shape_inference.NODE_TAG_METADATA_KEY,
    shape_inference.RELEASE_AFTER_METADATA_KEY,
    shape_inference.RELEASE_AFTER_SHAPE_TAG_METADATA_KEY,
    shape_inference.VALUE_TAG_METADATA_KEY,
}


def _metadata_map(entries):
    return {entry.key: entry.value for entry in entries}


def _expected_metadata_map(entries):
    return {entry.key: entry.value for entry in entries if entry.key in _EXPECTED_METADATA_KEYS}


def _value_metadata(graph):
    return {
        field: [_expected_metadata_map(value.metadata_props) for value in getattr(graph, field)]
        for field in ("input", "value_info", "output", "initializer")
    }


def _has_expected_metadata(graph_metadata, node_metadata, value_metadata):
    return (
        bool(graph_metadata)
        or any(node_metadata)
        or any(any(item for item in entries) for entries in value_metadata.values())
    )


def _clear_metadata(model):
    model.graph.metadata_props.clear()
    for node in model.graph.node:
        node.metadata_props.clear()
    for field in ("input", "value_info", "output", "initializer"):
        for value in getattr(model.graph, field):
            value.metadata_props.clear()


def _assert_metadata_matches(name, location, expected, computed):
    for key, expected_value in expected.items():
        assert key in computed, f"Missing metadata {key!r} on {location} for {name!r}"
        computed_value = computed[key]
        assert computed_value == expected_value, (
            f"Unexpected metadata {key!r} on {location} for {name!r}: "
            f"{computed_value!r} != {expected_value!r}"
        )


def metadata_coverage_check(model: onnxl.ModelProto, *_inputs):
    assert isinstance(model, onnxl.ModelProto), f"Unexpected type {type(model)}"
    name = model.graph.name
    expected_graph = _expected_metadata_map(model.graph.metadata_props)
    expected_nodes = [_expected_metadata_map(node.metadata_props) for node in model.graph.node]
    expected_values = _value_metadata(model.graph)
    model_copy = onnxl.ModelProto()
    model_copy.CopyFrom(model)
    _clear_metadata(model_copy)

    ctx = shape_inference.ShapesContext()
    shape_inference.compute_shape_model(ctx, model_copy)
    value_tags, _ = shape_inference.compute_value_and_node_tags(model_copy.graph)
    shape_inference.write_value_and_node_tags_to_metadata(model_copy.graph)
    shape_inference.write_inplace_reuse_to_metadata(ctx, model_copy.graph, value_tags=value_tags)

    if not _has_expected_metadata(expected_graph, expected_nodes, expected_values):
        return

    _assert_metadata_matches(
        name, "graph", expected_graph, _metadata_map(model_copy.graph.metadata_props)
    )
    for i, (expected, node) in enumerate(zip(expected_nodes, model_copy.graph.node)):
        _assert_metadata_matches(name, f"node {i}", expected, _metadata_map(node.metadata_props))
    for field, expected_entries in expected_values.items():
        values = getattr(model_copy.graph, field)
        for i, expected in enumerate(expected_entries):
            _assert_metadata_matches(
                name, f"{field} {i}", expected, _metadata_map(values[i].metadata_props)
            )


TestBackendMetadataCoverage = make_test_class(
    metadata_coverage_check,
    include_regex=["^test_cc_shape_inference_.*", "^test_cc_release_.*", "^test_cc_shape_tag_.*"],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
