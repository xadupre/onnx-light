import unittest

from onnx_light.ext_test_case import import_or_skip

import onnx_light.onnx as onnxl

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
collect_test_case = import_or_skip("onnx_light.onnx.backend", "collect_test_case")
make_test_class = import_or_skip("onnx_light.onnx.backend", "make_test_class")


def _metadata_map(entries):
    return {entry.key: entry.value for entry in entries}


def _check_inplace_model(name, model):
    nodes = model.graph.node
    assert len(nodes) == 3, f"Unexpected node count for {name!r}: {len(nodes)}"
    assert _metadata_map(nodes[1].metadata_props) == {
        "onnx_light.inplace_reuse": "0:0:equal",
        "onnx_light.release_after": "A",
    }, f"Unexpected metadata on node 1 for {name!r}"
    assert _metadata_map(nodes[2].metadata_props) == {
        "onnx_light.inplace_reuse": "0:0:equal",
        "onnx_light.release_after": "B",
    }, f"Unexpected metadata on node 2 for {name!r}"


def _check_release_model(name, model):
    nodes = model.graph.node
    assert len(nodes) == 2, f"Unexpected node count for {name!r}: {len(nodes)}"
    assert (
        _metadata_map(nodes[0].metadata_props) == {}
    ), f"Unexpected metadata on node 0 for {name!r}"
    assert _metadata_map(nodes[1].metadata_props) == {
        "onnx_light.release_after": "S"
    }, f"Unexpected metadata on node 1 for {name!r}"


def _check_shape_tag_model(name, model):
    graph = model.graph
    assert _metadata_map(graph.metadata_props) == {
        "onnx_light.value_tags": '{"S":"shape","X":"weight","Y":"weight"}'
    }, f"Unexpected graph metadata for {name!r}"
    nodes = graph.node
    assert len(nodes) == 2, f"Unexpected node count for {name!r}: {len(nodes)}"
    assert _metadata_map(nodes[0].metadata_props) == {
        "onnx_light.node_tag": "shape"
    }, f"Unexpected metadata on node 0 for {name!r}"
    assert _metadata_map(nodes[1].metadata_props) == {
        "onnx_light.node_tag": "weight"
    }, f"Unexpected metadata on node 1 for {name!r}"


_EXPECTED_TAGS = {
    "test_cc_shape_inference_inplace_reuse": "inplace",
    "test_cc_release_shape_reshape": "release",
    "test_cc_shape_tag_shape_reshape": "shape_tag",
}

_METADATA_CHECKS = {
    "test_cc_shape_inference_inplace_reuse": _check_inplace_model,
    "test_cc_release_shape_reshape": _check_release_model,
    "test_cc_shape_tag_shape_reshape": _check_shape_tag_model,
}


def metadata_coverage_check(model: onnxl.ModelProto, *inputs):
    del inputs
    assert isinstance(model, onnxl.ModelProto), f"Unexpected type {type(model)}"
    name = model.graph.name
    assert name in _EXPECTED_TAGS, f"Unexpected model name {name!r}"
    test_case = collect_test_case()[name]
    assert (
        test_case.tag == _EXPECTED_TAGS[name]
    ), f"Unexpected tag for {name!r}: {test_case.tag!r}"
    _METADATA_CHECKS[name](name, model)


TestBackendMetadataCoverage = make_test_class(
    metadata_coverage_check,
    include_regex=[
        "^test_cc_shape_inference_inplace_reuse$",
        "^test_cc_release_shape_reshape$",
        "^test_cc_shape_tag_shape_reshape$",
    ],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
