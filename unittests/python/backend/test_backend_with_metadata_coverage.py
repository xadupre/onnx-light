import unittest

from onnx_light.ext_test_case import ExtTestCase, import_or_skip

# The backend test registries are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
collect_test_cases_by_name = import_or_skip(
    "onnx_light.onnx.backend", "collect_test_cases_by_name"
)


def _metadata_map(entries):
    return {entry.key: entry.value for entry in entries}


def _check_inplace_test_case(tc):
    assert tc.tag == "inplace", f"Unexpected tag for {tc.name!r}: {tc.tag!r}"
    assert tc.model is not None, f"Missing model for {tc.name!r}"
    nodes = tc.model.graph.node
    assert len(nodes) == 3, f"Unexpected node count for {tc.name!r}: {len(nodes)}"
    assert _metadata_map(nodes[1].metadata_props) == {
        "onnx_light.inplace_reuse": "0:0:equal",
        "onnx_light.release_after": "A",
    }, f"Unexpected metadata on node 1 for {tc.name!r}"
    assert _metadata_map(nodes[2].metadata_props) == {
        "onnx_light.inplace_reuse": "0:0:equal",
        "onnx_light.release_after": "B",
    }, f"Unexpected metadata on node 2 for {tc.name!r}"


def _check_release_test_case(tc):
    assert tc.tag == "release", f"Unexpected tag for {tc.name!r}: {tc.tag!r}"
    assert tc.model is not None, f"Missing model for {tc.name!r}"
    nodes = tc.model.graph.node
    assert len(nodes) == 2, f"Unexpected node count for {tc.name!r}: {len(nodes)}"
    assert (
        _metadata_map(nodes[0].metadata_props) == {}
    ), f"Unexpected metadata on node 0 for {tc.name!r}"
    assert _metadata_map(nodes[1].metadata_props) == {
        "onnx_light.release_after": "S"
    }, f"Unexpected metadata on node 1 for {tc.name!r}"


def _check_shape_tag_test_case(tc):
    assert tc.tag == "shape_tag", f"Unexpected tag for {tc.name!r}: {tc.tag!r}"
    assert tc.model is not None, f"Missing model for {tc.name!r}"
    graph = tc.model.graph
    assert _metadata_map(graph.metadata_props) == {
        "onnx_light.value_tags": '{"S":"shape","X":"weight","Y":"weight"}'
    }, f"Unexpected graph metadata for {tc.name!r}"
    nodes = graph.node
    assert len(nodes) == 2, f"Unexpected node count for {tc.name!r}: {len(nodes)}"
    assert _metadata_map(nodes[0].metadata_props) == {
        "onnx_light.node_tag": "shape"
    }, f"Unexpected metadata on node 0 for {tc.name!r}"
    assert _metadata_map(nodes[1].metadata_props) == {
        "onnx_light.node_tag": "weight"
    }, f"Unexpected metadata on node 1 for {tc.name!r}"


class TestBackendMetadataCoverage(ExtTestCase):
    def _get_test_case(self, pattern):
        cases = collect_test_cases_by_name(pattern)
        self.assertEqual(len(cases), 1)
        return cases[0]

    def test_cc_shape_inference_inplace_reuse(self):
        _check_inplace_test_case(self._get_test_case("^test_cc_shape_inference_inplace_reuse$"))

    def test_cc_release_shape_reshape(self):
        _check_release_test_case(self._get_test_case("^test_cc_release_shape_reshape$"))

    def test_cc_shape_tag_shape_reshape(self):
        _check_shape_tag_test_case(self._get_test_case("^test_cc_shape_tag_shape_reshape$"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
