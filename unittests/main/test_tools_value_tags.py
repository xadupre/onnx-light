# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from unittest.mock import patch

from onnx_light.ext_test_case import ExtTestCase, HAS_OPTIM_EXT
from onnx_light.tools import (
    compute_value_and_node_tags,
    infer_value_and_node_tags,
    write_value_and_node_tags_to_metadata,
)
from onnx_light.tools._proto_utils import NODE_TAG_METADATA_KEY, VALUE_TAG_METADATA_KEY


def _meta_dict(proto_obj: object) -> dict[str, str]:
    return {m.key: m.value for m in getattr(proto_obj, "metadata_props", [])}


def _value_tags(graph: object) -> dict[str, str]:
    """Reconstructs the value-name -> tag map from per-value metadata.

    The value tags are stored on each ValueInfoProto (inputs, value_info,
    outputs) and initializer TensorProto, never on the graph metadata.
    """
    tags: dict[str, str] = {}
    for field in ("input", "value_info", "output", "initializer"):
        for value in getattr(graph, field, []):
            meta = _meta_dict(value)
            if VALUE_TAG_METADATA_KEY in meta:
                tags[value.name] = meta[VALUE_TAG_METADATA_KEY]
    return tags


class TestValueTagsErrors(ExtTestCase):
    def test_requires_cpp_bindings(self):
        from onnx_light.tools import _proto_utils

        with (
            patch.object(_proto_utils, "_shape_inference", None),
            self.assertRaisesRegex(
                RuntimeError, "onnx_light\\.onnx_py\\._onnxpycore.*is unavailable"
            ),
        ):
            compute_value_and_node_tags([])


@unittest.skipUnless(HAS_OPTIM_EXT, "requires onnx_light C++ shape_inference bindings")
class TestValueTags(ExtTestCase):
    def test_tags_graph_and_nodes(self):
        import onnx_light.onnx.helper as oh
        from onnx_light.onnx import TensorProto

        g = oh.make_graph(
            [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Reshape", ["X", "S"], ["Y"])],
            "g",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        )
        write_value_and_node_tags_to_metadata(g)
        tags = _value_tags(g)
        self.assertEqual(tags["X"], "weight")
        self.assertEqual(tags["Y"], "weight")
        self.assertEqual(_meta_dict(g.node[0])[NODE_TAG_METADATA_KEY], "shape")
        # The graph output Y should carry the per-ValueInfo value_tag.
        self.assertEqual(_meta_dict(g.output[0])[VALUE_TAG_METADATA_KEY], "weight")
        # The graph input X should carry the per-ValueInfo value_tag.
        self.assertEqual(_meta_dict(g.input[0])[VALUE_TAG_METADATA_KEY], "weight")

    def test_output_is_shape_tensor_gets_shape_tag(self):
        """When the model output is directly a shape tensor, it receives value_tag = 'shape'."""
        import onnx_light.onnx.helper as oh
        from onnx_light.onnx import TensorProto

        g = oh.make_graph(
            [oh.make_node("Shape", ["X"], ["Y"])],
            "g",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
            [oh.make_tensor_value_info("Y", TensorProto.INT64, [2])],
        )
        write_value_and_node_tags_to_metadata(g)
        tags = _value_tags(g)
        self.assertEqual(tags["X"], "weight")
        self.assertEqual(tags["Y"], "shape")
        self.assertEqual(_meta_dict(g.node[0])[NODE_TAG_METADATA_KEY], "shape")
        # The graph output Y must carry onnx_light.value_tag = "shape".
        self.assertEqual(_meta_dict(g.output[0])[VALUE_TAG_METADATA_KEY], "shape")

    def test_tags_subgraph(self):
        import onnx_light.onnx.helper as oh
        from onnx_light.onnx import TensorProto

        body = oh.make_graph(
            [oh.make_node("Shape", ["A"], ["SA"])],
            "body",
            [oh.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2])],
            [oh.make_tensor_value_info("SA", TensorProto.INT64, [2])],
        )
        if_node = oh.make_node("If", ["cond"], ["Y"])
        if_node.attribute.append(oh.make_attribute("then_branch", body))
        g = oh.make_graph(
            [if_node],
            "g",
            [oh.make_tensor_value_info("cond", TensorProto.BOOL, [])],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
        )
        write_value_and_node_tags_to_metadata(g)
        subgraph = g.node[0].attribute[0].g
        body_tags = _value_tags(subgraph)
        self.assertEqual(body_tags["SA"], "shape")

    def test_accepts_list_of_nodes_and_function(self):
        import onnx_light.onnx.helper as oh

        nodes = [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Expand", ["Y", "S"], ["Z"])]
        value_tags, node_tags = compute_value_and_node_tags(nodes)
        self.assertEqual(value_tags["S"], "shape")
        self.assertEqual(node_tags[0], "shape")

        function = oh.make_function("", "f", ["X", "Y"], ["Z"], nodes, [oh.make_opsetid("", 18)])
        write_value_and_node_tags_to_metadata(function)
        self.assertEqual(_meta_dict(function.node[0])[NODE_TAG_METADATA_KEY], "shape")

    def test_compute_value_and_node_tags_accepts_verbose(self):
        import onnx_light.onnx.helper as oh

        nodes = [oh.make_node("Shape", ["X"], ["S"])]
        value_tags, node_tags = compute_value_and_node_tags(nodes, verbose=1)
        self.assertEqual(value_tags["S"], "shape")
        self.assertEqual(node_tags, ["shape"])

    def test_accepts_repeated_proto_field_of_nodes(self):
        import onnx_light.onnx.helper as oh
        from onnx_light.onnx import TensorProto

        graph = oh.make_graph(
            [oh.make_node("Shape", ["X"], ["S"]), oh.make_node("Expand", ["Y", "S"], ["Z"])],
            "g",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
            [oh.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 2])],
        )

        value_tags, node_tags = compute_value_and_node_tags(graph.node)
        self.assertEqual(value_tags["S"], "shape")
        self.assertEqual(node_tags[0], "shape")

        write_value_and_node_tags_to_metadata(graph.node)
        self.assertEqual(_meta_dict(graph.node[0])[NODE_TAG_METADATA_KEY], "shape")

    def test_reshape_shape_tag_propagates_backward(self):
        import onnx_light.onnx.helper as oh

        nodes = [
            oh.make_node("Shape", ["X"], ["S0"]),
            oh.make_node("Identity", ["S0"], ["S1"]),
            oh.make_node("Reshape", ["X", "S1"], ["Y"]),
        ]
        value_tags, node_tags = compute_value_and_node_tags(nodes)
        self.assertEqual(value_tags["S1"], "shape")
        self.assertEqual(value_tags["S0"], "shape")
        self.assertEqual(node_tags[1], "shape")

    def test_conflicting_shape_axes_input_tag_becomes_ambiguous(self):
        import onnx_light.onnx.helper as oh

        nodes = [
            oh.make_node("Reshape", ["X", "IDX"], ["A"]),
            oh.make_node("ReduceSum", ["A", "IDX"], ["Y"]),
        ]
        value_tags, _ = infer_value_and_node_tags(nodes)
        self.assertEqual(value_tags["IDX"], "ambiguous")

    def test_constant_feeding_reshape_shape_input(self):
        import onnx_light.onnx.helper as oh
        from onnx_light.onnx import TensorProto

        nodes = [
            oh.make_node(
                "Constant",
                [],
                ["S"],
                value=oh.make_tensor("shape", TensorProto.INT64, [2], [2, 2]),
            ),
            oh.make_node("Reshape", ["X", "S"], ["Y"]),
        ]
        value_tags, node_tags = compute_value_and_node_tags(nodes)
        self.assertEqual(value_tags["S"], "shape")
        self.assertEqual(node_tags[0], "shape")

    def test_all_graph_inputs_are_seeded_as_weight(self):
        import onnx_light.onnx.helper as oh
        from onnx_light.onnx import TensorProto

        g = oh.make_graph(
            [],
            "g",
            [
                oh.make_tensor_value_info("W", TensorProto.FLOAT, [4, 3]),
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 3]),
                oh.make_tensor_value_info("I", TensorProto.INT64, [4, 3]),
            ],
            [],
        )
        value_tags, _ = compute_value_and_node_tags(g)
        self.assertEqual(value_tags.get("W"), "weight")
        self.assertEqual(value_tags.get("X"), "weight")
        self.assertEqual(value_tags.get("I"), "weight")

    def test_infer_alias_keeps_backward_compatibility(self):
        import onnx_light.onnx.helper as oh

        nodes = [oh.make_node("Shape", ["X"], ["S"])]
        self.assertEqual(compute_value_and_node_tags(nodes), infer_value_and_node_tags(nodes))


if __name__ == "__main__":
    unittest.main()
