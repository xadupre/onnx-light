# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from onnx_light.tools import infer_value_and_node_tags, write_value_and_node_tags_to_metadata
from onnx_light.tools._proto_utils import NODE_TAG_METADATA_KEY, VALUE_TAGS_METADATA_KEY


def _meta_dict(obj: SimpleNamespace) -> dict[str, str]:
    return {m.key: m.value for m in getattr(obj, "metadata_props", [])}


def _node(
    op_type: str, inputs: list[str], outputs: list[str], attrs: list | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        op_type=op_type,
        input=list(inputs),
        output=list(outputs),
        name="",
        attribute=attrs or [],
        metadata_props=[],
    )


def _vi(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, metadata_props=[], type=None)


def _graph(
    nodes: list[SimpleNamespace], inputs: list[SimpleNamespace], outputs: list[SimpleNamespace]
):
    return SimpleNamespace(
        node=nodes, input=inputs, output=outputs, initializer=[], value_info=[], metadata_props=[]
    )


class TestValueTags(unittest.TestCase):
    def test_tags_graph_and_nodes(self):
        g = _graph(
            [_node("Shape", ["X"], ["S"]), _node("Reshape", ["X", "S"], ["Y"])],
            [_vi("X")],
            [_vi("Y")],
        )
        write_value_and_node_tags_to_metadata(g)
        tags = json.loads(_meta_dict(g)[VALUE_TAGS_METADATA_KEY])
        self.assertEqual(tags["S"], "shape")
        self.assertEqual(_meta_dict(g.node[0])[NODE_TAG_METADATA_KEY], "shape")

    def test_tags_subgraph(self):
        body = _graph([_node("Shape", ["A"], ["SA"])], [_vi("A")], [_vi("SA")])
        attr = SimpleNamespace(g=body, graphs=[])
        g = _graph([_node("If", ["cond"], ["Y"], attrs=[attr])], [_vi("cond")], [_vi("Y")])
        write_value_and_node_tags_to_metadata(g)
        body_tags = json.loads(_meta_dict(body)[VALUE_TAGS_METADATA_KEY])
        self.assertEqual(body_tags["SA"], "shape")

    def test_accepts_list_of_nodes_and_function(self):
        nodes = [_node("Shape", ["X"], ["S"]), _node("Expand", ["Y", "S"], ["Z"])]
        value_tags, node_tags = infer_value_and_node_tags(nodes)
        self.assertEqual(value_tags["S"], "shape")
        self.assertEqual(node_tags[0], "shape")

        function = SimpleNamespace(node=nodes, input=["X"], output=["Z"], metadata_props=[])
        write_value_and_node_tags_to_metadata(function)
        self.assertIn(VALUE_TAGS_METADATA_KEY, _meta_dict(function))


if __name__ == "__main__":
    unittest.main()
