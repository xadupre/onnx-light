# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import unittest
from unittest.mock import patch

from onnx_light.tools import infer_value_and_node_tags, write_value_and_node_tags_to_metadata
from onnx_light.tools import _proto_utils
from onnx_light.tools._proto_utils import NODE_TAG_METADATA_KEY, VALUE_TAGS_METADATA_KEY

HAS_OPTIM_EXT = importlib.util.find_spec("onnx_light.onnx_py._onnxpyoptim") is not None


def _meta_dict(proto_obj: object) -> dict[str, str]:
    return {m.key: m.value for m in getattr(proto_obj, "metadata_props", [])}


class TestValueTagsErrors(unittest.TestCase):
    def test_requires_cpp_bindings(self):
        with (
            patch.object(_proto_utils, "_shape_inference", None),
            self.assertRaisesRegex(
                RuntimeError, "onnx_light\\.onnx_py\\._onnxpyoptim.*is unavailable"
            ),
        ):
            infer_value_and_node_tags([])


@unittest.skipUnless(HAS_OPTIM_EXT, "requires onnx_light C++ shape_inference bindings")
class TestValueTags(unittest.TestCase):
    def test_tags_graph_and_nodes(self):
        from onnx_light.onnx import TensorProto, helper

        g = helper.make_graph(
            [
                helper.make_node("Shape", ["X"], ["S"]),
                helper.make_node("Reshape", ["X", "S"], ["Y"]),
            ],
            "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        )
        write_value_and_node_tags_to_metadata(g)
        tags = json.loads(_meta_dict(g)[VALUE_TAGS_METADATA_KEY])
        self.assertEqual(tags["S"], "shape")
        self.assertEqual(_meta_dict(g.node[0])[NODE_TAG_METADATA_KEY], "shape")

    def test_tags_subgraph(self):
        from onnx_light.onnx import TensorProto, helper

        body = helper.make_graph(
            [helper.make_node("Shape", ["A"], ["SA"])],
            "body",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2])],
            [helper.make_tensor_value_info("SA", TensorProto.INT64, [2])],
        )
        if_node = helper.make_node("If", ["cond"], ["Y"])
        if_node.attribute.append(helper.make_attribute("then_branch", body))
        g = helper.make_graph(
            [if_node],
            "g",
            [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
        )
        write_value_and_node_tags_to_metadata(g)
        body_tags = json.loads(_meta_dict(body)[VALUE_TAGS_METADATA_KEY])
        self.assertEqual(body_tags["SA"], "shape")

    def test_accepts_list_of_nodes_and_function(self):
        from onnx_light.onnx import helper

        nodes = [
            helper.make_node("Shape", ["X"], ["S"]),
            helper.make_node("Expand", ["Y", "S"], ["Z"]),
        ]
        value_tags, node_tags = infer_value_and_node_tags(nodes)
        self.assertEqual(value_tags["S"], "shape")
        self.assertEqual(node_tags[0], "shape")

        function = helper.make_function(
            "", "f", ["X", "Y"], ["Z"], nodes, [helper.make_opsetid("", 18)]
        )
        write_value_and_node_tags_to_metadata(function)
        self.assertIn(VALUE_TAGS_METADATA_KEY, _meta_dict(function))


if __name__ == "__main__":
    unittest.main()
