"""
.. _l-example-plot-node-callback:

Inspect and edit nodes while parsing or serializing with a node callback
========================================================================

This example shows how to use :attr:`onnx_light.onnx.ParseOptions.node_callback`
and :attr:`onnx_light.onnx.SerializeOptions.node_callback` to hook into every
:class:`onnx_light.onnx.NodeProto` of a model.

The callback receives each node by reference along with its parent
:class:`onnx_light.onnx.GraphProto` and may inspect or modify the node in place.
The parent graph lets the callback locate the node's surrounding graph —
including the subgraphs nested inside control flow operators such as ``If``,
``Loop`` and ``Scan``.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/node_callback.png"

import numpy as np

import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
import onnx_light.onnx as onnxl

# %%
# Build a small model with a subgraph
# -----------------------------------
#
# The main graph holds an ``Add`` node and an ``If`` node.  The ``If`` node
# carries a ``then_branch`` subgraph with its own ``Identity`` node, so the
# callback visits nodes across two graphs.

arr = np.array([1.0, 2.0], dtype=np.float32)
add = oh.make_node("Add", ["X", "W"], ["Y"], name="add0")
sub_node = oh.make_node("Identity", ["cond"], ["Z"], name="id0")
then_graph = oh.make_graph([sub_node], "then_graph", [], [])
if_node = oh.make_node("If", ["cond"], ["Z"], name="if0", then_branch=then_graph)
graph = oh.make_graph([add, if_node], "main", [], [], initializer=[onh.from_array(arr, name="W")])
onnx_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)

serialized = onnx_model.SerializeToString()

# %%
# Inspect every node while parsing
# --------------------------------
#
# ``node_callback`` fires once per node.  The parent graph is passed as the
# second argument, so we can record which graph each node belongs to.  The
# subgraph node is visited while the parser reads the enclosing ``If`` node,
# before the rest of the main graph.

parse_options = onnxl.ParseOptions()
visited = []


def on_node(node: onnxl.NodeProto, graph: onnxl.GraphProto):
    """Records the node op_type and the name of its parent graph."""
    visited.append((node.op_type, graph.name))


parse_options.node_callback = on_node

parsed_model = onnxl.ModelProto()
parsed_model.ParseFromString(serialized, parse_options)

for op_type, graph_name in visited:
    print(f"parsed node {op_type!r} in graph {graph_name!r}")

# %%
# Edit nodes in place while parsing
# ---------------------------------
#
# Because the callback receives each node by reference, it can rewrite the node.
# Here we stamp a ``doc_string`` on every node as it is parsed.

edit_options = onnxl.ParseOptions()
edit_options.node_callback = lambda node, graph: setattr(node, "doc_string", "parsed")

edited = onnxl.ModelProto()
edited.ParseFromString(serialized, edit_options)
print(f"add0 doc_string: {edited.graph.node[0].doc_string!r}")
print(f"if0 doc_string:  {edited.graph.node[1].doc_string!r}")

# %%
# Edit nodes while serializing
# ----------------------------
#
# ``SerializeOptions.node_callback`` works the same way. The callback edits the
# nodes in place while the serialized bytes are produced, then onnx-light restores
# the original state, so edits never alter the model held by the caller. The
# stamped ``doc_string`` therefore appears only in the serialized bytes.

serialize_options = onnxl.SerializeOptions()
serialize_options.node_callback = lambda node, graph: setattr(node, "doc_string", "serialized")

stamped_bytes = onnx_model.SerializeToString(serialize_options)

reparsed = onnxl.ModelProto()
reparsed.ParseFromString(stamped_bytes)
print(f"serialized add0 doc_string:      {reparsed.graph.node[0].doc_string!r}")
sub = reparsed.graph.node[1].attribute[0].g
print(f"serialized subgraph doc_string:  {sub.node[0].doc_string!r}")

# %%
# The caller's model is untouched: onnx-light restored every node the callback
# edited once the serialized bytes were produced.

print(f"original add0 doc_string still empty: {onnx_model.graph.node[0].doc_string!r}")
