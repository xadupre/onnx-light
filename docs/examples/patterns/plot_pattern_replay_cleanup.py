"""
.. _l-example-plot-pattern-replay-cleanup:

Replaying graph cleanup modifications
=====================================

Graph cleanup algorithms also produce
:class:`~onnx_light.onnx_core.optimization.LocalRewriting` records. These
records replay identity removal, dead-end removal, and initializer
deduplication without running cleanup again.
"""

from __future__ import annotations

from onnx_light.onnx import TensorProto
import onnx_light.onnx.helper as oh
from onnx_light.onnx_core.optimization import GraphBuilder, GraphGraph, replay
from onnx_light.tools import pretty_onnx

#####################################
# Create and clean up the source model
# ++++++++++++++++++++++++++++++++++++
#
# This graph contains an ``Identity`` node, a dead-end ``Neg`` node, and two
# equal initializers used by retained nodes.

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Add", ["x", "weight"], ["summed"]),
            oh.make_node("Identity", ["summed"], ["forwarded"]),
            oh.make_node("Add", ["forwarded", "duplicate_weight"], ["y"]),
            oh.make_node("Neg", ["x"], ["dead_end"]),
        ],
        "cleanup",
        [oh.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=[
            oh.make_tensor("weight", TensorProto.FLOAT, [1], [1.0]),
            oh.make_tensor("duplicate_weight", TensorProto.FLOAT, [1], [1.0]),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
)

builder = GraphBuilder(model)
graph = GraphGraph(builder, use_global_patterns=False)
rewrites = list(graph.optimize())
optimized_graph = builder.build_graph()

assert {"RemoveIdentityNodes", "RemoveUnusedNodes", "RemoveDuplicateInitializers"} <= {
    rewrite.pattern_name for rewrite in rewrites
}

print("Original graph:")
print(pretty_onnx(model))
print("Optimized graph:")
print(pretty_onnx(builder.to_onnx("model")))

#####################################
# Inspect and replay the cleanup modifications
# ++++++++++++++++++++++++++++++++++++++++++++
#
# Every cleanup operation is captured as a ``LocalRewriting`` record. Replay
# applies the records to a fresh copy of the source model.

for rewrite in rewrites:
    print(rewrite)

replayed_graph = replay(model, rewrites)
assert replayed_graph.SerializeToString() == optimized_graph.SerializeToString()
print("Replayed graph:")
print(pretty_onnx(replayed_graph))
