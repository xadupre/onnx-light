"""
.. _l-example-plot-pattern-replay:

Replaying graph-rewriting patterns
==================================

Every successful pattern optimization produces a
:class:`~onnx_light.onnx_core.optimization.LocalRewriting` record. These
records can reconstruct the optimized graph from the original model without
running pattern matching again.
"""

from __future__ import annotations

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx_lib import parser
from onnx_light.onnx_core.optimization import GraphBuilder, GraphGraph, replay, standard_patterns
from onnx_light.tools import pretty_onnx

#####################################
# Create and optimize the source model
# ++++++++++++++++++++++++++++++++++++
#
# The source graph contains two type-preserving ``Cast`` nodes. The optimizer
# rewrites them and returns the corresponding modification records.

model = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 18]>\n'
    "agraph (float[4] x) => (float[4] y) {\n"
    "  casted = Cast <to=1> (x)\n"
    "  negated = Neg(casted)\n"
    "  y = Cast <to=1> (negated)\n"
    "}\n"
)

builder = GraphBuilder(model)
graph = GraphGraph(builder, standard_patterns(["Cast"]))
rewrites = graph.optimize()
optimized_graph = builder.build_graph()

print("Original graph:")
print(pretty_onnx(model))
print("Optimized graph:")
print(pretty_onnx(builder.to_onnx("model")))

#####################################
# Inspect the captured modifications
# ++++++++++++++++++++++++++++++++++
#
# Each record identifies the pattern, matched nodes, inserted nodes, and
# optimization iteration needed to reproduce one modification.

for rewrite in rewrites:
    print(rewrite)

#####################################
# Replay without matching patterns
# ++++++++++++++++++++++++++++++++
#
# :func:`~onnx_light.onnx_core.optimization.replay` applies the captured
# records to a fresh copy of the source model. The reconstructed graph is
# byte-for-byte identical to the graph produced by the optimizer.

replayed_graph = replay(model, rewrites)

assert replayed_graph.SerializeToString() == optimized_graph.SerializeToString()
print("Replayed graph:")
print(pretty_onnx(replayed_graph))

#####################################
# Replay cleanup modifications
# ++++++++++++++++++++++++++++
#
# Cleanup algorithms also create ``LocalRewriting`` records. This graph has an
# ``Identity`` node, a dead-end ``Neg`` node, and two equal initializers. It
# therefore demonstrates identity removal, dead-end removal, and initializer
# deduplication without applying any graph-rewriting patterns.

cleanup_model = helper.make_model(
    helper.make_graph(
        [
            helper.make_node("Add", ["x", "weight"], ["summed"]),
            helper.make_node("Add", ["summed", "duplicate_weight"], ["total"]),
            helper.make_node("Identity", ["total"], ["y"]),
            helper.make_node("Neg", ["x"], ["dead_end"]),
        ],
        "cleanup",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=[
            helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0]),
            helper.make_tensor("duplicate_weight", TensorProto.FLOAT, [1], [1.0]),
        ],
    ),
    opset_imports=[helper.make_opsetid("", 18)],
)

cleanup_builder = GraphBuilder(cleanup_model)
cleanup_rewriter = GraphGraph(cleanup_builder, use_global_patterns=False)
cleanup_rewrites = list(cleanup_rewriter.optimize())
cleanup_graph = cleanup_builder.build_graph()

assert {"RemoveIdentityNodes", "RemoveUnusedNodes", "RemoveDuplicateInitializers"} <= {
    rewrite.pattern_name for rewrite in cleanup_rewrites
}
for rewrite in cleanup_rewrites:
    print(rewrite)

replayed_cleanup_graph = replay(cleanup_model, cleanup_rewrites)
assert replayed_cleanup_graph.SerializeToString() == cleanup_graph.SerializeToString()
print("Replayed cleanup graph:")
print(pretty_onnx(replayed_cleanup_graph))
