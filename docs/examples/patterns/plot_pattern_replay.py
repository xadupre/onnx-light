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
# Each record has a concise one-line display. Use ``to_detailed_string`` to
# inspect the pattern, matched nodes, inserted nodes, and optimization iteration
# needed to reproduce one modification.

for rewrite in rewrites:
    print(rewrite)
    print(rewrite.to_detailed_string())

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
