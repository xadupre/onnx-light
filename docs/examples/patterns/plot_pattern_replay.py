"""
.. _l-example-plot-pattern-replay:

Replaying graph-rewriting patterns
==================================

Every successful pattern optimization produces a
:class:`~onnx_light.onnx.LocalRewritingProto` record. These
records can reconstruct the optimized graph from the original model without
running pattern matching again.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/pattern_replay.png"

from __future__ import annotations

from onnx_light.onnx_lib import RewritingProto, parser
from onnx_light.onnx_core.optimization import (
    GraphBuilder,
    GraphGraph,
    apply_rewritings,
    standard_patterns,
)
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
print(pretty_onnx(builder.to_onnx()))

#####################################
# Inspect the captured modifications
# ++++++++++++++++++++++++++++++++++
#
# The collection stores sequential edits without pattern pointers or timings.
# Each record's positions refer to the graph immediately before/after that edit.

for rewrite in rewrites.rewrites:
    print(rewrite)

#####################################
# ApplyRewriting without matching patterns
# ++++++++++++++++++++++++++++++++
#
# :func:`~onnx_light.onnx_core.optimization.apply_rewritings` applies the captured
# records to a fresh copy of the source model. The reconstructed graph is
# byte-for-byte identical to the graph produced by the optimizer.

serialized = rewrites.SerializeToString()
loaded = RewritingProto()
loaded.ParseFromString(serialized)
replayed_graph = apply_rewritings(model, loaded)

assert replayed_graph.SerializeToString() == optimized_graph.SerializeToString()
print("Replayed graph:")
print(pretty_onnx(replayed_graph))
