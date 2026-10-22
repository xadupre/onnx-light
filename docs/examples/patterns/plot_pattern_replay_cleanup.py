"""
.. _l-example-plot-pattern-replay-cleanup:

Separating cleanup from reversible rewrites
==========================================

Cleanups that redirect value references establish the canonical graph before
capture. Reversible records restore this cleaned graph, not the source graph
containing the removed identities and duplicate initializers.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/pattern_replay_cleanup.png"

from __future__ import annotations

from onnx_light.onnx import TensorProto
import onnx_light.onnx.helper as oh
from onnx_light.onnx_core.optimization import (
    GraphBuilder,
    GraphGraph,
    apply_rewritings,
    standard_patterns,
)
from onnx_light.tools import pretty_onnx

#####################################
# Create and clean up the source model
# ++++++++++++++++++++++++++++++++++++
#
# This graph contains an ``Identity`` node, a redundant ``Cast``, a dead-end
# ``Neg`` node, and two equal initializers used by retained nodes.

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Add", ["x", "weight"], ["summed"]),
            oh.make_node("Identity", ["summed"], ["forwarded"]),
            oh.make_node("Cast", ["forwarded"], ["casted"], to=TensorProto.FLOAT),
            oh.make_node("Add", ["summed", "duplicate_weight"], ["direct"]),
            oh.make_node("Add", ["casted", "direct"], ["y"]),
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
graph = GraphGraph(builder, standard_patterns(["Cast"]))
rewrites = graph.optimize()
canonical = oh.make_model(
    builder.undo_rewritings(rewrites).build_graph(), opset_imports=[oh.make_opsetid("", 18)]
)
assert all(node.op_type != "Identity" for node in canonical.graph.node)
assert [value.name for value in canonical.graph.initializer] == ["weight"]
optimized_graph = builder.build_graph()

assert rewrites.rewrites
assert not {"RemoveIdentityNodes", "RemoveDuplicateNodes", "RemoveDuplicateInitializers"} & {
    rewrite.pattern_name for rewrite in rewrites.rewrites
}

print("Original graph:")
print(pretty_onnx(model))
print("Cleaned canonical graph:")
print(pretty_onnx(canonical))
print("Optimized graph:")
print(pretty_onnx(optimized_graph))

#####################################
# Replay and undo only the captured modifications
# ++++++++++++++++++++++++++++++++++++++++++++++
#
# Forward replay starts from the cleaned canonical model. Inverse replay
# restores the Cast and dead-end node, but not the pre-capture Identity or
# duplicate weight. Undo uses history retained by the builder that optimized
# the graph. That history is never exported in the model or rewriting protos.

for rewrite in rewrites.rewrites:
    print(rewrite)

replayed_graph = apply_rewritings(canonical, rewrites)
assert replayed_graph.SerializeToString() == optimized_graph.SerializeToString()
print("Replayed graph:")
print(pretty_onnx(replayed_graph))

restored = builder.undo_rewritings(rewrites).build_graph()
assert restored.SerializeToString() == canonical.graph.SerializeToString()
assert all(node.op_type != "Identity" for node in restored.node)
assert restored.SerializeToString() != model.graph.SerializeToString()
print("Restored cleaned canonical graph:")
print(pretty_onnx(restored))
