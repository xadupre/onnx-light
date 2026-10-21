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

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from onnx_light import onnx
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import OptimizationMode
from onnx_light.onnx_lib import RewritingProto, parser
from onnx_light.onnx_core.optimization import (
    GraphBuilder,
    GraphGraph,
    TransposeTransposePattern,
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

# %%
# Export a materialized model or bundle its rewritings
# ---------------------------------------------------
#
# The existing ``to_onnx`` method can optimize an independent copy. With
# ``OptimizationMode.kOptimize`` it exports the optimized model without plans.
# With ``OptimizationMode.kRewritings`` it keeps the canonical graph and stores
# the captured plan under ``"optimized"``. The default ``OptimizationMode.kNone``
# preserves ordinary export. The source builder is not optimized in place.
# Composing two constant transposes illustrates why the bundle stores a recipe
# instead of another weight payload.

exporter = GraphBuilder("export_weights", opsets=18)
x = exporter.inp("x", onnx.TensorProto.FLOAT, [3, 2])
weight = exporter.init(numpy.arange(6, dtype=numpy.float32).reshape(2, 3), name="weight")
weight_identity = exporter.op.Transpose(weight, perm=[0, 1])
transposed = exporter.op.Transpose(weight_identity, perm=[1, 0])
y = exporter.op.Add(x, transposed, outputs="y")
exporter.out(y)
exporter.register_pattern(TransposeTransposePattern(priority=-1))
canonical = exporter.to_onnx()
materialized = exporter.to_onnx(optimize=OptimizationMode.kOptimize)
bundled = exporter.to_onnx(optimize=OptimizationMode.kRewritings)
assert bundled.graph.SerializeToString() == canonical.graph.SerializeToString()
assert not materialized.rewritings
assert any(edit.initializer_calls for edit in bundled.rewritings["optimized"].rewrites)
assert any(
    node.op_type == "Transpose"
    for function in bundled.rewritings["optimized"].added_functions
    for node in function.node
)

# %%
# Save, relocate weights, reload, and select a plan
# ------------------------------------------------
#
# Both files use the existing external-data writer. Loading explicitly brings
# external weights into memory so replay can prepare its constant transpose.
# Loading alone leaves the canonical graph unchanged; selecting a named plan
# invokes replay without running any pattern matchers.

feeds = {"x": numpy.ones((3, 2), dtype=numpy.float32)}
expected = ReferenceEvaluator(canonical).run(None, feeds)[0]
with TemporaryDirectory() as directory:
    path = Path(directory)
    for name, exported in (("materialized", materialized), ("bundled", bundled)):
        onnx.save(exported, path / f"{name}.onnx", save_as_external_data=True, size_threshold=0)
    restored = onnx.load(path / "bundled.onnx", load_external_data=True)
    selected = GraphBuilder(restored).apply_rewritings("optimized").to_standard_model()
    plain = onnx.load(path / "materialized.onnx", load_external_data=True)
    for exported in (restored, selected, plain):
        numpy.testing.assert_array_equal(
            ReferenceEvaluator(exported).run(None, feeds)[0], expected
        )
    assert selected.graph.SerializeToString() == materialized.graph.SerializeToString()
    print(f"Canonical nodes: {len(restored.graph.node)}")
    print(f"Materialized nodes: {len(plain.graph.node)}")
    print("Both exports and replay produce identical outputs.")
