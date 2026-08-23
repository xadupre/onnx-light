"""
.. _l-example-plot-pattern-optimization:

Optimizing a model with graph-rewriting patterns
=================================================

*onnx-light* optimizes a model by repeatedly matching small
:class:`~onnx_light.onnx_core.optimization.PatternOptimization` subgraphs and
replacing them with a simplified equivalent. This example walks through the
whole workflow on a tiny model:

1. Build a model containing redundant nodes (two useless ``Cast`` and two
   consecutive ``Neg``).
2. Run the standard patterns and inspect the optimization *statistics*
   (:class:`~onnx_light.onnx_core.optimization.OptimizationReport`).
3. List every applied modification
   (:class:`~onnx_light.onnx_core.optimization.LocalRewriting`) and *replay*
   them from the original model to reproduce the optimized graph.
4. Write a custom pattern.
5. Inspect a candidate rejected by that pattern and the recorded failure reason.
6. Apply the custom pattern together with the standard ones.

See :ref:`l-howto-add-custom-pattern` for a companion how-to that focuses on
writing and registering a pattern (including how priorities order patterns),
in both Python and C++.
"""

from __future__ import annotations

import onnx_light.onnx.helper as oh
from onnx_light.onnx_lib import parser
from onnx_light.onnx_core.optimization import (
    GraphBuilder,
    GraphGraph,
    PatternOptimization,
    replay,
    standard_patterns,
)
from onnx_light.tools import pretty_onnx

#####################################
# Build a model with redundant nodes
# ++++++++++++++++++++++++++++++++++
#
# ``x`` is cast to ``float`` (a no-op, since it is already ``float``) and then
# negated twice before being cast back. The standard ``Cast`` pattern removes
# the two ``Cast`` nodes; the custom pattern added further below removes the
# two ``Neg`` nodes.

model = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 18]>\n'
    "agraph (float[4] x) => (float[4] y) {\n"
    "  casted = Cast <to=1> (x)\n"
    "  middle = Neg(casted)\n"
    "  negated = Neg(middle)\n"
    "  y = Cast <to=1> (negated)\n"
    "}\n"
)
print(pretty_onnx(model))

#####################################
# Run the standard patterns and read the statistics
# +++++++++++++++++++++++++++++++++++++++++++++++++
#
# :meth:`~onnx_light.onnx_core.optimization.GraphGraph.optimize` returns the
# list of applied rewrites; passing ``report=True`` additionally returns an
# :class:`~onnx_light.onnx_core.optimization.OptimizationReport` with timing
# and match/no-match counters for every pattern that was tried.

builder = GraphBuilder(model)
graph = GraphGraph(builder, standard_patterns(["Cast"]))
rewrites, report = graph.optimize(report=True)
optimized_graph = builder.build_graph()

print(pretty_onnx(builder.to_onnx("model")))
print(report)

for pattern_stats in report.patterns:
    print(
        f"{pattern_stats.pattern_name}: {pattern_stats.matches} match(es) over "
        f"{pattern_stats.attempts} attempt(s)"
    )

#####################################
# List the modifications and replay them
# ++++++++++++++++++++++++++++++++++++++
#
# Each :class:`~onnx_light.onnx_core.optimization.LocalRewriting` records which
# pattern fired, the positions of the matched nodes, and the nodes it added.
# :func:`~onnx_light.onnx_core.optimization.replay` reconstructs the optimized
# graph by reapplying that captured sequence to a **fresh copy** of the
# original model, without running the pattern matcher again.

for rewrite in rewrites:
    print(rewrite)

replayed_graph = replay(model, rewrites)
assert replayed_graph.SerializeToString() == optimized_graph.SerializeToString()
print("replay reproduced the optimized graph")

#####################################
# Add a custom pattern
# ++++++++++++++++++++
#
# A pattern derives from
# :class:`~onnx_light.onnx_core.optimization.PatternOptimization` and
# implements ``fast_op_type`` (the operator types it may start from), ``match``
# (returns ``self.result(...)`` on success or ``self.no_match(...)`` with a
# diagnostic otherwise) and ``apply`` (builds the replacement nodes). Passing
# it to ``GraphGraph`` alongside the standard patterns runs both together, in
# ascending ``priority`` order.


class NegNegPattern(PatternOptimization):
    """Replaces two consecutive Neg nodes with Identity."""

    def __init__(self):
        super().__init__(priority=1, name="NegNeg")

    def fast_op_type(self):
        return {"Neg"}

    def match(self, graph, node):
        previous = graph.node_before(node.input[0])
        if previous is None or previous.op_type != "Neg":
            return self.no_match(node, "the input is not produced by Neg")
        return self.result([previous, node], insert_at=node)

    def apply(self, graph, nodes):
        del graph
        previous, node = nodes
        return [oh.make_node("Identity", [previous.input[0]], list(node.output))]


#####################################
# Inspect a failed pattern candidate
# ++++++++++++++++++++++++++++++++++
#
# The same custom pattern rejects a single ``Neg`` because its input is not
# produced by another ``Neg``. Returning ``self.no_match(candidate, reason)``
# keeps that rejection out of normal output, but the optional report stores the
# aggregated reason so the pattern author can understand why nothing changed.

failure_model = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 18]>\n'
    "agraph (float[4] x) => (float[4] y) {\n"
    "  y = Neg(x)\n"
    "}\n"
)
builder = GraphBuilder(failure_model)
graph = GraphGraph(builder, [NegNegPattern()])
failed_rewrites, failed_report = graph.optimize(report=True)

print(f"failed rewrite count: {len(failed_rewrites)}")
for pattern_stats in failed_report.patterns:
    for no_match in pattern_stats.no_matches:
        print(
            f"{pattern_stats.pattern_name} rejected "
            f"{no_match.occurrences} candidate(s): {no_match.reason}"
        )

assert not failed_rewrites
failed_no_match_reasons = {
    no_match.reason
    for pattern_stats in failed_report.patterns
    if pattern_stats.pattern_name == "NegNeg"
    for no_match in pattern_stats.no_matches
}
assert "the input is not produced by Neg" in failed_no_match_reasons

#####################################
# Apply the custom pattern with the standard patterns
# +++++++++++++++++++++++++++++++++++++++++++++++++++

builder = GraphBuilder(model)
graph = GraphGraph(builder, [*standard_patterns(["Cast"]), NegNegPattern()])
rewrites = graph.optimize()
fully_optimized = builder.to_onnx("model")

print(pretty_onnx(fully_optimized))
print([rewrite.pattern_name for rewrite in rewrites])
assert [node.op_type for node in fully_optimized.graph.node] == ["Identity"]
