graph_graph.h
=============

``ReplayModel`` applies the same ordered edits as ``Replay`` and retains model
metadata, shared function definitions and new opset imports. Existing imports
cannot be changed; identical function definitions are reused and conflicting
definitions with the same domain/name/overload are rejected.

Initializer calls run once before the edit removes their source constants.
They may consume immutable initializers, new literals and preceding call outputs.
Their flattened outputs and literal additions share one resulting initializer
coordinate system. Preparation validates the complete reachable function tree
before execution: inputs must be explicit, operations deterministic and backed
by registered CPU implementations. Control flow, stateful/random operations,
recursive calls (or nesting beyond 64 functions), graph-valued attributes and
function attribute parameters are not supported for preparation.
Ordinary added functions are not restricted to this preparation subset.

``GraphGraph::Optimize`` records folded computations as shared, canonicalized
functions rather than duplicate tensor payloads. Retained initializers keep
their existing ownership during edits and internal metadata analysis; public
graph exports retain value semantics. ``Replay`` also executes recipes but
rejects surviving nodes that require newly added imports/functions; use
``ReplayModel`` (Python ``replay_model``) for those plans.

``Replay`` coalesces consecutive node-only edits in the same graph when they
replace original nodes at their first matched position and preserve all
externally used tensor descriptors. It validates each replacement's inference
using only its input descriptors, then applies the combined edit with one graph
analysis. Live-rank queries translate sequential positions without rebuilding
the node order for every record.

Edits consuming previously added nodes or reading values produced by earlier edits,
changing boundary descriptors, modifying initializers or renames, replacing nested
graphs, or calling model-local functions or context-specific inference callbacks
end the batch and retain ordinary sequential validation. This includes replacing
an existing constant under the same name before introducing a new consumer.
Local functions take precedence even when they share a native operator's name.
This keeps subgraph identities and invalid-plan diagnostics while avoiding
repeated full analyses for independent replacements.

.. doxygenfile:: onnx_core/builder/graph_graph.h
   :project: onnx-light
