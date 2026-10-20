graph_graph.h
=============

``ReplayModel`` applies the same ordered edits as ``Replay`` and retains model
metadata, shared function definitions and new opset imports. Existing imports
cannot be changed; identical function definitions are reused and conflicting
definitions with the same domain/name/overload are rejected.
Imported functions use a namespace independent of graph values and subgraphs;
different domains and overloads may share a function name. Shape inference
preserves those identities, including the overload on imported call nodes.
Model-level imports include additions in nested graph scopes, while each
function keeps its own imports.

Initializer calls run once before the edit removes their source constants.
They may consume immutable initializers, new literals and preceding call outputs.
Public graph inputs remain overridable and cannot be consumed as constants,
including defaults introduced by literals or preceding calls in the same edit.
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
This check includes reachable function bodies and their nested graphs, without
confusing function-local imports with model-level imports.
Native constant folding skips operations shadowed by model-local functions,
so it cannot disagree with function-first dispatch during replay.
Serialized folding also excludes operations outside the preparation subset,
including inference-mode ``Dropout``, which remain ordinary added nodes.
Optimization rejects a completed plan if a function introduced later would
override a native operation already captured in an initializer recipe.

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
