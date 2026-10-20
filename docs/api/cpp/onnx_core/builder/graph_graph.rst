graph_graph.h
=============

``GraphGraph::Optimize`` captures ordered edits. Applying them belongs to
``GraphBuilder::ApplyRewritings``, which returns a new builder and retains model
metadata, shared function definitions and new opset imports. Existing imports
cannot be changed; identical function definitions are reused and conflicting
definitions with the same domain/name/overload are rejected.
Imported functions use a namespace independent of graph values and subgraphs;
different domains and overloads may share a function name. Shape inference
preserves those identities, including the overload on imported call nodes.
Identity components escape percent signs and colons so a colon in a function
name cannot be confused with an overload separator.
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
graph exports retain value semantics. ``ApplyRewritings`` executes recipes once.
Use the returned builder's ``ToModel()`` to export functions and imports;
``BuildGraph()`` exports only its graph.
Native constant folding skips operations shadowed by model-local functions,
so it cannot disagree with function-first dispatch during replay.
Serialized folding also excludes operations outside the preparation subset,
including inference-mode ``Dropout``, which remain ordinary added nodes.
Optimization rejects a completed plan if a function introduced later would
override a native operation already captured in an initializer recipe.

``ApplyRewritings`` applies serialized edits directly to a new ``GraphBuilder``,
preserving the source builder. Added functions must be compatible with existing
calls before any edit is applied: an incompatible signature or a conflicting
function name raises an error rather than postponing validation or renaming the
function. Identical definitions are reused after normalizing builder-generated
annotations and imports. Each edit updates affected shapes and annotations in
dependency order; the returned builder preserves recorded node positions and has
up-to-date memory metadata.

Consecutive node-only edits in the same graph are coalesced when they replace
original nodes at their first matched position. Live-rank queries translate
sequential positions without rebuilding the node order for every record.
Edits consuming previously added nodes, modifying initializers or renames, or
replacing nested graphs end the batch. Coordinates, initializer conflicts and
preparation eligibility are checked when applying each edit. Subgraph identities
remain stable across edits even when their referencing nodes move.

.. doxygenfile:: onnx_core/builder/graph_graph.h
   :project: onnx-light
