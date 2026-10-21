.. _l-next-steps-serialized-graph-rewriting:

Serialized graph rewrites and model variants
============================================

:Date: 2026-10-20

**Steps 1 and 2 implemented: proto records and extended replay.** The optimizer
captures initializer recipes and shared function definitions in
``LocalRewritingProto`` / ``RewritingProto``. ``GraphBuilder::ApplyRewritings``
(Python: ``builder.apply_rewritings``) returns a new builder, prepares constants
once and preserves model-level additions.
The model envelope and export operations below remain proposals.

Existing building blocks
++++++++++++++++++++++++

``PatternOptimization`` implements recognition (``Match``) and replacement
construction (``Apply``), in C++ or Python. ``MatchResult`` identifies one
match. ``GraphGraph::Optimize`` returns a ``RewritingProto`` containing ordered
``LocalRewritingProto`` edits:
matched node positions, added nodes and their positions, added/removed
initializers, value renames and subgraph path.

``builder.ApplyRewritings(rewrites)`` reconstructs a ``GraphBuilder`` without
running the matcher. Matched and removed positions refer to the graph
immediately before each edit; added positions describe its result. Internal
optimizer batches are normalized to this sequential coordinate system. Later
edits may consume nodes created by earlier ones.
Normalization keeps an indexed live order and materializes the final sequence
once. Retained nodes displaced across other retained nodes are explicitly
recorded in the edit, so replay preserves subsequent positional coordinates.
These records own snapshots of displaced nodes, independent of later cleanup
and mutations of the live builder.
Every rewrite immediately maintains dependency-consistent node order, shapes,
and compute annotations. Coordinates refer to that actual order, so the returned
builder can receive another plan, including an empty plan, without exporting first.
New producers are inserted before existing consumers whenever their own inputs
permit it. Otherwise, only blocked consumers and their affected consumers are
displaced. Independent retained nodes keep their relative order. Mutations do
not topologically sort the graph, and export only uses that fallback if a
dependency-order violation is actually detected.

The two rewrite messages support binary serialization and Python bindings.
Records contain a diagnostic pattern name, not a live pattern pointer or
timings. Timing remains in ``OptimizationReport``. There is not yet a rewrite
map in ``ModelProto``.

Constant folding retains the computation as shared functions and ordered
``initializer_calls``, not materialized results in ``added_initializers``.
Literal constants still use ``added_initializers``. Cleanup records only
removed initializer positions and renames, without re-embedding retained
weights. Serializing a transposed-weight recipe therefore stores no second
transposed payload.

The messages are defined in ``onnx_proto/include/onnx_proto/onnx.h`` using
the existing protobuf-free message layer. The schema below includes fields
used by constant preparation. The returned builder exposes ``BuildGraph()``
for a graph-only result and ``ToModel()`` to export the graph together with its
functions and opset imports.

Decision: use protos as runtime records
++++++++++++++++++++++++++++++++++++++++

**Use ``LocalRewritingProto`` as the record used
by the optimizer and existing replay engine.** ``RewritingProto`` holds the
ordered records and their shared function definitions/imports. Do not maintain
parallel C++/proto records or introduce a second replay engine and conversion
layer.

Migrate ``Optimize`` results, replay arguments, cleanup recording, C++/Python
bindings, diagnostics, tests and examples to the new messages. Keep matching
and ``MatchResult`` separate: the live pattern pointer belongs to matching,
not the recorded edit. Applying rewrites must not require that pointer while retaining
its structural validation. Keep timing in ``OptimizationReport`` rather than
requiring it in replay records.

.. code-block:: text

   Optimize + capture recipes -> RewritingProto -> GraphBuilder::ApplyRewritings
                                       |
                              file / ModelProto.rewritings

The proto sequence applies one rewrite at a time: matched/removed
positions refer to the graph immediately before that rewrite, and added
positions to its result. There is no serialized ``iteration`` or batch ID.
If the optimizer still applies internal batches, normalize their positions
when producing the ordered proto records: rebase node/initializer positions
and affected subgraph paths after each preceding edit. Internal batching may
remain an implementation detail, not a second persistent record format.
Verify that sequential replay preserves the batch result. Check integer
ranges before converting proto positions to ``size_t``.
Normalization indexes insertion anchors and maintains live node and initializer
ranks incrementally; it does not rescan or sort the full live order per edit.

New and modified proto messages
+++++++++++++++++++++++++++++++

The wire schema is below. The first two messages use onnx-light's existing
``BEGIN_PROTO`` / ``FIELD_*`` layer, parser and serializer, without adding a
protobuf compiler or ``libprotobuf``. The ``ModelProto.rewritings`` field
remains deferred to step 3.

.. code-block:: protobuf

   // New messages; NodeProto/TensorProto and the other ONNX messages exist.
   message LocalRewritingProto {
     string pattern_name = 1;
     repeated string graph_path = 2;
     repeated uint64 matched_nodes = 3;
     repeated uint64 removed_initializers = 4;
     repeated NodeProto added_nodes = 5;
     repeated uint64 added_nodes_positions = 6;
     repeated TensorProto added_initializers = 7;
     repeated uint64 added_initializer_positions = 8;
     repeated StringStringEntryProto value_renames = 9;
     repeated StringStringEntryProto metadata_props = 10;
     repeated NodeProto initializer_calls = 11;
     repeated uint64 computed_initializer_positions = 12;
     repeated uint64 removed_extended_initializers = 13;
   }

   message RewritingProto {
     repeated LocalRewritingProto rewrites = 1;
     repeated OperatorSetIdProto required_opset_imports = 2;
     repeated StringStringEntryProto metadata_props = 3;
     repeated FunctionProto added_functions = 4;
   }

Removal positions are relative to the corresponding initializer list before
each edit: ``removed_initializers`` indexes dense initializers,
``removed_extended_initializers`` indexes the unified
``GraphProto.extended_initializer`` list of encoded constants and paged caches.
Both payload alternatives share this index space and retain their declaration
order across builder import/export and replay.

**Modify only onnx-light's ``ModelProto``** to add
``map<string, RewritingProto> rewritings``, keyed by variant name, without
introducing another named proto message in this plan. Reserve and document
its extension field number before implementation: field **1001** is reserved
for this map, following ``struct_types`` at 1000 within ModelProto's local
1000-1099 extension range. Reject duplicate keys. Each map entry uses the
protobuf map encoding (string key at field 1, RewritingProto value at field 2);
no additional public proto message is introduced.

``NodeProto``, ``GraphProto``, ``TensorProto``, ``FunctionProto`` and
``OperatorSetIdProto`` need no new fields. Their existing forms carry the
replacement graph data. Extend model binary I/O and Python bindings to expose
and preserve the new field. This is an onnx-light extension, not a standard
ONNX field or an upstream ONNX schema change.

The local proto replaces the live pointer with its diagnostic pattern name.
Producer identity uses the existing ``ModelProto.producer_name`` and
``producer_version`` fields. Do not serialize timings in V1:
they are not required for replay. Both ``LocalRewritingProto.metadata_props``
and ``RewritingProto.metadata_props`` carry
optional descriptive key/value metadata, not executable compatibility rules.
Runtime schema and kernel checks must not depend on these annotations.
Imports are collection-level additions checked against the canonical model,
not permission to reinterpret surviving operators.

Deferred constant folding and added functions
+++++++++++++++++++++++++++++++++++++++++++++

Optimizer capture populates ``LocalRewritingProto.initializer_calls``
and ``computed_initializer_positions`` directly. Collect and deduplicate
definitions across all local edits into ``RewritingProto.added_functions``
during capture. Local records contain calls, not repeated definitions.
Capture the folded operations and their dependencies before
``ConstantFoldNodes`` discards them, and package the computation as a
``FunctionProto`` with explicit inputs and outputs. Calls use ``NodeProto``:

.. code-block:: text

   RewritingProto.added_functions:
     PrepareWeights(W) -> WT { WT = Transpose(W) }
   LocalRewritingProto.initializer_calls:
     WT = PrepareWeights(W)
   LocalRewritingProto.computed_initializer_positions:
     [position of WT in the resulting initializers]

Call inputs reference existing initializers or preceding call outputs. Calls
are ordered, evaluated once during variant preparation, and their outputs
become initializers, not per-inference graph nodes. Flatten call outputs in
call/output order to associate them with ``computed_initializer_positions``.
These positions and ``added_initializer_positions`` share the resulting
initializer coordinate system and must not overlap.

Read source constants and compute call results before deleting any referenced
initializers. Then apply removals and install literal/computed initializers
and replacement nodes. Keep ``added_initializers`` for literal constants;
do not serialize a computed payload as well as its recipe. Rebasing to
sequential records must preserve preparation dependencies and positions too.

Before applying local edits, ``ApplyRewritings`` merges collection-level functions:
add absent definitions, reuse identical definitions,
and reject conflicting definitions with the same identity, including conflicts
between local edits during capture. Return a new ``GraphBuilder`` containing
the merged functions/imports, and use its existing ``BuildGraph`` / ``ToModel``
exports rather than separate graph/model replay functions. Update dependent
callers explicitly. Validate constant-only, deterministic preparation
operations and available implementations; do not execute arbitrary stateful
functions as initializer recipes.

This avoids storing a second transposed weight payload in the file, not
necessarily allocating a second buffer during preparation. Reuse a view only
when supported by the tensor representation and consumer. Materialized plain
ONNX export may intentionally store computed weights instead.

``LocalRewritingProto.value_renames`` reuses ``StringStringEntryProto``:
``key`` is the old reference and ``value`` its replacement. It mirrors the
former ``LocalRewriting.value_renames`` and redirects references to an
eliminated value, rather than renaming a
surviving producer. For example, removing ``b = Identity(a)`` redirects
consumers of ``b`` to ``a``. Existing identity/duplicate-node cleanup and
initializer deduplication record these substitutions, including nested
captures. Ordinary replacements can create fresh internal names and preserve
boundary outputs without this field; retain it to replay cleanup too.

Every variant starts from the canonical model containing it, not another
variant. Standalone distribution of rewrite plans is outside this proposal.
Any modification to the source model must clear or regenerate its variants.
Without a source fingerprint, structural validation alone cannot detect every
stale plan; maintaining this association is the responsibility of model
mutation and export APIs.

This serializes matched edits, not a general declarative matching language.

PR sequence
+++++++++++

Each PR depends on the preceding one.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - PR
     - Scope
     - Acceptance
   * - 1. Proto records (implemented)
     - Add ``LocalRewritingProto`` and ``RewritingProto`` and replace the
       existing C++ record. Migrate optimizer, cleanup and replay to ordered
       proto edits, with position rebasing; update bindings and direct callers.
     - Binary round-trip; no matcher pointer or duplicate record type;
       sequential replay preserves existing results, including batch edits;
       invalid positions rejected.
   * - 2. Extend existing replay (implemented)
     - Capture initializer recipes and shared function definitions directly
       in the protos. Extend existing replay for one-time constant preparation
       and model-level function/import additions; preserve payload ownership.
     - Serialized and in-memory replay agree; cover nested graphs, cleanup,
       initializers, renames, multi-batch edits and invalid plans. A transposed-weight
       recipe stores no duplicate payload; verify results, source removal
       ordering and function conflicts. Retained initializer payloads are reused,
       not deep-copied for every edit, including node-only edits.
   * - 3. Model envelope (implemented)
     - Add the ``ModelProto.rewritings`` map, model I/O and bindings;
       define explicit selection and compatibility.
     - Two independent variants survive save/load; the canonical graph stays
       unchanged; mutations invalidate variants; duplicate keys and missing
       kernels are diagnosed.
   * - 4. Export workflow
     - Add explicit canonical, bundled and materialized export modes, with a
       small end-to-end example and external-data handling.
     - Save/reload/replay matches direct optimization; compare inference
       outputs and states; plain exports use only standard ONNX fields.

Export workflow
+++++++++++++++

The model-envelope API reserves field 1001 for ``ModelProto.rewritings``.
``ModelProto.add_rewriting`` (C++ ``add_rewritings``) and
``GraphBuilder.add_rewriting`` (C++ ``AddRewriting``) reject duplicate names.
``GraphBuilder.apply_rewritings(name)`` (C++ ``ApplyRewritings``) selects exactly
one independent plan using existing replay and checks reachable operator
implementations, including local functions and nested graphs. The replay result
has no attached variants. Loading a model never executes its plans.
The method is overloaded to accept either a ``RewritingProto`` or a plan name.
GraphBuilder mutations invalidate the map, including mutations in nested
builders and finalization that changes node order. Finalize before attaching
plans. Callers editing proto fields directly must explicitly clear or
regenerate the map; no source fingerprint or hidden repair is performed.
ORT and standard-model export reject attached variants rather than dropping
them. The export modes and packaging below remain step 4.

1. Finalize the canonical model first, including cleanup, ordering, names,
   imports and external-data layout.
2. Optimize an independent copy for each selected target, recording all edits.
   Capture constant computations as functions/calls instead of embedding
   computed weight payloads. The optimizer returns ``RewritingProto`` directly;
   optionally attach descriptive metadata. Verify replay against direct
   optimization before publishing the variant.
3. Export either the untouched canonical model, a bundled onnx-light model
   containing the canonical graph and rewrite map, or a standard ONNX model
   with one selected variant materialized and the extension map removed.

Bundling is opt-in. Do not replace the canonical graph while retaining
positions that refer to its previous contents. Renaming, reordering or
transforming the source invalidates its variants; regenerate them rather than
silently reusing stale positions. When saving elsewhere, relocate or copy
external payloads referenced by both canonical and replacement tensors.
Do not assume other ONNX tools preserve unknown extension fields.

At load time, explicit variant selection validates positions, graph structure,
schemas, kernels and resource requirements before replay on a working model.
A requested incompatible variant fails with a clear error; canonical
execution remains an explicit alternative, not a silent fallback.

Comparison with ORT runtime fusion
++++++++++++++++++++++++++++++++++

`ORT issue #33403 <https://github.com/microsoft/onnxruntime/issues/33403>`_
proposes optional runtime fusion packages that match the graph after EP
assignment and submit replacements using existing kernels. This proposal
instead records concrete edits during export and bundles them with the
canonical model for later apply_rewritings, without rerunning the matcher. Updating
export-time matching decisions requires regenerating the affected plans;
ORT's proposed packages can update matching and adapter logic independently
of the canonical artifact. Runtime matching can account for actual EP
ownership and session kernel availability, whereas serialized variants must
be selected and validated against the target runtime. Both approaches keep a
canonical graph, but differ in when rewrites are discovered and how they are
distributed.

Rewrite definitions can live outside the canonical graph: a Python exporter
can construct and serialize ``RewritingProto`` records as data without
providing a separately compiled runtime extension. Requiring exporters to
ship a native fusion package would complicate Python-based export workflows;
keep that requirement out of this design. ORT's proposed runtime packages
instead carry executable matching and adapter logic behind an experimental
extension interface. Serialized edits still require compatible operator
implementations in the target runtime, but the exporter need not supply a
binary merely to describe those edits. The bundled model remains the proposed
distribution format; this does not introduce standalone rewrite files.
