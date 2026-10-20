.. _l-next-steps-serialized-graph-rewriting:

Serialized graph rewrites and model variants
============================================

:Date: 2026-10-20

**Discussion -- not implemented.** Proto names and export operations below
are proposals, not available APIs.

Existing building blocks
++++++++++++++++++++++++

``PatternOptimization`` implements recognition (``Match``) and replacement
construction (``Apply``), in C++ or Python. ``MatchResult`` identifies one
match. ``GraphGraph::Optimize`` records concrete changes as ``LocalRewriting``:
matched node positions, added nodes and their positions, added/removed
initializers, value renames, subgraph path and iteration.

``Replay(model, rewrites)`` already reconstructs a ``GraphProto`` without
running the matcher. Positions refer to the start of each iteration batch;
added positions describe the result of that batch. One batch targets one
graph. Later batches may consume nodes created by earlier ones.

There is no dedicated ``LocalRewritingProto``, serialized rewrite collection,
or rewrite map in ``ModelProto``. The current record contains an in-memory
pattern pointer; its ``NodeProto`` and ``TensorProto`` members alone do not
make the complete record serializable.

Constant folding currently stores materialized results: after
``ConstantFoldNodes``, the optimizer copies computed tensors into
``added_initializers`` and removes folded operations from ``added_nodes``.
It does not retain the computation that produced those tensors. Serializing
this record alone would embed transformed weights, such as a second,
transposed copy.

The existing ``LocalRewriting`` in
``onnx_light/onnx_core/builder/pattern_optimization.h`` contains:

.. code-block:: cpp

   // Existing fields; namespace qualifiers omitted.
   std::shared_ptr<const PatternOptimization> pattern;
   std::vector<std::string> graph_path;
   std::vector<std::size_t> matched_nodes;
   RepeatedProtoField<NodeProto> added_nodes;
   std::vector<std::size_t> added_nodes_positions;
   RepeatedProtoField<TensorProto> added_initializers;
   std::vector<std::size_t> added_initializer_positions;
   std::vector<std::size_t> removed_initializers;
   std::vector<std::pair<std::string, std::string>> value_renames;
   std::size_t iteration;
   int64_t match_time_ns;
   int64_t apply_time_ns;

Decision: use protos as runtime records
++++++++++++++++++++++++++++++++++++++++

**Replace ``LocalRewriting`` with ``LocalRewritingProto`` as the record used
by the optimizer and existing replay engine.** ``RewritingProto`` holds the
ordered records and their shared function definitions/imports. Do not maintain
parallel C++/proto records or introduce a second replay engine and conversion
layer.

Migrate ``Optimize`` results, replay arguments, cleanup recording, C++/Python
bindings, diagnostics, tests and examples to the new messages. Keep matching
and ``MatchResult`` separate: the live pattern pointer belongs to matching,
not the recorded edit. Replay must stop requiring that pointer while retaining
its structural validation. Keep timing in ``OptimizationReport`` rather than
requiring it in replay records.

.. code-block:: text

   Optimize + capture recipes -> RewritingProto -> existing Replay, extended
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

New and modified proto messages
+++++++++++++++++++++++++++++++

The proposed wire schema is below. These are message definitions to implement
with onnx-light's existing ``BEGIN_PROTO`` / ``FIELD_*`` layer, parser and
serializer, not a requirement to add a protobuf compiler or ``libprotobuf``.

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
   }

   message RewritingProto {
     repeated LocalRewritingProto rewrites = 1;
     repeated OperatorSetIdProto required_opset_imports = 2;
     repeated StringStringEntryProto metadata_props = 3;
     repeated FunctionProto added_functions = 4;
   }

   message RewritingEntryProto {
     string key = 1;
     RewritingProto value = 2;
   }

**Modify only onnx-light's ``ModelProto``** to add a repeated
``RewritingEntryProto rewritings`` field, exposed as a map keyed by variant
name. Reserve and document its extension field number before implementation;
no number is assigned by this proposal. The entry message is the wire
representation of ``map<string, RewritingProto>``. Reject duplicate keys.

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

Extend optimizer capture to populate ``LocalRewritingProto.initializer_calls``
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

Before applying local edits, replay adds collection-level functions to
``ModelProto.functions``: add absent definitions, reuse identical definitions,
and reject conflicting definitions with the same identity, including conflicts
between local edits during capture. Add a model-level replay API for
functions/imports while retaining the graph-only entry point where useful;
both use the existing replay engine, migrated to proto records. Update
dependent callers explicitly. Validate constant-only, deterministic preparation
operations and available implementations; do not execute arbitrary stateful
functions as initializer recipes.

This avoids storing a second transposed weight payload in the file, not
necessarily allocating a second buffer during preparation. Reuse a view only
when supported by the tensor representation and consumer. Materialized plain
ONNX export may intentionally store computed weights instead.

``LocalRewritingProto.value_renames`` reuses ``StringStringEntryProto``:
``key`` is the old reference and ``value`` its replacement. It mirrors
existing ``LocalRewriting.value_renames`` and redirects references to an
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
   * - 1. Proto records
     - Add ``LocalRewritingProto`` and ``RewritingProto`` and replace the
       existing C++ record. Migrate optimizer, cleanup and replay to ordered
       proto edits, with position rebasing; update bindings and direct callers.
     - Binary round-trip; no matcher pointer or duplicate record type;
       sequential replay preserves existing results, including batch edits;
       invalid positions rejected.
   * - 2. Extend existing replay
     - Capture initializer recipes and shared function definitions directly
       in the protos. Extend existing replay for one-time constant preparation
       and model-level function/import additions; preserve payload ownership.
     - Serialized and in-memory replay agree; cover nested graphs, cleanup,
       initializers, renames, multi-batch edits and invalid plans. A transposed-weight
       recipe stores no duplicate payload; verify results, source removal
       ordering and function conflicts.
   * - 3. Model envelope
     - Add ``RewritingEntryProto`` and the ``ModelProto.rewritings`` field,
       model I/O and bindings; define explicit selection and compatibility.
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
canonical model for later replay, without rerunning the matcher. Updating
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
