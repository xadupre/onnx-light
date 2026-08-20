.. _l-next-steps-model-resolution:

Model resolution before weight loading
======================================

:Date: 2026-08

**discussion**

Objective
+++++++++

Model resolution is a strict preliminary stage between model metadata parsing
and :ref:`l-next-steps-prepared-execution`. It must determine the final graph,
kernel requirements, prepacked substitutions, and exact payload set before the
runtime submits any general weight I/O.

This ordering is required because:

* a graph transformation may remove an initializer, replace it, create a
  derived initializer, fuse several weights, or make another branch reachable;
* a compatible prepared object may replace a portable source-plus-prepack
  recipe, so reading the source would waste I/O and memory;
* kernel and device selection determine which physical representation is
  compatible.

The complete boundary is:

.. code-block:: text

    model and cache metadata
      -> transformations and analyses
      -> prepacked substitution and payload liveness
      -> immutable ResolvedModel
      -> PreparedExecutionPlan
      -> payload reads

No I/O task for a large portable or prepacked payload may be submitted before
``ResolvedModel`` is frozen.

Resolved model contract
+++++++++++++++++++++++

Resolution produces an immutable object:

.. code-block:: cpp

    struct ResolvedModel {
      ModelProto metadata_model;  // transformed graph, payloads still omitted
      std::optional<std::string> selected_scenario;
      InferredGraph inferred_graph;
      WeightDescriptors weights;
      std::vector<PreparedRequirementDescriptor> prepared_requirements;
      RequiredPayloadManifest payload_manifest;
      TransformationReport transformations;
    };

``RequiredPayloadManifest`` contains active payloads and dormant fallback
recipes, not every initializer found in the portable or prepared model. A
compatible cache hit makes the packed payload active and leaves the portable
source descriptor available only as fallback. A miss makes the portable source
and prepack recipe active. An initializer that is dead after transformation
appears in neither set.

The resolved object owns or shares every model source, prepared-model store,
external-data reader, and descriptor needed by the later execution plan. These
resources must outlive all deferred reads.

Resolution pipeline
+++++++++++++++++++

The pipeline is ordered and bounded:

.. code-block:: text

    parse metadata and payload descriptors
      -> select and replay one serialized modification scenario
      -> portable canonicalization / additional GraphGraph rewrites
      -> cleanup and recursive subgraph/function liveness
      -> shape, type, and constant-result inference
      -> target-aware rewrites and device placement
      -> cleanup and inference again
      -> kernel selection and prepared-object requirements
      -> compatible prepack resolution
      -> payload dependency closure
      -> freeze ResolvedModel

The number and order of transformation phases are configuration, not an
unconstrained fixed-point loop. A target-aware transformation that changes
graph topology invalidates cleanup, inference, placement, and kernel selection.
Those analyses run again before prepared-object resolution. No transformation
may mutate the model after kernel requirements or the payload manifest have
been frozen.

Step 1: catalogue payloads
++++++++++++++++++++++++++

``ParseOptions.skip_raw_data`` loads graph structure while skipping large
``raw_data`` fields. External initializers still expose:

.. code-block:: text

    name
    element type
    dimensions
    external file
    offset
    length
    checksum, when available

Parsing produces descriptors instead of runtime tensors:

.. code-block:: cpp

    struct WeightDescriptor {
      std::string name;
      int32_t element_type;
      std::vector<int64_t> dimensions;
      PayloadOrigin origin;  // portable, prepared cache, or derived
      PayloadDescriptor payload;
      std::optional<SourceDigest> source_digest;
      std::optional<DerivedWeightRecipe> derivation;
    };

``PayloadDescriptor`` records the model file or external file, offset, length,
checksum, and a recoverable inline byte range when applicable. It identifies
bytes without reading them. Inline ``raw_data`` can only be recovered after
metadata-only parsing when the parser retains a seekable source and exact byte
offsets into the original model.

The prepared companion model produces the same kind of descriptors. Parsing
does not yet choose them over portable sources because transformed-graph
liveness and kernel compatibility are not known.

Step 2: transform the metadata model
++++++++++++++++++++++++++++++++++++

The runtime imports the metadata model and lazy initializer descriptors into
``GraphBuilder``. It then runs configured ``GraphGraph`` phases or replays a
captured ``std::vector<LocalRewriting>``. ``GraphGraph`` mutates its associated
builder; the returned rewriting list is the audit and replay record, not a
second model.

Each transformation reports:

* removed and created values and initializers;
* descriptors for new payloads or recipes deriving them from existing
  descriptors;
* source lineage required to validate prepared-cache entries;
* whether shape, type, constant, liveness, placement, or kernel information
  became stale.

Cleanup after every topology-changing phase removes dead nodes,
``value_info``, functions, and initializers recursively before payload closure
is computed.

The builder must preserve omitted initializer payloads as lazy descriptors.
Treating an omitted empty ``raw_data`` as the actual tensor is an error. A
replacement initializer is one of:

* a small value materialized by the transformation;
* an existing lazy descriptor under its preserved logical value;
* a ``DerivedWeightRecipe`` with explicit source descriptors;
* a reference to a prepared requirement resolved later.

A preliminary transformation must decide from metadata and available small
constants. It must not silently materialize a large omitted tensor. A
payload-dependent transformation instead returns a ``DerivedWeightRecipe``.
Its sources become active only when no compatible prepared result satisfies the
derived requirement. A fused QKV requirement can therefore use one persisted
packed payload without first reading three portable matrices.

Serialized modification scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A model may persist several ordered rewrite sequences returned by
``GraphGraph::Optimize``. The first version applies either no scenario or
exactly one named scenario; scenarios are alternatives and are not composed.
The absence of a scenario is represented by ``std::nullopt``, not by an empty
string.

Scenario selection boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runtime must select the scenario before replay, liveness, and any weight
read. It must not start loading a scenario-specific file speculatively and
later discover that another scenario was selected.

Selection is based only on information available without tensor payloads:

* an explicit caller option;
* model and scenario metadata;
* processor, accelerator, runtime, memory, or kernel capabilities;
* deployment configuration.

The selected scenario is immutable for one ``ResolvedModel`` and
``RuntimeSession``. If the choice genuinely depends on an inference input or
mutable runtime state, it cannot use this static scenario mechanism. The caller
must create the appropriate resolved session lazily once the choice is known,
or represent the choice as ordinary graph control flow with demand-driven
weights.

The resolution order is therefore:

.. code-block:: text

    parse model and scenario metadata
      -> select exactly one scenario or the default graph
      -> replay metadata-only modifications
      -> compute final liveness and materialization recipes
      -> discover required external_data locations
      -> validate selected files
      -> freeze payload manifest
      -> start weight I/O

``LocalRewriting`` cannot be serialized directly because it contains a
``shared_ptr<PatternOptimization>`` and process-local timing information. A
protobuf record stores only replay data:

.. code-block:: text

    message LocalRewritingProto {
        string operation_name = 1;                                            // diagnostic name of the pattern or cleanup
        repeated string graph_path = 2;                                       // stable path from the root to the modified graph
        repeated uint64 removed_nodes = 3;                                    // node positions before this transition
        repeated NodeProto added_nodes = 4;                                   // replacement nodes owned by this transition
        repeated uint64 added_nodes_positions = 5;                            // positions after this transition
        repeated TensorProto added_initializers = 6;                          // initializers appended by this transition
        repeated uint64 removed_initializers = 7;                             // initializer positions before this transition
        repeated StringStringEntryProto metadata_props = 8;                   // extension metadata
    }

    message ModelModificationScenarioProto {
        string scenario = 1;                                                  // unique non-empty scenario selected by the loader
        bytes base_graph_digest = 2;                                          // digest of the graph before any scenario rewrite
        string digest_algorithm = 3;                                          // algorithm used for base_graph_digest
        uint32 replay_format_version = 4;                                     // replay ABI for the serialized records
        repeated LocalRewritingProto rewrites = 5;                            // ordered sequential transitions
        repeated StringStringEntryProto metadata_props = 6;                   // scenario-level metadata
    }

    message ModelProto {
        ...
        repeated ModelModificationScenarioProto modification_scenarios = <P>; // proposed alternatives
    }

``<P>`` (*proposed*) is the convention used in this plan for an ONNX field that
does not exist in the current format. A prototype allocates ``P`` in the local
provisional range ``20000..20999``. ``<P>`` is not valid protobuf syntax and
this range is not reserved upstream; adoption by ONNX may require another tag
and a migration. ``LocalRewritingProto`` and
``ModelModificationScenarioProto`` are likewise proposed messages, not current
ONNX definitions.

``operation_name`` is diagnostic; replay does not require the original pattern
implementation. ``MatchResult::insert_at`` has already been resolved into
``LocalRewriting::added_nodes_positions``. Match/apply timings, ``iteration``,
and added initializer positions are deliberately omitted.

Each proto is one normalized, sequential graph transition. ``removed_nodes``
contains positions in the graph immediately before that transition.
``added_nodes_positions`` has exactly one entry per ``added_nodes`` value and
contains its position in the graph immediately after the transition. Surviving
nodes fill all other positions in their previous relative order. Added
initializers are appended in record order because initializer order has no
semantic meaning; ``removed_initializers`` refers to positions before that
transition. ``graph_path`` identifies the nested graph receiving the
transition.

Conversion normalizes the ordered ``LocalRewriting`` list into sequential
coordinates. For records originally produced in one internal ``GraphGraph``
batch, it adjusts later positions as if earlier records had already been
applied and computes every exact final added-node position.
``ModelModificationScenarioProto.rewrites`` order is therefore sufficient for
replay; no serialized batch number or insertion hint is needed.

Renaming an existing value or initializer is not a valid replay operation.
``LocalRewritingProto`` therefore has no rename map. Replacement nodes must
preserve every externally visible output name they replace. When an internal
``LocalRewriting::value_renames`` only represents consumer rewiring, conversion
materializes the affected consumers as explicit removed/added nodes at the same
positions. Conversion fails if doing so would rename a graph input, graph
output, initializer, node output, or nested capture.

Metadata keys are unique within each proto. Unknown metadata is preserved
during round trips and has no replay semantics unless its key is explicitly
recognized by the replay-format version.

``base_graph_digest`` covers the graph, functions, initializer names, types,
dimensions, and payload identities before replay, but not physical external
locations or ``modification_scenarios``. It therefore remains stable as
scenarios are appended while preventing a valid rewrite list from being
applied to a different graph. Scenario names are non-empty and unique. Unknown
scenarios, duplicate names, unsupported replay versions, digest mismatches,
malformed positions, and invalid graph paths fail before payload selection.

Registration API
^^^^^^^^^^^^^^^^

The requested function appends one scenario after converting and validating
the records:

.. code-block:: cpp

    struct AddModificationScenarioOptions {
      std::string digest_algorithm = "blake3";
      uint32_t replay_format_version = 1;
      bool replace_existing = false;
    };

    void AddModificationScenario(
        ModelProto &model,
        const std::string &scenario,
        const std::vector<LocalRewriting> &rewrites,
        const AddModificationScenarioOptions &options = {});

    GraphProto ReplayModificationScenario(
        const ModelProto &model,
        const ModelModificationScenarioProto &scenario);

The function:

1. rejects an empty scenario and, unless ``replace_existing`` is true, a
   duplicate scenario;
2. verifies that replaying the supplied records against the current base model
   succeeds and that the resulting graph passes structural checks;
3. normalizes the ordered records into sequential ``LocalRewritingProto``
   coordinates without serializing pattern ownership, timings, iteration
   numbers, or insertion hints;
4. computes the base graph digest and appends
   ``ModelModificationScenarioProto``;
5. registers metadata and derivation descriptors for added initializers, but
   does not repartition or write weight files.

Weight files cannot be split correctly inside
``AddModificationScenario`` because adding a later scenario can change the
variant-usage classification of existing weights. Partitioning therefore
occurs once, when the complete model package is written.

Scenario weight file layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^

No additional model field is needed to describe weight groups. Every base
initializer and every initializer stored in
``LocalRewritingProto.added_initializers`` already carries its physical
location through ``TensorProto.external_data``. After replay, the loader can
reconstruct the required file set by collecting the locations of live
initializers.

The writer exposes:

.. code-block:: cpp

    enum class ScenarioWeightLayout {
      kKeepExisting,
      kSplitCommonDefaultAndScenario,
    };

    struct ModelPackageWriteOptions {
      ScenarioWeightLayout scenario_weight_layout =
          ScenarioWeightLayout::kKeepExisting;
      std::string weight_file_prefix;
    };

    void WriteModelPackage(
        const ModelProto &model,
        const ModelPackageWriteOptions &options);

For ``kSplitCommonDefaultAndScenario``, the writer replays every scenario
metadata-only and computes the set of variants using each initializer:

.. code-block:: text

    users(W) = {variants in which W is live}

    users(W) == {default}  -> default file
    users(W) == {S}        -> scenario S file
    otherwise              -> common file

The common file is the shared pool: it includes weights used by every variant
and weights shared by any two or more variants. This stores each payload once
and avoids introducing a second scenario-to-location mapping. A constant added
only by scenario ``S`` goes to its scenario file. A base weight used only when
no scenario is applied goes to the default file.

The writer updates ``external_data.location``, ``offset``, ``length``, and
checksum in the owning base or added ``TensorProto``. File-group membership is
therefore reconstructible and is not serialized separately. Conventional file
names are:

.. code-block:: text

    <prefix>.common.data
    <prefix>.default.data
    <prefix>.<scenario>.data

Missing files and final validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model loading separates metadata availability from payload availability.
Parsing reads the base model, scenario records, and initializer descriptors
without opening every referenced weight file. It may therefore replay a
scenario even when the default file or another scenario's file is absent.

For a requested scenario, loading follows this order:

1. parse metadata and verify the selected scenario record;
2. replay its modifications without dereferencing weight payloads;
3. run cleanup and compute final initializer liveness;
4. collect payload descriptors directly from the live initializers;
5. close over derived and prepacked recipes;
6. verify that every live required initializer has exactly one eligible
   payload or derivation;
7. only then check the files containing active descriptors and freeze the
   payload manifest.

A missing unselected file is irrelevant. A missing file containing only a
weight removed by the selected modifications is also irrelevant. Resolution
fails when a weight remains live after modification and its selected payload
descriptor is absent, its selected file cannot be opened, its byte range is
invalid, or no selected prepared/derived recipe supplies it. The error reports
the selected scenario, initializer identity, and expected location.
Checksum validation still occurs when an execution task reads bytes; a checksum
failure activates an eligible fallback or fails that task explicitly.

File existence must not be checked globally before replay. A missing file is
ignored when none of its payloads survive recipe closure, for example because a
compatible prepacked payload replaced those portable sources. Conversely,
replay success must never be treated as proof that the selected model is
loadable: final liveness and selected-payload validation are mandatory before
``ResolvedModel`` is returned.

Step 3: infer available information
+++++++++++++++++++++++++++++++++++

Shape inference uses graph inputs, initializer types and dimensions, operator
attributes, and inferred intermediate shapes without loading weight payloads.

Constant-result analysis is distinct from constant folding. It determines that
a value is constant without necessarily computing its bytes:

1. initializers and ``Constant`` outputs are constant;
2. a deterministic node output is constant when all required inputs are
   constant and no external state is involved;
3. a shape result may be known from metadata even when tensor bytes are not;
4. values depending on graph inputs, randomness, mutable state, or unsupported
   control flow remain dynamic.

The result distinguishes:

.. code-block:: text

    value name -> dynamic
                constant, bytes not materialized
                constant, small value known

Operators such as ``Shape``, ``Size``, ``Gather`` over a known shape, and small
shape arithmetic may provide metadata-only evaluators. Resolution must not fold
every large constant merely because it can prove that the result is constant.

Step 4: collect kernel requirements
+++++++++++++++++++++++++++++++++++

After target-aware transformations and analysis are stable, resolution selects
the device and execution kernel for every node. Kernel initialization receives
the node, inferred graph, constant information, and weight descriptors, but
must not load weights directly:

.. code-block:: cpp

    KernelInitialization Kernel::Initialize(
        const NodeProto &node,
        const InferredGraph &graph,
        const WeightDescriptors &weights);

It returns logical prepared-object requirements and materialization
alternatives. Each alternative declares its input descriptors, output
``PreparedKey``, device, kernel ABI, physical layout, dependencies, estimated
I/O, operations, and peak temporary memory.

For ``Gemm``, the requirement includes ``transB`` because the prepared object
represents ``op(B)``. Two consumers of one initializer may therefore request
different packed objects. A shared compatible request is deduplicated before
payload closure.

Step 5: resolve prepacked substitutions
+++++++++++++++++++++++++++++++++++++++

For each prepared requirement, the resolver orders eligible recipes:

.. code-block:: text

    resident prepared object
    compatible persisted packed payload
    live derived initializer recipe
    portable source plus prepack
    unpacked portable fallback, when supported

Prepacked substitution does not change ONNX tensor semantics and does not
replace a node input with implementation-specific bytes. The graph retains its
logical value while the selected kernel binds that value to a prepared object:

.. code-block:: cpp

    struct ResolvedWeightBinding {
      std::string logical_value;
      PreparedKey prepared_key;
      MaterializationRecipe selected;
      std::vector<MaterializationRecipe> fallbacks;
      std::vector<SourceIdentity> source_lineage;
    };

.. code-block:: text

    logical B
      -> compatible packed payload --------> publish PreparedKey(B)
      -> portable B -> prepack B [fallback] -> publish PreparedKey(B)

Compatibility includes complete source lineage, target device and processor
features, kernel ABI, packed-format version, shape, quantization parameters,
and payload metadata. An unavailable trusted source digest makes the cache
entry unverified: resolution selects the portable source/prepack recipe without
reading the source merely to decide the manifest.

Liveness starts from graph outputs and stateful effects over the transformed
graph, then closes transitively over the selected recipes. It includes sources
needed by a live derived recipe but excludes sources bypassed by a valid packed
entry. It applies recursively to subgraphs and functions. Duplicate or tied
initializers share one read only when source identity and required byte ranges
are exactly compatible.

The frozen manifest records why every descriptor is active and which
requirement or fallback consumes it. Every later read task must reference one
active manifest entry.

API
+++

.. code-block:: cpp

    struct ScenarioSelectionContext {
      const ModelProto &metadata_model;
      const DeviceCapabilities &devices;
    };

    using ScenarioSelector = std::function<std::optional<std::string>(
        const ScenarioSelectionContext &)>;

    struct ModelResolutionOptions {
      GraphTransformationPipeline transformations;
      PreparedModelOptions prepared;
      std::optional<std::string> scenario;  // nullopt allows selector or base graph
      ScenarioSelector select_scenario;     // optional metadata-only selector
    };

    ResolvedModel ResolveModel(
        const ModelProto &source,
        const ModelResolutionOptions &options);

``ResolveModel`` is synchronous metadata planning. It may inspect descriptors
and small inline constants but does not enqueue general weight I/O. It replays
the explicit ``options.scenario`` or invokes ``select_scenario`` before
caller-supplied transformations, then validates only the files required by
final liveness and recipe closure. Supplying both selection mechanisms is an
error; supplying neither selects the base graph. ``ScenarioSelector`` receives
model/scenario metadata and device capabilities but cannot access weight
payloads. The runtime then transfers ownership to the execution-plan builder:

.. code-block:: cpp

    ResolvedModel resolved = ResolveModel(source, options);
    PreparedExecutionPlan plan =
        BuildPreparedExecutionPlan(std::move(resolved));

Validation
++++++++++

Tests must verify that:

* ``AddModificationScenario`` round-trips every replay field and reconstructs
  the same root graph and nested subgraphs;
* empty/duplicate/unknown scenarios, base digest mismatches, malformed replay
  positions, and unsupported versions fail explicitly;
* explicit and callback selection are mutually exclusive, selector access to
  weight payloads is impossible, and an unknown selected scenario fails before
  file validation;
* attempts to rename existing values or initializers fail, while internal
  consumer rewiring is serialized as explicit node replacement;
* split packaging reconstructs exact variant usage, stores each payload once,
  and updates ``external_data`` for initializers created or removed by rewrites;
* a missing default or unrelated scenario file does not prevent another
  scenario from resolving;
* a missing file whose weights are all removed or replaced after replay is
  ignored, while a missing payload still live after replay fails with its
  scenario, initializer identity, and location;
* a removed or dead initializer is never read;
* a replacement initializer and the sources of a selected derived recipe are
  read exactly once;
* a compatible packed entry skips all portable sources in its lineage;
* an incompatible packed entry selects the portable recipe;
* a corrupt packed payload activates the dormant portable fallback at
  execution time rather than producing a successful cache result;
* nested subgraphs, functions, duplicate initializers, and tied weights produce
  the correct closure;
* changing the graph after resolution is rejected;
* every execution-plan read belongs to the frozen manifest.

The benchmark separately measures metadata parsing, transformation and cleanup,
shape/constant analysis, placement and kernel selection, cache resolution, and
payload closure. No weight-read time is included in the resolution measurement.

Implementation order
++++++++++++++++++++

1. Build recoverable ``WeightDescriptor`` and ``PayloadDescriptor`` objects
   for portable and prepared models with ``skip_raw_data=true``.
2. Add lazy initializer descriptors to ``GraphBuilder`` and distinguish omitted
   payloads from real empty tensors.
3. Add ``LocalRewritingProto``, ``ModelModificationScenarioProto``, conversion,
   base-graph digest validation, and ``AddModificationScenario``.
4. Add package writing for common, default, and per-scenario files by rewriting
   initializer ``external_data`` descriptors without adding a model field.
5. Add explicit and metadata-only callback scenario selection, followed by
   replay, recursive cleanup, invalidation, and source-lineage reporting.
6. Add post-replay file validation driven only by final live descriptors,
   including precise missing-weight diagnostics.
7. Add constant-result propagation without payload materialization.
8. Add device/kernel selection and prepared-object requirement queries without
   direct weight reads.
9. Add ``DerivedWeightRecipe``, compatible prepared-entry resolution, and
   ``RequiredPayloadManifest`` closure.
10. Add the validation and benchmark cases above before wiring the result into
    :ref:`l-next-steps-prepared-execution`.
