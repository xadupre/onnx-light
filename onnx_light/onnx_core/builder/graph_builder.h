// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file graph_builder.h
 * @brief Incremental builder for ONNX graphs, models and functions.
 *
 * :cpp:class:`core::builder::GraphBuilder` accumulates graph inputs,
 * initializers, nodes and outputs one call at a time and keeps the
 * associated compute metadata (shapes, in-place reuse, value tags and
 * per-node peak memory) up to date through an owned
 * :cpp:class:`core::compute::ComputeContext`.
 *
 * The builder does not use a :cpp:class:`GraphProto` as its working container.
 * Inputs, outputs, nodes, initializers and the nested local functions /
 * subgraphs (each of which is itself a :cpp:class:`GraphBuilder`) are kept in
 * plain vectors, in declaration order; each entry carries its own name so no
 * side map is needed. A proto is only materialised on demand by
 * :cpp:func:`BuildGraph` and the finalizers.
 *
 * A builder starts empty. Every value name it hands out (graph inputs,
 * initializers and node outputs) is recorded so a name can never be reused.
 * Each :cpp:func:`GraphBuilder::MakeNode` call resolves the operator opset
 * (falling back to the latest known one when the domain has no explicit
 * opset), validates the node against the matching
 * :cpp:class:`core::schema::LightOpSchema` when one is available, assigns
 * output names when the caller left them empty and runs incremental shape
 * inference for the new node.
 *
 * :cpp:func:`GraphBuilder::ToModel`, :cpp:func:`GraphBuilder::ToGraph` and
 * :cpp:func:`GraphBuilder::ToFunction` finalize the accumulated graph: they
 * run the whole-graph compute analyses and write the inferred shapes, the
 * in-place / release-after metadata, the value tags and the peak-memory
 * estimates into the produced proto.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx_core/compute/compute_context.h"
#include "onnx_core/light_op_schema/light_op_schema.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

using ::onnx_light::core::compute::ComputeContext;
using ::onnx_light::core::schema::LightOpSchema;
using ::onnx_light::core::shapes::ShapesContext;
using ::onnx_light::core::symbolic::Device;
using ::onnx_light::core::symbolic::SymShape;
using ::onnx_light::core::symbolic::SymTensor;
using ::onnx_light::core::symbolic::TensorType;

/// Read-only index over a :cpp:class:`GraphBuilder` used by the pattern
/// optimizer (declared in ``graph_graph.h``).
class GraphGraph;

/// Selects which ONNX proto :cpp:func:`GraphBuilder::ToOnnx` produces.
enum class ProtoKind {
  /// Produce a :cpp:class:`ModelProto` (the default).
  kModel,
  /// Produce a bare :cpp:class:`GraphProto`.
  kGraph,
  /// Produce a :cpp:class:`FunctionProto`.
  kFunction,
};

/// Thrown when :cpp:class:`GraphBuilder` is used incorrectly, for example when
/// a name is reused or the opset version of a domain cannot be resolved.
class BuilderError : public std::runtime_error {
public:
  /// Constructs a BuilderError with the given diagnostic message.
  explicit BuilderError(const std::string &message) : std::runtime_error(message) {}
};

/// Options controlling :cpp:func:`GraphBuilder::ConstantFold`.
///
/// Constant folding evaluates every node whose outputs are known before
/// inference (initializers and, transitively, the outputs of deterministic
/// nodes fed only by constants) and replaces the node with the resulting
/// initializers. The evaluation uses the process-wide runtime kernel registry
/// (:cpp:func:`core::runtime::KernelDispatchTable`), which must be populated by
/// a kernel library (``onnx_kernels``) for folding to happen.
struct ConstantFoldingOptions {
  /// Master switch: when ``false`` :cpp:func:`ConstantFold` is a no-op and
  /// returns ``0`` without touching the graph.
  bool enabled = true;

  /// Skips folding a node when any of its outputs would hold strictly more than
  /// ``max_element_count`` elements. A negative value (the default) means no
  /// limit, so results of any size are folded.
  int64_t max_element_count = -1;

  /// ``(domain, op_type)`` pairs that must never be folded. An empty domain
  /// matches every domain and an empty op_type matches every operator, so an
  /// empty-empty pair disables folding for every node. The domain is matched
  /// after normalisation (``""`` and ``"ai.onnx"`` compare equal). A ``std::set``
  /// is used so exclusion lookups stay logarithmic instead of scanning a vector.
  std::set<std::pair<std::string, std::string>> excluded_ops;

  /// Controls whether nodes whose results are tagged ``"weight"`` (or untagged)
  /// are folded. Shape-tagged results are always foldable and can be folded at
  /// any point in an optimization pipeline; weight results are usually better
  /// folded only at the end (after other passes have run), so this switch lets a
  /// caller fold shapes early and defer weight folding to a final pass. When
  /// ``false`` only shape-tagged results are folded.
  bool fold_weights = true;

  /// When ``true`` a node whose outputs are tagged ``"weight"`` (or untagged)
  /// but for which no runtime kernel is registered raises a
  /// :cpp:class:`BuilderError` instead of being left untouched. Nodes whose
  /// outputs are tagged ``"shape"`` always raise when their kernel is missing,
  /// regardless of this flag.
  bool raise_on_missing_weight_kernel = false;
};

/**
 * Incrementally builds an ONNX graph, model or function while keeping the
 * associated compute metadata up to date.
 *
 * The builder owns a :cpp:class:`core::compute::ComputeContext`; every node
 * added through :cpp:func:`MakeNode` is immediately fed to incremental shape
 * inference so the shape of any produced value can be queried mid-build with
 * :cpp:func:`GetShape`.
 */
class GraphBuilder {
public:
  // The pattern optimizer's read-only index needs access to the private
  // subgraph-reference helpers (ReferencedSubgraphs / CollectImplicitInputs)
  // to compute the set of values captured by nested subgraphs.
  friend class GraphGraph;

  /// Signature of the optional callback used to resolve the versioned schema
  /// history of an operator. Given an ``op_type`` it returns every
  /// :cpp:class:`core::schema::LightOpSchema` registered for that operator
  /// (across every domain); an empty vector means the operator is unknown.
  /// ``onnx_core`` owns :cpp:class:`core::schema::LightOpSchema` but not the
  /// built-in operator schemas (those live in the ``onnx_op`` library, which
  /// depends on ``onnx_core``), so the provider is injected by the caller (see
  /// the Python bindings, which wire the built-in ONNX schemas).
  using SchemaLookupFn = std::function<std::vector<LightOpSchema>(const std::string &op_type)>;

  /// Constructs an empty builder.
  ///
  /// @param name          Name given to the produced graph / function.
  /// @param schema_lookup Optional schema provider used to validate nodes and
  ///                      to resolve the "latest opset" of a domain.
  explicit GraphBuilder(std::string name = "graph", SchemaLookupFn schema_lookup = {});

  /// Constructs a builder from an existing model by replaying every graph and
  /// function node through :cpp:func:`MakeNode`.
  ///
  /// Graph-valued node attributes are converted into nested subgraph builders,
  /// and the owning node stores a ``<attr_name>_ref`` STRING (or STRINGS)
  /// attribute carrying the nested builder name(s). :cpp:func:`BuildGraph` /
  /// :cpp:func:`ToGraph` / :cpp:func:`ToModel` materialize those references
  /// back into GRAPH / GRAPHS attributes.
  explicit GraphBuilder(const ModelProto &model, SchemaLookupFn schema_lookup = {});

  ~GraphBuilder();

  GraphBuilder(GraphBuilder &&) noexcept;
  GraphBuilder &operator=(GraphBuilder &&) noexcept;
  GraphBuilder(const GraphBuilder &) = delete;
  GraphBuilder &operator=(const GraphBuilder &) = delete;

  /// Name given to the produced graph / function.
  const std::string &name() const noexcept { return name_; }

  // ── Opset management ─────────────────────────────────────────────────

  /// Records the opset version to use for ``domain`` (an empty string denotes
  /// the default ONNX domain). Explicitly setting an opset prevents the
  /// builder from deriving it from operator schemas.
  void SetOpsetVersion(const std::string &domain, int version);

  /// Returns the opset version recorded for ``domain`` or
  /// :cpp:var:`core::shapes::kUnknownOpsetVersion` when none is set.
  int OpsetVersion(const std::string &domain) const;

  /// Read-only access to the recorded ``domain -> opset version`` map.
  const std::unordered_map<std::string, int> &OpsetVersions() const noexcept { return opsets_; }

  // ── Name management ──────────────────────────────────────────────────

  /// Returns ``true`` when ``name`` has already been handed out.
  bool HasName(const std::string &name) const noexcept;

  /// Records ``name`` as used and returns it. Throws :cpp:class:`BuilderError`
  /// when the name is empty or already used.
  const std::string &ReserveName(const std::string &name);

  /// Returns a fresh, unused name starting with ``prefix`` and records it.
  std::string UniqueName(const std::string &prefix = "n");

  // ── Initializers ─────────────────────────────────────────────────────

  /// Appends ``tensor`` as a graph initializer. The tensor may carry external
  /// data (``data_location == EXTERNAL``). Returns the initializer name.
  const std::string &MakeInitializer(const TensorProto &tensor);

  /// Builds and appends an initializer whose data lives in an external file.
  ///
  /// @param name     Initializer name.
  /// @param dtype    Element type.
  /// @param dims     Tensor shape.
  /// @param location Path of the external data file (relative to the model).
  /// @param offset   Byte offset of the data inside the file.
  /// @param length   Number of bytes of the data inside the file.
  /// @return The initializer name.
  const std::string &MakeExternalInitializer(const std::string &name, TensorType dtype,
                                             const std::vector<int64_t> &dims,
                                             const std::string &location, int64_t offset,
                                             int64_t length);

  /// Read-only access to the graph initializers, in declaration order.
  const utils::RepeatedProtoField<TensorProto> &Initializers() const noexcept {
    return initializers_;
  }

  // ── Inputs / outputs ─────────────────────────────────────────────────

  /// Declares a graph input from a ready-made :cpp:class:`ValueInfoProto` and
  /// returns its name.
  const std::string &MakeInput(const ValueInfoProto &value_info);

  /// Declares a graph input described by ``type`` and returns its name.
  const std::string &MakeInput(const std::string &name, const SymTensor &type);

  /// Declares a graph input with element type ``dtype`` and shape ``shape``.
  const std::string &MakeInput(const std::string &name, TensorType dtype, const SymShape &shape);

  /// Declares a graph output from a ready-made :cpp:class:`ValueInfoProto`.
  void MakeOutput(const ValueInfoProto &value_info);

  /// Declares ``name`` (which must already exist) as a graph output described
  /// by ``type``.
  void MakeOutput(const std::string &name, const SymTensor &type);

  /// Declares ``name`` as a graph output with element type ``dtype`` and shape
  /// ``shape``.
  void MakeOutput(const std::string &name, TensorType dtype, const SymShape &shape);

  /// Declares ``name`` as a graph output without a declared type; the inferred
  /// type is filled in by :cpp:func:`ToModel` / :cpp:func:`ToGraph`.
  void MakeOutput(const std::string &name);

  /// Read-only access to the declared graph inputs (in declaration order).
  const utils::RepeatedProtoField<ValueInfoProto> &Inputs() const noexcept { return inputs_; }

  /// Read-only access to the declared graph outputs (in declaration order).
  const utils::RepeatedProtoField<ValueInfoProto> &Outputs() const noexcept { return outputs_; }

  // ── Nodes ────────────────────────────────────────────────────────────

  /// Appends a node to the graph.
  ///
  /// The opset version of ``domain`` is resolved (defaulting to the latest
  /// known one when unset), the node is validated against the matching
  /// :cpp:class:`core::schema::LightOpSchema` when a schema provider is
  /// available, missing output names are generated, the node is appended and
  /// incremental shape inference is run for it.
  ///
  /// @param op_type    Operator type (e.g. ``"Add"``).
  /// @param inputs     Input value names.
  /// @param outputs    Output value names; empty entries (or a shorter list
  ///                   than the operator produces) are auto-generated.
  /// @param domain     Operator domain (empty for the default ONNX domain).
  /// @param name       Optional node name.
  /// @param attributes Node attributes.
  /// @return The (possibly generated) output names of the node.
  std::vector<std::string> MakeNode(const std::string &op_type,
                                    const std::vector<std::string> &inputs,
                                    const std::vector<std::string> &outputs = {},
                                    const std::string &domain = "", const std::string &name = "",
                                    const utils::RepeatedProtoField<AttributeProto> &attributes =
                                        utils::RepeatedProtoField<AttributeProto>());

  /// Read-only access to the accumulated nodes (in insertion order).
  const utils::RepeatedProtoField<NodeProto> &Nodes() const noexcept { return nodes_; }

  /// Removes dead-end (unused) nodes from the builder.
  ///
  /// A node is kept only when at least one of its outputs is (transitively)
  /// needed to compute a declared graph output. The analysis is recursive:
  /// removing a node can turn the nodes that only fed it into dead ends as
  /// well, and the pruning descends into nested subgraphs and local functions
  /// to remove their own unused nodes. Values a subgraph consumes from the
  /// enclosing scope are treated as inputs of the owning control-flow node, so
  /// the producers a subgraph body relies on are kept alive.
  ///
  /// @return The total number of nodes removed, including those pruned from
  ///         nested subgraphs and local functions.
  std::size_t RemoveUnusedNodes();

  /// Removes duplicated initializers from the builder.
  ///
  /// Initializers that carry byte-for-byte identical content (same element
  /// type, shape and data, whether stored inline or as external data) are
  /// collapsed onto a single copy: the first occurrence is kept and every
  /// later duplicate is dropped. The content is compared field by field --
  /// element type and shape first, then the payload -- without copying or
  /// serializing the tensors. All references to a dropped initializer -- in
  /// this builder's node inputs and in the node inputs of nested subgraphs,
  /// which capture values from the enclosing scope -- are rewritten to the
  /// surviving initializer name. An initializer that also happens to be a
  /// declared graph output keeps its own name and is never dropped.
  ///
  /// The deduplication spans the enclosing graph and its subgraphs: because a
  /// subgraph body sees the initializers of the enclosing scope, a subgraph
  /// initializer that duplicates one visible from an enclosing graph is dropped
  /// and its references rewritten to that enclosing initializer. Local
  /// functions have an isolated scope and are deduplicated on their own.
  ///
  /// @return The total number of initializers removed, including those pruned
  ///         from nested subgraphs and local functions.
  std::size_t RemoveDuplicateInitializers();

  /// Removes :onnx:`Identity` nodes from the builder.
  ///
  /// Every default-domain ``Identity`` node simply forwards its single input to
  /// its single output. Such a node is dropped and every reference to its
  /// output -- in this builder's node inputs and in the node inputs of nested
  /// subgraphs, which capture values from the enclosing scope -- is rewritten to
  /// the node's input. Chains of identities are collapsed transitively, so a
  /// value that flowed through several identities ends up pointing at the
  /// original producer in a single pass.
  ///
  /// An ``Identity`` whose output is a declared graph output is kept, because
  /// the graph must still produce a value under that name. Nodes with an empty
  /// input or output name are left untouched.
  ///
  /// The removal is recursive: it descends into nested subgraphs and local
  /// functions to remove their own identities as well.
  ///
  /// @return The total number of ``Identity`` nodes removed, including those
  ///         pruned from nested subgraphs and local functions.
  std::size_t RemoveIdentityNodes();

  /// Removes duplicated nodes (common subexpressions) from the builder.
  ///
  /// Two nodes are duplicates when they share the same operator type, domain,
  /// inputs and attributes and therefore compute the same value. The first
  /// occurrence is kept and every later duplicate is dropped; each reference to
  /// a dropped node's output -- in this builder's node inputs and in the node
  /// inputs of nested subgraphs, which capture values from the enclosing scope
  /// -- is rewritten to the surviving node's matching output. Inputs are
  /// resolved against earlier-dropped duplicates while nodes are scanned in
  /// insertion (topological) order, so a whole duplicated branch collapses in a
  /// single pass: once a node's producers point at the survivors, the node
  /// itself becomes a duplicate of the corresponding surviving node.
  ///
  /// A node whose output is a declared graph output is kept, because the graph
  /// must still produce a value under that name (it can still act as the
  /// survivor for a later duplicate). Nodes referencing control-flow subgraphs
  /// carry a per-node unique subgraph name and are never merged.
  ///
  /// The removal is recursive: it descends into nested subgraphs and local
  /// functions to collapse their own duplicates as well.
  ///
  /// @return The total number of duplicated nodes removed, including those
  ///         pruned from nested subgraphs and local functions.
  std::size_t RemoveDuplicateNodes();

  /// Inlines calls to local functions into the calling graph.
  ///
  /// A node calls a local function when its operator type and domain match a
  /// local function declared on this builder. Every such call is replaced,
  /// in place, by a renamed copy of the function body: the function formal
  /// inputs are rewired to the call inputs, the formal outputs to the call
  /// outputs, and every other body value (node outputs, and any function
  /// initializers or control-flow subgraphs) is copied under a fresh, unused
  /// name so it can never collide with a value of the calling graph. Body-node
  /// attributes that reference a function attribute (``ref_attr_name``) are
  /// resolved against the attributes carried by the call node, falling back to
  /// the operator default when the call leaves the attribute unset.
  ///
  /// The expansion runs to a fixed point, so a function that itself calls
  /// another local function is fully inlined in a single pass, and it descends
  /// into nested subgraphs, whose bodies may call the enclosing local
  /// functions too. Local function definitions that are no longer referenced
  /// once every call site has been inlined are dropped, so a fully inlined
  /// model carries no leftover function.
  ///
  /// The set of functions to inline is selected by ``(domain, name)`` pairs.
  /// A pair matches a local function when both its domain and name match; an
  /// empty domain matches every domain (all functions sharing the name) and an
  /// empty name matches every name (all functions in the domain), so an
  /// empty-empty pair matches every function. When both ``include`` and
  /// ``exclude`` are empty (the default) every local function is inlined. When
  /// ``include`` is non-empty only the functions matched by one of its pairs
  /// are inlined. When ``exclude`` is non-empty every local function except
  /// those matched by one of its pairs is inlined. Passing a non-empty
  /// ``include`` together with a non-empty ``exclude`` throws a ``BuilderError``.
  ///
  /// @param include ``(domain, name)`` pairs of the only functions to inline;
  ///                empty for all.
  /// @param exclude ``(domain, name)`` pairs of the functions to leave
  ///                untouched.
  /// @return The total number of call nodes that were inlined.
  std::size_t
  InlineLocalFunctions(const std::vector<std::pair<std::string, std::string>> &include = {},
                       const std::vector<std::pair<std::string, std::string>> &exclude = {});

  /// Folds constant subgraphs into initializers.
  ///
  /// A node is *constant* when every value it reads is constant (a graph
  /// initializer or, transitively, the output of an earlier constant node) and
  /// its operator is deterministic. Every such node is evaluated once through
  /// the process-wide runtime kernel registry
  /// (:cpp:func:`core::runtime::KernelDispatchTable`, populated by
  /// ``onnx_kernels``) and replaced by initializers carrying its computed
  /// outputs; the freshly materialized constants let the nodes that only fed it
  /// fold in the same pass.
  ///
  /// A node's outputs are classified by their inferred value tag. Outputs
  /// tagged ``"shape"`` (shape-carrying values, e.g. the output of
  /// :onnx:`Shape` or a :onnx:`Concat` of shapes) *must* be foldable: when no
  /// kernel is registered for such a node a :cpp:class:`BuilderError` is thrown.
  /// Every other constant node (``"weight"`` or untagged) is folded only when a
  /// kernel is available; otherwise it is left untouched, unless
  /// :cpp:member:`ConstantFoldingOptions::raise_on_missing_weight_kernel` asks
  /// for a :cpp:class:`BuilderError` instead.
  ///
  /// Folding is skipped for a node when it is listed in
  /// :cpp:member:`ConstantFoldingOptions::excluded_ops`, when any of its outputs
  /// would exceed :cpp:member:`ConstantFoldingOptions::max_element_count`
  /// elements, or when it carries a control-flow subgraph. A node whose output
  /// is a declared graph output is still folded: the computed constant is
  /// materialized as an initializer carrying that name, which remains a valid
  /// graph output. Setting :cpp:member:`ConstantFoldingOptions::enabled` to
  /// ``false`` turns the whole pass into a no-op.
  ///
  /// The pass is recursive: it descends into nested subgraphs and local
  /// functions to fold their own constants as well. Values a subgraph captures
  /// from the enclosing scope are not seeded as constants there, so a subgraph
  /// node reading such a capture is left untouched.
  ///
  /// @param options Folding options (enable flag, size threshold, op blacklist,
  ///                strictness for missing weight kernels).
  /// @return The total number of nodes folded away, including those folded in
  ///         nested subgraphs and local functions.
  std::size_t ConstantFold(const ConstantFoldingOptions &options = {});

  // ── Local functions / subgraphs ──────────────────────────────────────

  /// Creates and returns a nested builder for a local function named ``name``.
  /// The nested builder is appended to this builder's local function list;
  /// local functions are emitted into the produced :cpp:class:`ModelProto`.
  /// Throws when ``name`` is already used.
  GraphBuilder &MakeLocalFunction(const std::string &name, const std::string &domain = "");

  /// Returns ``true`` when a local function named ``name`` exists.
  bool HasLocalFunction(const std::string &name) const {
    return FindNamedBuilder(local_functions_, name) != nullptr;
  }

  /// Returns the nested local-function builder named ``name``. Throws when it
  /// does not exist.
  GraphBuilder &LocalFunction(const std::string &name) {
    return NamedBuilderOrThrow(local_functions_, name, "local function");
  }
  const GraphBuilder &LocalFunction(const std::string &name) const {
    return NamedBuilderOrThrow(local_functions_, name, "local function");
  }

  /// Read-only access to the local function list, in declaration order.
  const std::vector<std::unique_ptr<GraphBuilder>> &LocalFunctions() const noexcept {
    return local_functions_;
  }

  /// Creates and returns a nested builder for a subgraph named ``name`` (used
  /// as the body of a control-flow node such as :onnx:`If`, :onnx:`Loop` or
  /// :onnx:`Scan`). The nested builder is appended to this builder's subgraph
  /// list. Throws when ``name`` is already used.
  GraphBuilder &MakeSubgraph(const std::string &name);

  /// Returns ``true`` when a subgraph named ``name`` exists.
  bool HasSubgraph(const std::string &name) const {
    return FindNamedBuilder(subgraphs_, name) != nullptr;
  }

  /// Returns the nested subgraph builder named ``name``. Throws when it does
  /// not exist.
  GraphBuilder &Subgraph(const std::string &name) {
    return NamedBuilderOrThrow(subgraphs_, name, "subgraph");
  }
  const GraphBuilder &Subgraph(const std::string &name) const {
    return NamedBuilderOrThrow(subgraphs_, name, "subgraph");
  }

  /// Read-only access to the subgraph list, in declaration order.
  const std::vector<std::unique_ptr<GraphBuilder>> &Subgraphs() const noexcept {
    return subgraphs_;
  }

  // ── Queries ──────────────────────────────────────────────────────────

  /// Returns ``true`` when the shape of ``name`` has been inferred.
  bool HasShape(const std::string &name) const;

  /// Returns the inferred descriptor of ``name``. Throws when it is unknown.
  const SymTensor &GetShape(const std::string &name) const;

  /// Assembles (without finalising) the accumulated inputs, initializers,
  /// nodes and outputs into a :cpp:class:`GraphProto`.
  GraphProto BuildGraph() const;

  /// Read-only access to the owned :cpp:class:`ComputeContext`.
  const ComputeContext &Compute() const noexcept { return compute_; }

  /// Read-only access to the :cpp:class:`ShapesContext` holding the inferred
  /// descriptors computed so far.
  const ShapesContext &Shapes() const noexcept { return compute_.Shapes(); }

  /// Returns a comprehensive, human-readable description of the current content
  /// of the builder: its name, resolved opsets, inputs, initializers, nodes,
  /// outputs and nested local functions / subgraphs.
  std::string ToString() const;

  /// Logical device used for the peak-memory analysis run by the finalizers.
  void set_device(Device device) noexcept { device_ = device; }
  Device device() const noexcept { return device_; }

  // ── Finalization ─────────────────────────────────────────────────────

  /// Returns the finalized :cpp:class:`GraphProto`.
  GraphProto ToGraph();

  /// Returns the finalized graph wrapped in a :cpp:class:`ModelProto`.
  ///
  /// @param ir_version IR version to write; ``0`` selects the library default.
  ModelProto ToModel(int64_t ir_version = 0);

  /// Returns the finalized nodes wrapped in a :cpp:class:`FunctionProto`.
  ///
  /// @param domain Function domain.
  FunctionProto ToFunction(const std::string &domain = "");

private:
  std::size_t RemoveUnusedNodesImpl(bool recursive);
  std::size_t
  RemoveIdentityNodesImpl(bool recursive,
                          std::unordered_map<std::string, std::string> *applied_renames = nullptr);
  std::size_t
  RemoveDuplicateNodesImpl(bool recursive,
                           std::unordered_map<std::string, std::string> *applied_renames = nullptr);

  std::size_t ConstantFoldNodes(const ConstantFoldingOptions &options,
                                const std::unordered_set<std::string> &included_outputs);
  std::size_t ConstantFoldImpl(const ConstantFoldingOptions &options,
                               const std::unordered_set<std::string> *included_outputs);

  // Resolves and records the opset version to use for a node of ``domain``,
  // given the domain-filtered schema history ``schemas``.
  int ResolveNodeOpset(const std::string &domain, const std::vector<LightOpSchema> &schemas);

  // Returns the schema history registered for ``op_type`` in ``normalised_domain``
  // (empty when none), populating the lookup table on first use. ``schema_lookup_``
  // is queried at most once per operator; results are grouped by domain so node
  // validation is a map lookup rather than a linear scan on every call.
  const std::vector<LightOpSchema> &DomainSchemas(const std::string &op_type,
                                                  const std::string &normalised_domain);

  // Returns ``true`` when a shape function is registered for ``node``.
  bool ShapeFunctionAvailable(const NodeProto &node) const;

  // Returns the nested subgraph builders referenced by ``node`` through its
  // ``*_ref`` STRING / STRINGS attributes.
  std::vector<GraphBuilder *> ReferencedSubgraphs(const NodeProto &node) const;

  // Collects the implicit inputs of this builder (value names it references but
  // does not itself define as an input, initializer or node output) into
  // ``out``. Implicit inputs of nested subgraphs are resolved relative to this
  // scope and propagated when they remain undefined here.
  void CollectImplicitInputs(std::unordered_set<std::string> &out) const;

  // Appends to ``refs`` the value names ``node`` depends on: its explicit
  // inputs plus the implicit inputs of any nested subgraph it references.
  void CollectNodeReferences(const NodeProto &node, std::vector<std::string> &refs) const;

  // Rewrites node inputs that name a dropped initializer to the surviving name
  // given by ``rename`` (dropped name -> kept name). The rewrite descends into
  // nested subgraphs, whose bodies capture values from the enclosing scope.
  void RewriteInitializerReferences(const std::unordered_map<std::string, std::string> &rename);
  void RewriteCapturedReferences(const std::unordered_map<std::string, std::string> &rename);

  // Inlines every call (in this builder's nodes and, recursively, in its nested
  // subgraphs) to one of ``functions``, expanding matches to a fixed point.
  // Returns the number of call nodes inlined.
  std::size_t InlineFunctionCalls(const std::vector<GraphBuilder *> &functions);

  // Returns the function in ``functions`` called by ``node`` (its operator type
  // and domain match the function name and domain), or nullptr when none does.
  static GraphBuilder *FindCalledFunction(const std::vector<GraphBuilder *> &functions,
                                          const NodeProto &node);

  // Expands the body of ``function`` for the call ``call`` and appends the
  // renamed body nodes to ``out``. Function initializers and control-flow
  // subgraphs are cloned into this builder under fresh names.
  void AppendInlinedBody(GraphBuilder &function, const NodeProto &call,
                         utils::RepeatedProtoField<NodeProto> &out);

  // Counts, in this builder's nodes and recursively in its nested subgraphs,
  // the nodes that call the function named ``name`` in domain ``domain``.
  std::size_t CountFunctionCalls(const std::string &name, const std::string &domain) const;

  // Maps a cheap content hash to initializers visible in scope (pointers into
  // the owning builder's ``initializers_``). Collisions are resolved by an
  // exact field-by-field comparison. Used by RemoveDuplicateInitializers to
  // collapse duplicates across the enclosing graph and its subgraphs.
  using InitializerContentIndex = std::unordered_map<int64_t, std::vector<const TensorProto *>>;

  // Collapses duplicated initializers in this scope against ``inherited`` (the
  // initializers visible from enclosing graphs) and recurses into subgraphs
  // (passing down the augmented index) and local functions (with a fresh,
  // isolated index). Returns the number of initializers removed.
  std::size_t
  DeduplicateInitializers(const InitializerContentIndex &inherited, bool recursive,
                          std::unordered_map<std::string, std::string> *applied_renames = nullptr);

  // Seeds the incremental annotation state (value tag + in-place-reuse
  // lifetime) for a declared graph input named ``name``.
  void SeedInputAnnotations(const std::string &name);

  // Seeds the owned ShapesContext with the descriptor of ``name``.
  void SeedShape(const std::string &name, SymTensor tensor);

  // Imports ``graph`` by replaying its inputs, initializers, nodes and outputs.
  void ImportGraph(const GraphProto &graph);

  // Imports ``function`` by replaying its body nodes and formal inputs/outputs.
  void ImportFunction(const FunctionProto &function);

  // Converts node attributes from proto form to builder form: GRAPH/GRAPHS
  // attributes become ``*_ref`` STRING/STRINGS attributes that reference nested
  // subgraph builders.
  utils::RepeatedProtoField<AttributeProto>
  ImportAttributes(const NodeProto &node,
                   const std::unordered_set<std::string> &excluded_inherited_names = {});

  // Materializes ``*_ref`` STRING/STRINGS attributes in ``node`` back to
  // GRAPH/GRAPHS attributes from the referenced nested builders.
  void MaterializeGraphReferences(NodeProto &node) const;

  // Returns ``true`` when ``name`` ends with ``"_ref"``.
  static bool HasGraphReferenceSuffix(const std::string &name);

  // Returns ``true`` when ``node`` carries a control-flow subgraph, either
  // inline (GRAPH / GRAPHS attribute) or through a builder ``*_ref`` reference.
  static bool NodeCarriesSubgraph(const NodeProto &node);

  // Runs the whole-graph compute analyses and writes their result into
  // ``graph`` (shapes, in-place / release-after / value-tag metadata and
  // per-node peak memory).
  void Finalize(GraphProto &graph);

  // Returns the nested builder named ``name`` in ``builders`` or nullptr.
  static GraphBuilder *FindNamedBuilder(const std::vector<std::unique_ptr<GraphBuilder>> &builders,
                                        const std::string &name);

  // Returns the nested builder named ``name`` in ``builders`` or throws a
  // BuilderError mentioning ``kind`` (e.g. "subgraph") when it is absent.
  static GraphBuilder &
  NamedBuilderOrThrow(const std::vector<std::unique_ptr<GraphBuilder>> &builders,
                      const std::string &name, const char *kind);

  std::string name_;
  std::string function_domain_;
  SchemaLookupFn schema_lookup_;
  // Lazily-built lookup table: op_type -> normalised domain -> schema history.
  std::unordered_map<std::string, std::unordered_map<std::string, std::vector<LightOpSchema>>>
      schema_table_;
  ComputeContext compute_;
  utils::RepeatedProtoField<ValueInfoProto> inputs_;
  utils::RepeatedProtoField<ValueInfoProto> outputs_;
  utils::RepeatedProtoField<NodeProto> nodes_;
  utils::RepeatedProtoField<TensorProto> initializers_;
  std::vector<std::unique_ptr<GraphBuilder>> local_functions_;
  std::vector<std::unique_ptr<GraphBuilder>> subgraphs_;
  std::unordered_set<std::string> names_;
  std::unordered_set<std::string> inherited_names_;
  std::unordered_map<std::string, int> opsets_;
  std::unordered_set<std::string> user_opsets_;
  Device device_ = Device::kUndefined;
  std::uint64_t auto_counter_ = 0;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
