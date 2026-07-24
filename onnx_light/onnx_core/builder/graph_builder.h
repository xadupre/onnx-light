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
 * Inputs, outputs and nodes are kept in plain vectors while initializers and
 * the nested local functions / subgraphs (each of which is itself a
 * :cpp:class:`GraphBuilder`) live in insertion-ordered maps. A proto is only
 * materialised on demand by :cpp:func:`BuildGraph` and the finalizers.
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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx_core/builder/ordered_map.h"
#include "onnx_core/compute/compute_context.h"
#include "onnx_core/light_op_schema/light_op_schema.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace builder {

using ::onnx_light::core::compute::ComputeContext;
using ::onnx_light::core::schema::LightOpSchema;
using ::onnx_light::core::shapes::ShapesContext;
using ::onnx_light::core::symbolic::Device;
using ::onnx_light::core::symbolic::SymShape;
using ::onnx_light::core::symbolic::SymTensor;
using ::onnx_light::core::symbolic::TensorType;

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

  /// Read-only access to the insertion-ordered ``name -> initializer`` map.
  const OrderedMap<TensorProto> &Initializers() const noexcept { return initializers_; }

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
  const std::vector<ValueInfoProto> &Inputs() const noexcept { return inputs_; }

  /// Read-only access to the declared graph outputs (in declaration order).
  const std::vector<ValueInfoProto> &Outputs() const noexcept { return outputs_; }

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
                                    const std::vector<AttributeProto> &attributes = {});

  /// Read-only access to the accumulated nodes (in insertion order).
  const std::vector<NodeProto> &Nodes() const noexcept { return nodes_; }

  // ── Local functions / subgraphs ──────────────────────────────────────

  /// Creates and returns a nested builder for a local function named ``name``.
  /// The nested builder is stored in this builder's insertion-ordered local
  /// function map; local functions are emitted into the produced
  /// :cpp:class:`ModelProto`. Throws when ``name`` is already used.
  GraphBuilder &MakeLocalFunction(const std::string &name, const std::string &domain = "");

  /// Returns ``true`` when a local function named ``name`` exists.
  bool HasLocalFunction(const std::string &name) const { return local_functions_.Contains(name); }

  /// Returns the nested local-function builder named ``name``. Throws when it
  /// does not exist.
  GraphBuilder &LocalFunction(const std::string &name) { return *local_functions_.At(name); }
  const GraphBuilder &LocalFunction(const std::string &name) const {
    return *local_functions_.At(name);
  }

  /// Read-only access to the insertion-ordered local function map.
  const OrderedMap<std::unique_ptr<GraphBuilder>> &LocalFunctions() const noexcept {
    return local_functions_;
  }

  /// Creates and returns a nested builder for a subgraph named ``name`` (used
  /// as the body of a control-flow node such as :onnx:`If`, :onnx:`Loop` or
  /// :onnx:`Scan`). The nested builder is stored in this builder's
  /// insertion-ordered subgraph map. Throws when ``name`` is already used.
  GraphBuilder &MakeSubgraph(const std::string &name);

  /// Returns ``true`` when a subgraph named ``name`` exists.
  bool HasSubgraph(const std::string &name) const { return subgraphs_.Contains(name); }

  /// Returns the nested subgraph builder named ``name``. Throws when it does
  /// not exist.
  GraphBuilder &Subgraph(const std::string &name) { return *subgraphs_.At(name); }
  const GraphBuilder &Subgraph(const std::string &name) const { return *subgraphs_.At(name); }

  /// Read-only access to the insertion-ordered subgraph map.
  const OrderedMap<std::unique_ptr<GraphBuilder>> &Subgraphs() const noexcept { return subgraphs_; }

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
  // Resolves and records the opset version to use for a node of ``domain``,
  // given the domain-filtered schema history ``schemas``.
  int ResolveNodeOpset(const std::string &domain,
                       const std::vector<const LightOpSchema *> &schemas);

  // Returns ``true`` when a shape function is registered for ``node``.
  bool ShapeFunctionAvailable(const NodeProto &node) const;

  // Seeds the owned ShapesContext with the descriptor of ``name``.
  void SeedShape(const std::string &name, SymTensor tensor);

  // Runs the whole-graph compute analyses and writes their result into
  // ``graph`` (shapes, in-place / release-after / value-tag metadata and
  // per-node peak memory).
  void Finalize(GraphProto &graph);

  std::string name_;
  SchemaLookupFn schema_lookup_;
  ComputeContext compute_;
  std::vector<ValueInfoProto> inputs_;
  std::vector<ValueInfoProto> outputs_;
  std::vector<NodeProto> nodes_;
  OrderedMap<TensorProto> initializers_;
  OrderedMap<std::unique_ptr<GraphBuilder>> local_functions_;
  OrderedMap<std::unique_ptr<GraphBuilder>> subgraphs_;
  std::unordered_set<std::string> names_;
  std::unordered_map<std::string, int> opsets_;
  std::unordered_set<std::string> user_opsets_;
  Device device_ = Device::kUndefined;
  std::uint64_t auto_counter_ = 0;
};

} // namespace builder
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
