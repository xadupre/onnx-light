// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file graph_graph.h
 * @brief Read-only index over a :cpp:class:`core::builder::GraphBuilder`,
 *        the first building block of the C++ ``GraphBuilderPatternOptimization``.
 *
 * :cpp:class:`core::builder::GraphGraph` mirrors the ``_build`` method of the
 * Python ``GraphBuilderPatternOptimization``: it does not own the graph, it
 * owns an index over the builder nodes. A pattern optimizer rebuilds one
 * :cpp:class:`GraphGraph` after every iteration and queries it to reason about
 * the local structure of the graph without mutating the builder.
 *
 * Nodes are identified by their address, so the index stores
 * ``const NodeProto *`` and maps each to its position in
 * :cpp:func:`GraphBuilder::Nodes` (this mirrors ``make_idn`` / ``id(node)``
 * in Python). The index tracks:
 *
 *   - the producing node of every value (``predecessors``);
 *   - the consuming nodes of every value, deduplicated and in insertion order
 *     (``successors``);
 *   - the declared graph outputs;
 *   - the values captured by a nested subgraph from the enclosing scope, so a
 *     rewrite never deletes a producer a subgraph still relies on.
 *
 * On top of the structural index the class also exposes the read-only value
 * queries a pattern needs -- shapes, element types and constants -- drawing on
 * the builder's inferred information and constant analysis rather than
 * re-implementing them. Constant-folding results (added by later steps) are
 * cached by value name because a name is assigned once and never reused.
 */

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

using ::onnx_light::core::symbolic::SymTensor;
using ::onnx_light::core::symbolic::TensorType;

/**
 * Read-only index over the nodes of a :cpp:class:`GraphBuilder`.
 *
 * A :cpp:class:`GraphGraph` is a snapshot: it reflects the builder as it was
 * when the index was constructed. Mutating the builder afterwards invalidates
 * the stored node pointers, so callers rebuild the index after each rewrite.
 * The referenced :cpp:class:`GraphBuilder` must outlive the index.
 */
class GraphGraph {
public:
  /// Builds the index from ``builder``. The builder must outlive the index.
  explicit GraphGraph(const GraphBuilder &builder);

  // ── Structural queries ───────────────────────────────────────────────

  /// Returns the node producing ``name``, or ``nullptr`` when ``name`` is a
  /// graph input, an initializer or otherwise not produced by any node.
  const NodeProto *NodeBefore(const std::string &name) const;

  /// Returns the nodes consuming ``name`` (in insertion order, deduplicated).
  /// Returns a reference to a shared empty vector when ``name`` is unused.
  const std::vector<const NodeProto *> &NextNodes(const std::string &name) const;

  /// Returns the nodes producing the inputs of ``node`` (its immediate
  /// predecessors in the data-flow graph), deduplicated and in input order.
  /// Inputs that are graph inputs or initializers contribute no predecessor.
  std::vector<const NodeProto *> Predecessors(const NodeProto &node) const;

  /// Returns the nodes consuming the outputs of ``node`` (its immediate
  /// successors in the data-flow graph), deduplicated and in output order.
  std::vector<const NodeProto *> Successors(const NodeProto &node) const;

  /// Returns ``true`` when ``name`` is a declared graph output.
  bool IsOutput(const std::string &name) const;

  /// Returns ``true`` when ``name`` is consumed by a node, captured by a nested
  /// subgraph, or declared as a graph output.
  bool IsUsed(const std::string &name) const;

  /// Returns ``true`` when ``name`` is consumed by more than one node, captured
  /// by a nested subgraph, or declared as a graph output.
  bool IsUsedMoreThanOnce(const std::string &name) const;

  /// Returns ``true`` when ``name`` is captured by a nested subgraph from the
  /// enclosing scope.
  bool IsUsedBySubgraph(const std::string &name) const;

  /// Returns the position of ``node`` in :cpp:func:`GraphBuilder::Nodes`.
  /// Throws :cpp:class:`BuilderError` when ``node`` is not part of the index.
  std::size_t Position(const NodeProto &node) const;

  // ── Value queries ────────────────────────────────────────────────────

  /// Returns ``true`` when the shape of ``name`` has been inferred.
  bool HasShape(const std::string &name) const;

  /// Returns the inferred descriptor of ``name``. Throws when it is unknown.
  const SymTensor &GetShape(const std::string &name) const;

  /// Returns ``true`` when the element type of ``name`` is known.
  bool HasType(const std::string &name) const;

  /// Returns the element type of ``name``. Throws when it is unknown.
  TensorType GetType(const std::string &name) const;

  /// Returns ``true`` when ``name`` is a constant value (an initializer, a
  /// ``Constant`` output, or the output of a deterministic node whose inputs
  /// are all constant), as tracked by the builder's constant analysis.
  bool IsConstant(const std::string &name) const;

  /// Returns ``true`` when ``name`` is a constant scalar.
  ///
  /// A value is scalar when its shape is ``()`` or ``(1,)``. When ``broadcast``
  /// is ``true`` a shape whose every dimension is ``1`` (e.g. ``(1, 1)``) also
  /// qualifies. This overload does not compare the stored value.
  bool IsConstantScalar(const std::string &name, bool broadcast = false) const;

  /// Returns ``true`` when ``name`` is a constant scalar equal to ``value``.
  ///
  /// The scalar-shape rules of the other overload apply; in addition the stored
  /// value must be readable and compare equal to ``value``.
  bool IsConstantScalar(const std::string &name, double value, bool broadcast) const;

  /// Returns the tensor value of the constant ``name``, or ``nullptr`` when the
  /// value is not materialised as a :cpp:class:`TensorProto` (for example a
  /// ``Constant`` node using ``value_int`` / ``value_float``, or a constant
  /// that has not been folded). Registered computed constants take precedence
  /// over the builder's initializers and ``Constant`` node attributes.
  const TensorProto *GetComputedConstant(const std::string &name) const;

  /// Records a folded constant value for ``name`` so later queries can read it
  /// through :cpp:func:`GetComputedConstant`. A name is assigned once and never
  /// reused, so the cached value stays valid for the lifetime of the index.
  void SetComputedConstant(const std::string &name, TensorProto value);

private:
  // Returns the concrete dims of the constant ``name`` (empty vector for a
  // scalar) into ``dims``, or ``false`` when the shape cannot be determined.
  bool ConstantShape(const std::string &name, std::vector<int64_t> &dims) const;

  // Reads the scalar value of the constant ``name`` as a double into ``out``,
  // or returns ``false`` when it cannot be read.
  bool ConstantScalarValue(const std::string &name, double &out) const;

  const GraphBuilder &builder_;
  // Value name -> producing node (mirrors Python ``predecessors_``).
  std::unordered_map<std::string, const NodeProto *> predecessors_;
  // Value name -> consuming nodes, deduplicated, in insertion order.
  std::unordered_map<std::string, std::vector<const NodeProto *>> successors_;
  // Node address -> position in GraphBuilder::Nodes.
  std::unordered_map<const NodeProto *, std::size_t> positions_;
  // Declared graph output names.
  std::unordered_set<std::string> output_names_;
  // Values captured by a nested subgraph from the enclosing scope.
  std::unordered_set<std::string> subgraph_captured_;
  // Initializer name -> initializer tensor.
  std::unordered_map<std::string, const TensorProto *> initializers_;
  // Folded-constant cache keyed by value name.
  std::unordered_map<std::string, TensorProto> computed_constants_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
