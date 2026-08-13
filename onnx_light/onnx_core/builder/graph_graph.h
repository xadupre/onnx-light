// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file graph_graph.h
 * @brief Graph index and pattern-rewrite driver over a
 *        :cpp:class:`core::builder::GraphBuilder`.
 *
 * :cpp:class:`core::builder::GraphGraph` owns an index over the builder nodes
 * and applies graph-rewrite patterns directly to that builder. It rebuilds its
 * index after every optimization iteration.
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
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

using ::onnx_light::core::symbolic::SymTensor;
using ::onnx_light::core::symbolic::TensorType;

/**
 * Index and rewrite driver over the nodes of a :cpp:class:`GraphBuilder`.
 *
 * The referenced :cpp:class:`GraphBuilder` must outlive this object. Mutating
 * the builder outside :cpp:func:`Optimize` invalidates the index.
 */
class GraphGraph {
public:
  /// Predicate that protects matching nodes from removal.
  using DoNotRemovePredicate = std::function<bool(const NodeProto &)>;

  /// Builds the index from ``builder``. The builder must outlive the index.
  explicit GraphGraph(GraphBuilder &builder);

  /// Builds the index and uses the supplied patterns in their given order.
  GraphGraph(GraphBuilder &builder, std::vector<std::unique_ptr<PatternOptimization>> patterns,
             DoNotRemovePredicate do_not_remove = {});

  /// Returns the builder being indexed and optimized.
  GraphBuilder &Builder() noexcept { return builder_; }

  /**
   * Applies patterns and cleanup passes until convergence.
   *
   * Patterns are considered in ascending priority order. A negative
   * ``max_iter`` selects ``max(node_count, 10) * priority_count``.
   *
   * Returns:
   *   Self-contained records of the applied rewrites, in application order.
   */
  std::vector<LocalRewriting> Optimize(int max_iter = -1);

  /// Returns the patterns shared by this graph optimizer and its rewrites.
  const std::vector<std::shared_ptr<PatternOptimization>> &Patterns() const noexcept {
    return patterns_;
  }

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
  friend GraphProto Replay(const ModelProto &model, const std::vector<LocalRewriting> &rewrites,
                           GraphBuilder::SchemaLookupFn schema_lookup);

  void Rebuild();
  void RebuildSuccessors();
  std::size_t Cleanup();
  void ApplyRewritingBatch(const std::vector<LocalRewriting> &rewrites, std::size_t begin,
                           std::size_t end);

  // Returns the concrete dims of the constant ``name`` (empty vector for a
  // scalar) into ``dims``, or ``false`` when the shape cannot be determined.
  bool ConstantShape(const std::string &name, std::vector<int64_t> &dims) const;

  // Reads the scalar value of the constant ``name`` as a double into ``out``,
  // or returns ``false`` when it cannot be read.
  bool ConstantScalarValue(const std::string &name, double &out) const;

  GraphBuilder &builder_;
  std::vector<std::shared_ptr<PatternOptimization>> patterns_;
  DoNotRemovePredicate do_not_remove_;
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

/**
 * Reconstructs an optimized graph by replaying captured rewrites.
 *
 * Rewrites must be ordered as returned by :cpp:func:`GraphGraph::Optimize`.
 * Cleanup passes run at the same iteration boundaries as the live optimizer.
 */
GraphProto Replay(const ModelProto &model, const std::vector<LocalRewriting> &rewrites,
                  GraphBuilder::SchemaLookupFn schema_lookup = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
