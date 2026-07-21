// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"
#include "onnx_proto/onnx.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @file execution_plan.h
 * @brief Precomputed per-graph release schedule (:cpp:class:`ExecutionPlan`)
 *        used by the node dispatcher to free intermediate values as soon
 *        as they are no longer referenced.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

// Forward declaration so ExecutionPlan::ReleaseAfter can reference
// RuntimeContext by reference without a full definition at this point.
class RuntimeContext;

/**
 * Precomputed per-graph release schedule used by
 * :cpp:func:`RunGraph` / :cpp:func:`RunFunction` /
 * :cpp:func:`RunNodes` when
 * :cpp:func:`RuntimeContext::release_intermediates` is enabled.
 *
 * An :cpp:class:`ExecutionPlan` captures two complementary pieces of
 * information for a given node sequence:
 *
 *  * ``keep`` — the *structural* set of names that must never be
 *    released by the per-node release loop. For a :cpp:class:`GraphProto`
 *    this is the union of declared inputs, initializers, and declared
 *    outputs; for a :cpp:class:`FunctionProto` it is the union of
 *    declared inputs and outputs.
 *  * ``releasable[i]`` — the list of names whose last reference (per
 *    :cpp:func:`RuntimeContext::CollectNodeInputs`) falls at node ``i``
 *    and that are not in ``keep`` — i.e. the intermediates that may be
 *    removed from the tensor / sequence map right after node ``i``
 *    finishes.
 *
 * The analysis depends only on the graph topology and not on any
 * runtime value, so a single plan can be reused across every
 * invocation of the same model. :cpp:func:`RuntimeContext::GetExecutionPlan`
 * builds and caches one plan per graph / function for that reason.
 */
class ExecutionPlan {
public:
  ExecutionPlan() = default;

  /// Builds the plan for ``graph``. ``keep`` is seeded with the graph's
  /// declared inputs, initializers and declared outputs; ``releasable``
  /// is computed by :cpp:func:`RuntimeContext::ComputeReleasableInputs`.
  explicit ExecutionPlan(const GraphProto &graph);

  /// Builds the plan for ``func``. ``keep`` is seeded with the
  /// function's declared inputs and outputs.
  explicit ExecutionPlan(const FunctionProto &func);

  /// Builds the plan for a free-standing node range. ``keep`` is the
  /// user-supplied set of names that must never be released (typically
  /// the names already populated in the runtime context at run start
  /// plus any graph / function outputs).
  ExecutionPlan(const utils::RepeatedProtoField<NodeProto> &nodes,
                std::unordered_set<std::string> keep);

  /// Structural set of names that must never be released. See the
  /// class-level documentation for the exact contents.
  const std::unordered_set<std::string> &keep() const noexcept { return keep_; }

  /// For each node ``i`` in the underlying node range, the list of
  /// names whose last reference falls at ``i`` and that are not in
  /// :cpp:func:`keep`.
  const std::vector<std::vector<std::string>> &releasable() const noexcept { return releasable_; }

  /// Number of nodes covered by this plan (``releasable().size()``).
  size_t num_nodes() const noexcept { return releasable_.size(); }

  /// Releases from ``rt`` every name in the ``releasable()`` slot
  /// associated with ``node``. ``node`` must be one of the
  /// :cpp:class:`NodeProto` instances the plan was built from
  /// (lookup is by address); if it is not, this is a no-op. Each
  /// removal is performed on both the tensor map and the sequence
  /// map: :cpp:func:`RuntimeContext::Remove` is a no-op if the name
  /// is absent and emits a :cpp:enumerator:`RuntimeEventAction::kRemove`
  /// event when event logging is on; sequence removals do not emit
  /// events (sequence values live outside the tensor event stream).
  void ReleaseAfter(const NodeProto &node, RuntimeContext &rt) const;

private:
  std::unordered_set<std::string> keep_;
  std::vector<std::vector<std::string>> releasable_;
  std::unordered_map<const NodeProto *, size_t> node_index_;
};

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
