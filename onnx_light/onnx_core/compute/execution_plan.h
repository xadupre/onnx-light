// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/execute_action.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
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
 * Precomputed per-graph release schedule used by :cpp:class:`RuntimeSession`
 * when :cpp:func:`RuntimeContext::release_intermediates` is enabled.
 *
 * An :cpp:class:`ExecutionPlan` captures, for a given node sequence:
 *
 *  * ``keep`` — the *structural* set of names that must never be
 *    released by the per-node release loop. For a :cpp:class:`GraphProto`
 *    this is the union of declared inputs, initializers, and declared
 *    outputs; for a :cpp:class:`FunctionProto` it is the union of
 *    declared inputs and outputs.
 *  * ``actions`` — the ordered list of :cpp:class:`ExecuteAction` steps
 *    (lock / unlock, allocate / delete buffer, create / delete shape,
 *    allocate / delete temporary buffer, execute node) derived from the
 *    in-place / lifetime / peak-memory metadata written to each node by
 *    :cpp:class:`annotations::ComputeContext` and
 *    :cpp:func:`annotations::WritePeakMemoryToMetadata`.
 *
 * The memory-management schedule is entirely metadata-driven: the
 * :cpp:class:`ComputeContext` is responsible for annotating each node
 * with the in-place reuse, release and last-use information, and
 * :cpp:func:`BuildActions` consumes it. When a node range carries those
 * annotations, :cpp:func:`BuildActions` also *validates* that the
 * metadata is complete (every intermediate result is released, every
 * input / initializer is unlocked at its last use, every released shape
 * was created) and throws otherwise.
 *
 * The analysis depends only on the graph topology / metadata and not on
 * any runtime value, so a single plan can be reused across every
 * invocation of the same model. :cpp:func:`RuntimeContext::GetExecutionPlan`
 * builds and caches one plan per graph / function for that reason.
 */
class ExecutionPlan {
public:
  /// Builds an empty plan. ``allocator`` is referenced by every allocation
  /// and deallocation action the plan schedules (may be ``nullptr``).
  explicit ExecutionPlan(RawBufferAllocator *allocator = nullptr);

  /// Builds the plan for ``graph``. ``keep`` is seeded with the graph's
  /// declared inputs, initializers and declared outputs. ``allocator`` is
  /// referenced by every allocation / deallocation action.
  explicit ExecutionPlan(const GraphProto &graph, RawBufferAllocator *allocator = nullptr);

  /// Builds the plan for ``func``. ``keep`` is seeded with the
  /// function's declared inputs and outputs. ``allocator`` is referenced by
  /// every allocation / deallocation action.
  explicit ExecutionPlan(const FunctionProto &func, RawBufferAllocator *allocator = nullptr);

  /// Builds the plan for a free-standing node range. ``keep`` is the
  /// user-supplied set of names that must never be released (typically
  /// the names already populated in the runtime context at run start
  /// plus any graph / function outputs). ``allocator`` is referenced by
  /// every allocation / deallocation action.
  ExecutionPlan(const utils::RepeatedProtoField<NodeProto> &nodes,
                std::unordered_set<std::string> keep, RawBufferAllocator *allocator = nullptr);

  virtual ~ExecutionPlan() = default;

  /// Structural set of names that must never be released. See the
  /// class-level documentation for the exact contents.
  const std::unordered_set<std::string> &keep() const noexcept { return keep_; }

  /// Number of nodes covered by this plan.
  size_t num_nodes() const noexcept { return nodes_.size(); }

  /// Non-owning pointers to the nodes covered by this plan, in execution
  /// order. Entry ``i`` corresponds to the node an
  /// :cpp:enumerator:`ExecuteActionKind::kExecuteNode` action with
  /// :cpp:func:`ExecuteAction::node_index` ``i`` runs, so
  /// :cpp:class:`RuntimeSession` can recover the node list from the plan
  /// alone. The pointers reference the caller-owned graph / function and are
  /// valid only while it (and this plan) outlive the session.
  const std::vector<const NodeProto *> &nodes() const noexcept { return nodes_; }

  /// Releases from ``rt`` every intermediate whose last use falls at
  /// ``node``. ``node`` must be one of the :cpp:class:`NodeProto` instances
  /// the plan was built from (lookup is by address); if it is not, this is a
  /// no-op. The names to release are the ones carried by the
  /// :cpp:enumerator:`ExecuteActionKind::kDeleteBuffer` /
  /// :cpp:enumerator:`ExecuteActionKind::kDeleteShape` actions
  /// :cpp:func:`BuildActions` scheduled for ``node`` (their
  /// :cpp:func:`ExecuteAction::node_index`), so the release schedule lives
  /// entirely in :cpp:func:`actions`. Each removal is performed on both the
  /// tensor map and the sequence map: :cpp:func:`RuntimeContext::Remove` is a
  /// no-op if the name is absent and emits a
  /// :cpp:enumerator:`RuntimeEventAction::kRemove` event when event logging is
  /// on; sequence removals do not emit events (sequence values live outside the
  /// tensor event stream).
  ///
  /// This scans :cpp:func:`actions` for the node's delete actions, so it is
  /// linear in the plan size per call; the runtime does not use it on the hot
  /// path (:cpp:class:`RuntimeSession` replays the whole action list once
  /// instead).
  /// It is kept as a per-node convenience for callers that drive execution
  /// themselves.
  void ReleaseAfter(const NodeProto &node, RuntimeContext &rt) const;

  /// Ordered list of :cpp:class:`ExecuteAction` steps the runtime performs
  /// while executing the underlying node sequence. Built once at
  /// construction by :cpp:func:`BuildActions`.
  const std::vector<ExecuteAction> &actions() const noexcept { return actions_; }

  /// Returns the allocator referenced by every allocation / deallocation
  /// action, or ``nullptr`` when no allocator was supplied.
  RawBufferAllocator *allocator() const noexcept { return allocator_; }

protected:
  /// Populates :cpp:func:`actions` from the seeded members (``inputs_``,
  /// ``initializers_``, ``outputs_``, ``nodes_``) and the in-place / lifetime
  /// annotations carried by each node's ``metadata_props`` (written by
  /// :cpp:class:`annotations::ComputeContext`):
  /// :cpp:var:`annotations::kInPlaceReuseMetadataKey`,
  /// :cpp:var:`annotations::kReleaseAfterMetadataKey`,
  /// :cpp:var:`annotations::kNotUsedAfterMetadataKey` and
  /// :cpp:var:`annotations::kReleaseAfterShapeTagMetadataKey`. Inputs and
  /// initializers are locked on first use and unlocked on their last use
  /// (:cpp:var:`annotations::kNotUsedAfterMetadataKey`); each output is either
  /// allocated as a result (or reused in place per the in-place annotation) or
  /// created as a shape when value-tagged ``"shape"``, and freed on its last
  /// use. When at least one node carries
  /// :cpp:var:`annotations::kReleaseAfterMetadataKey`, that metadata drives the
  /// :cpp:enumerator:`ExecuteActionKind::kDeleteBuffer` /
  /// :cpp:enumerator:`ExecuteActionKind::kDeleteShape` schedule; otherwise the
  /// releases are derived from graph topology (each intermediate is freed after
  /// its last use, excluding :cpp:func:`keep` names). When a node
  /// carries a peak-memory estimate
  /// (:cpp:var:`annotations::kNodePeakMemoryMetadataKey`, written by
  /// :cpp:func:`annotations::WritePeakMemoryToMetadata`), a temporary buffer of
  /// that size is allocated right before the node runs and deleted right after.
  ///
  /// When the node range carries explicit lock-lifetime metadata
  /// (:cpp:var:`annotations::kNotUsedAfterMetadataKey`), that metadata is
  /// treated as the single source of truth and its completeness is enforced: an
  /// exception is thrown when an intermediate result is never released, when an
  /// input / initializer reaching its last use is never unlocked, or when a
  /// released shape was never created. When the node range carries no lifetime
  /// metadata (e.g. a model executed without first running the in-place reuse
  /// pass, or annotated only for memory profiling), the plan is built
  /// best-effort and no completeness check is performed.
  ///
  /// Every constructor calls this once, after seeding, so derived plans can
  /// override the action schedule. Overrides run against the base-class
  /// members only, since virtual dispatch during construction resolves to
  /// :cpp:class:`ExecutionPlan`.
  virtual void BuildActions();

private:
  RawBufferAllocator *allocator_ = nullptr;
  std::unordered_set<std::string> keep_;
  std::unordered_map<const NodeProto *, size_t> node_index_;
  /// Declared inputs (in order) used to schedule lock / unlock actions.
  std::vector<std::string> inputs_;
  /// Declared initializers (in order) used to schedule lock / unlock actions.
  std::vector<std::string> initializers_;
  /// Declared outputs used to distinguish kept results from intermediates.
  std::vector<std::string> outputs_;
  /// Nodes (in order) whose outputs drive allocation / shape actions.
  std::vector<const NodeProto *> nodes_;
  std::vector<ExecuteAction> actions_;
};

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
