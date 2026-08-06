// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx_proto/onnx.h"

/**
 * @file result_lifetime.h
 * @brief Standalone value-lifetime analysis shared by
 *        :cpp:func:`ComputeContext::ComputeInPlaceReuseGraph`.
 *
 * This header isolates the purely name-based bookkeeping that decides, for
 * every node of a graph, which referenced values reach their last use at
 * that node (:cpp:var:`ResultLifetimeNodeInfo::release_after` /
 * :cpp:var:`ResultLifetimeNodeInfo::not_used_after`). It does not depend on
 * shape inference or byte sizes: those live in ``compute_context.cc`` and
 * combine this lifetime information with the shape-driven in-place reuse
 * matching.
 */

namespace ONNX_LIGHT_NAMESPACE::core::compute {

/// Holds per-node value-lifetime bookkeeping from :cpp:func:`ComputeResultLifetimeInfo`.
struct ResultLifetimeNodeInfo {
  /// Per-node list of top-level intermediates that reach their last use at
  /// that node and are therefore releasable after it runs (excludes declared
  /// graph inputs, initializers and outputs).
  std::vector<std::string> release_after;

  /// Per-node list of declared graph inputs / initializers that reach their
  /// last use at that node. Not released by the runtime (their lifetime is
  /// owned by the caller / model) but still exposed as "last read" info.
  std::vector<std::string> not_used_after;
};

/**
 * Result of :cpp:func:`ComputeResultLifetimeInfo`: a per-node list of
 * :cpp:struct:`ResultLifetimeNodeInfo` plus the producer / lifetime maps and
 * name sets it was derived from, so callers can reuse them without
 * recomputing.
 */
class ResultLifetimeInfo {
public:
  using value_type = ResultLifetimeNodeInfo;
  using storage_type = std::vector<value_type>;
  using iterator = storage_type::iterator;
  using const_iterator = storage_type::const_iterator;

  std::size_t size() const { return nodes_.size(); }

  bool empty() const { return nodes_.empty(); }

  void resize(std::size_t count) { nodes_.resize(count); }

  value_type &operator[](std::size_t index) { return nodes_[index]; }

  const value_type &operator[](std::size_t index) const { return nodes_[index]; }

  iterator begin() { return nodes_.begin(); }

  iterator end() { return nodes_.end(); }

  const_iterator begin() const { return nodes_.begin(); }

  const_iterator end() const { return nodes_.end(); }

  const_iterator cbegin() const { return nodes_.cbegin(); }

  const_iterator cend() const { return nodes_.cend(); }

  std::vector<std::vector<std::string>> MoveReleaseAfter() {
    std::vector<std::vector<std::string>> release_after(nodes_.size());
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      release_after[i].swap(nodes_[i].release_after);
    }
    return release_after;
  }

  std::vector<std::vector<std::string>> MoveNotUsedAfter() {
    std::vector<std::vector<std::string>> not_used_after(nodes_.size());
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      not_used_after[i].swap(nodes_[i].not_used_after);
    }
    return not_used_after;
  }

  /// Producer node index for every top-level intermediate (``-1`` marks a
  /// declared graph input made available before the first node when
  /// ``allow_input_overwrite`` is set).
  std::unordered_map<std::string, int> producer;

  /// Index of the last node that references each name, directly or through
  /// a subgraph capture.
  std::unordered_map<std::string, int> last_use;

  /// Names whose buffers must never be reused in place: initializers and
  /// declared graph outputs are always kept; declared graph inputs are kept
  /// unless ``allow_input_overwrite`` was requested.
  std::unordered_set<std::string> keep;

  std::unordered_set<std::string> graph_inputs;
  std::unordered_set<std::string> graph_initializers;
  std::unordered_set<std::string> graph_outputs;

private:
  storage_type nodes_;
};

/**
 * Computes the per-node value-lifetime information for ``graph``: which
 * values are read for the last time at each node, and which of those are
 * releasable top-level intermediates versus caller-owned inputs /
 * initializers.
 *
 * @param graph  Graph whose nodes are analysed, in topological order.
 * @param allow_input_overwrite  When true, declared graph inputs are treated
 *                                as available before the first node (producer
 *                                index ``-1``) so they can be reused once
 *                                they reach their last use; when false they
 *                                are always kept alive.
 */
ResultLifetimeInfo ComputeResultLifetimeInfo(const GraphProto &graph, bool allow_input_overwrite);

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
