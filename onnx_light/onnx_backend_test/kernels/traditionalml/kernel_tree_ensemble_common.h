// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/simple_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

/// Node modes for TreeEnsembleRegressor / TreeEnsembleClassifier (string form).
enum class TreeNodeMode {
  kBranchLeq,
  kBranchLt,
  kBranchGte,
  kBranchGt,
  kBranchEq,
  kBranchNeq,
  kLeaf,
};

/// Node modes for TreeEnsemble v5 (integer form in nodes_modes uint8 tensor).
enum class TreeNodeModeV5 : uint8_t {
  kBranchLeq = 0,
  kBranchLt = 1,
  kBranchGte = 2,
  kBranchGt = 3,
  kBranchEq = 4,
  kBranchNeq = 5,
  kBranchMember = 6,
};

inline TreeNodeMode ParseTreeNodeMode(const std::string &mode) {
  if (mode == "BRANCH_LEQ")
    return TreeNodeMode::kBranchLeq;
  if (mode == "BRANCH_LT")
    return TreeNodeMode::kBranchLt;
  if (mode == "BRANCH_GTE")
    return TreeNodeMode::kBranchGte;
  if (mode == "BRANCH_GT")
    return TreeNodeMode::kBranchGt;
  if (mode == "BRANCH_EQ")
    return TreeNodeMode::kBranchEq;
  if (mode == "BRANCH_NEQ")
    return TreeNodeMode::kBranchNeq;
  if (mode == "LEAF")
    return TreeNodeMode::kLeaf;
  throw std::invalid_argument("Unsupported tree node mode: " + mode);
}

/// Applies a comparison at an interior node.
/// Returns true if the sample should follow the 'true' branch.
inline bool ApplyTreeNodeMode(TreeNodeMode mode, double feature_value, double threshold) {
  switch (mode) {
  case TreeNodeMode::kBranchLeq:
    return feature_value <= threshold;
  case TreeNodeMode::kBranchLt:
    return feature_value < threshold;
  case TreeNodeMode::kBranchGte:
    return feature_value >= threshold;
  case TreeNodeMode::kBranchGt:
    return feature_value > threshold;
  case TreeNodeMode::kBranchEq:
    return feature_value == threshold;
  case TreeNodeMode::kBranchNeq:
    return feature_value != threshold;
  case TreeNodeMode::kLeaf:
    throw std::invalid_argument("ApplyTreeNodeMode: LEAF nodes should not be compared.");
  }
  throw std::invalid_argument("ApplyTreeNodeMode: unknown mode.");
}

inline bool ApplyTreeNodeModeV5(TreeNodeModeV5 mode, double feature_value, double threshold) {
  switch (mode) {
  case TreeNodeModeV5::kBranchLeq:
    return feature_value <= threshold;
  case TreeNodeModeV5::kBranchLt:
    return feature_value < threshold;
  case TreeNodeModeV5::kBranchGte:
    return feature_value >= threshold;
  case TreeNodeModeV5::kBranchGt:
    return feature_value > threshold;
  case TreeNodeModeV5::kBranchEq:
    return feature_value == threshold;
  case TreeNodeModeV5::kBranchNeq:
    return feature_value != threshold;
  case TreeNodeModeV5::kBranchMember:
    throw std::invalid_argument(
        "ApplyTreeNodeModeV5: BRANCH_MEMBER is not supported in this kernel.");
  }
  throw std::invalid_argument("ApplyTreeNodeModeV5: unknown mode.");
}

/// Applies the aggregate function to accumulate leaf weights into per-target
/// accumulators.
///
/// ``agg``: "SUM" (default), "AVERAGE", "MIN", "MAX".
inline void AggregateTreeLeafWeight(std::vector<float> &accum, int64_t target_id, float weight,
                                    const std::string &agg, std::vector<int64_t> &counts) {
  EXT_ENFORCE_INVALID(target_id >= 0 && target_id < static_cast<int64_t>(accum.size()),
                      "AggregateTreeLeafWeight: target_id out of range.");
  if (agg == "SUM" || agg == "AVERAGE") {
    accum[static_cast<size_t>(target_id)] += weight;
    counts[static_cast<size_t>(target_id)]++;
  } else if (agg == "MIN") {
    if (counts[static_cast<size_t>(target_id)] == 0) {
      accum[static_cast<size_t>(target_id)] = weight;
    } else {
      accum[static_cast<size_t>(target_id)] =
          std::min(accum[static_cast<size_t>(target_id)], weight);
    }
    counts[static_cast<size_t>(target_id)]++;
  } else if (agg == "MAX") {
    if (counts[static_cast<size_t>(target_id)] == 0) {
      accum[static_cast<size_t>(target_id)] = weight;
    } else {
      accum[static_cast<size_t>(target_id)] =
          std::max(accum[static_cast<size_t>(target_id)], weight);
    }
    counts[static_cast<size_t>(target_id)]++;
  } else {
    throw std::invalid_argument("AggregateTreeLeafWeight: unsupported aggregate_function: " + agg);
  }
}

/// Finalizes the aggregation for one sample (divides by count for AVERAGE).
inline void FinalizeAggregation(std::vector<float> &accum, const std::vector<int64_t> &counts,
                                const std::string &agg) {
  if (agg != "AVERAGE") {
    return;
  }
  for (size_t i = 0; i < accum.size(); ++i) {
    if (counts[i] > 0) {
      accum[i] /= static_cast<float>(counts[i]);
    }
  }
}

/// Applies the post_transform to the scores for a single sample.
/// Only "NONE" is fully supported; other transforms raise an error.
inline void ApplyPostTransform(std::vector<float> &scores, const std::string &post_transform) {
  if (post_transform == "NONE") {
    return;
  }
  if (post_transform == "SOFTMAX") {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float &s : scores) {
      s = std::exp(s - max_val);
      sum += s;
    }
    for (float &s : scores) {
      s /= sum;
    }
    return;
  }
  if (post_transform == "LOGISTIC") {
    for (float &s : scores) {
      s = 1.0f / (1.0f + std::exp(-s));
    }
    return;
  }
  if (post_transform == "SOFTMAX_ZERO") {
    float max_val = *std::max_element(scores.begin(), scores.end());
    if (max_val == 0.0f) {
      return;
    }
    float sum = 0.0f;
    for (float &s : scores) {
      s = std::exp(s - max_val);
      sum += s;
    }
    for (float &s : scores) {
      s /= sum;
    }
    return;
  }
  throw std::invalid_argument("ApplyPostTransform: unsupported post_transform: " + post_transform);
}

/// Classic tree node record used by TreeEnsembleRegressor and
/// TreeEnsembleClassifier.
struct ClassicTreeNode {
  int64_t feature_id{0};
  double threshold{0.0};
  TreeNodeMode mode{TreeNodeMode::kLeaf};
  int64_t true_node_id{0};
  int64_t false_node_id{0};
  bool missing_tracks_true{false};
};

/// Key for indexing classic tree nodes: (tree_id, node_id).
struct TreeNodeKey {
  int64_t tree_id;
  int64_t node_id;
  bool operator==(const TreeNodeKey &o) const noexcept {
    return tree_id == o.tree_id && node_id == o.node_id;
  }
};

struct TreeNodeKeyHash {
  size_t operator()(const TreeNodeKey &k) const noexcept {
    // FNV-like mix
    size_t h = static_cast<size_t>(k.tree_id);
    h ^= static_cast<size_t>(k.node_id) * 0x9e3779b97f4a7c15ULL;
    return h;
  }
};

using ClassicNodeMap = std::unordered_map<TreeNodeKey, ClassicTreeNode, TreeNodeKeyHash>;

/// One leaf contribution (classic encoding): maps (tree_id, node_id) to a
/// list of (target_id, weight) pairs.
struct LeafEntry {
  int64_t target_id;
  float weight;
};

using ClassicLeafMap = std::unordered_map<TreeNodeKey, std::vector<LeafEntry>, TreeNodeKeyHash>;

/// Builds the classic node map from the parallel nodes_* attribute arrays.
template <typename ThresholdT>
inline ClassicNodeMap BuildClassicNodeMap(
    const std::vector<int64_t> &nodes_treeids, const std::vector<int64_t> &nodes_nodeids,
    const std::vector<int64_t> &nodes_featureids, const std::vector<ThresholdT> &nodes_values,
    const std::vector<std::string> &nodes_modes, const std::vector<int64_t> &nodes_truenodeids,
    const std::vector<int64_t> &nodes_falsenodeids, const std::vector<int64_t> &nodes_missing) {
  const size_t n_nodes = nodes_treeids.size();
  EXT_ENFORCE_INVALID(nodes_nodeids.size() == n_nodes && nodes_featureids.size() == n_nodes &&
                          nodes_values.size() == n_nodes && nodes_modes.size() == n_nodes &&
                          nodes_truenodeids.size() == n_nodes &&
                          nodes_falsenodeids.size() == n_nodes,
                      "BuildClassicNodeMap: all nodes_* arrays must have the same length.");

  ClassicNodeMap map;
  map.reserve(n_nodes);
  for (size_t i = 0; i < n_nodes; ++i) {
    ClassicTreeNode nd;
    nd.feature_id = nodes_featureids[i];
    nd.threshold = static_cast<double>(nodes_values[i]);
    nd.mode = ParseTreeNodeMode(nodes_modes[i]);
    nd.true_node_id = nodes_truenodeids[i];
    nd.false_node_id = nodes_falsenodeids[i];
    nd.missing_tracks_true = !nodes_missing.empty() && nodes_missing[i] != 0;
    map[{nodes_treeids[i], nodes_nodeids[i]}] = nd;
  }
  return map;
}

/// Traverses a single tree for a single sample and returns the leaf node id
/// reached.
inline int64_t TraverseClassicTree(const ClassicNodeMap &node_map, int64_t tree_id,
                                   const double *x_row, int64_t feature_count) {
  int64_t cur_node_id = 0;
  for (;;) {
    auto it = node_map.find({tree_id, cur_node_id});
    EXT_ENFORCE_INVALID(it != node_map.end(), "TraverseClassicTree: node not found in tree.");
    const ClassicTreeNode &nd = it->second;
    if (nd.mode == TreeNodeMode::kLeaf) {
      return cur_node_id;
    }
    EXT_ENFORCE_INVALID(nd.feature_id >= 0 && nd.feature_id < feature_count,
                        "TraverseClassicTree: feature_id out of range.");
    const double feature_value = x_row[nd.feature_id];
    bool go_true;
    if (std::isnan(feature_value)) {
      go_true = nd.missing_tracks_true;
    } else {
      go_true = ApplyTreeNodeMode(nd.mode, feature_value, nd.threshold);
    }
    cur_node_id = go_true ? nd.true_node_id : nd.false_node_id;
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
