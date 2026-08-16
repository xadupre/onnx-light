// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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

/// Parses a classic (string) tree node mode attribute value.
TreeNodeMode ParseTreeNodeMode(const std::string &mode);

/// Applies a comparison at an interior node.
/// Returns true if the sample should follow the 'true' branch.
bool ApplyTreeNodeMode(TreeNodeMode mode, double feature_value, double threshold);

/// Applies a comparison at a TreeEnsemble v5 interior node.
/// Returns true if the sample should follow the 'true' branch.
bool ApplyTreeNodeModeV5(TreeNodeModeV5 mode, double feature_value, double threshold);

/// Applies the aggregate function to accumulate leaf weights into per-target
/// accumulators.
///
/// ``agg``: "SUM" (default), "AVERAGE", "MIN", "MAX".
void AggregateTreeLeafWeight(std::vector<float> &accum, int64_t target_id, float weight,
                             const std::string &agg, std::vector<int64_t> &counts);

/// Finalizes the aggregation for one sample (divides by count for AVERAGE).
void FinalizeAggregation(std::vector<float> &accum, const std::vector<int64_t> &counts,
                         const std::string &agg);

/// Applies the post_transform in place to a single sample's scores held in the
/// contiguous range ``[scores, scores + count)``. Supports "NONE", "SOFTMAX",
/// "LOGISTIC", and "SOFTMAX_ZERO"; other values raise an error.
void ApplyPostTransform(float *scores, size_t count, const std::string &post_transform);

/// Applies the post_transform to the scores for a single sample. Supports
/// "NONE", "SOFTMAX", "LOGISTIC", and "SOFTMAX_ZERO"; other values raise an
/// error.
void ApplyPostTransform(std::vector<float> &scores, const std::string &post_transform);

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
    // Combine two int64 fields using the golden-ratio constant (Fibonacci hashing).
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
    const ::onnx_light::core::runtime::ParamStrings &nodes_modes,
    const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_missing) {
  const size_t n_nodes = nodes_treeids.size();
  EXT_ENFORCE_INVALID(nodes_nodeids.size() == n_nodes && nodes_featureids.size() == n_nodes &&
                          nodes_values.size() == n_nodes && nodes_modes.size() == n_nodes &&
                          nodes_truenodeids.size() == n_nodes &&
                          nodes_falsenodeids.size() == n_nodes,
                      "BuildClassicNodeMap: all nodes_* arrays must have the same length.");

  ClassicNodeMap map;
  map.reserve(n_nodes);
  for (size_t i = 0; i < n_nodes; ++i) {
    ClassicTreeNode node;
    node.feature_id = nodes_featureids[i];
    node.threshold = static_cast<double>(nodes_values[i]);
    node.mode = ParseTreeNodeMode(nodes_modes[i]);
    node.true_node_id = nodes_truenodeids[i];
    node.false_node_id = nodes_falsenodeids[i];
    node.missing_tracks_true = !nodes_missing.empty() && nodes_missing[i] != 0;
    map[{nodes_treeids[i], nodes_nodeids[i]}] = node;
  }
  return map;
}

/// Traverses a single tree for a single sample and returns the leaf node id
/// reached.
int64_t TraverseClassicTree(const ClassicNodeMap &node_map, int64_t tree_id, const double *x_row,
                            int64_t feature_count);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
