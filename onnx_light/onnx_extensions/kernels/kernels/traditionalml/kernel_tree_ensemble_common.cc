// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_tree_ensemble_common.h"

#include <algorithm>
#include <cmath>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

TreeNodeMode ParseTreeNodeMode(const std::string &mode) {
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
  EXT_THROW_INVALID("Unsupported tree node mode: ", mode);
}

bool ApplyTreeNodeMode(TreeNodeMode mode, double feature_value, double threshold) {
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
    EXT_THROW_INVALID("ApplyTreeNodeMode: LEAF nodes should not be compared.");
  }
  EXT_THROW_INVALID("ApplyTreeNodeMode: unknown mode.");
}

bool ApplyTreeNodeModeV5(TreeNodeModeV5 mode, double feature_value, double threshold) {
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
    EXT_THROW_INVALID("ApplyTreeNodeModeV5: BRANCH_MEMBER must be evaluated via "
                      "membership_values, not via ApplyTreeNodeModeV5.");
  }
  EXT_THROW_INVALID("ApplyTreeNodeModeV5: unknown mode.");
}

void AggregateTreeLeafWeight(std::vector<float> &accum, int64_t target_id, float weight,
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
    EXT_THROW_INVALID("AggregateTreeLeafWeight: unsupported aggregate_function: ", agg);
  }
}

void FinalizeAggregation(std::vector<float> &accum, const std::vector<int64_t> &counts,
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

void ApplyPostTransform(float *scores, size_t count, const std::string &post_transform) {
  if (post_transform == "NONE") {
    return;
  }
  if (post_transform == "SOFTMAX") {
    float max_val = *std::max_element(scores, scores + count);
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      scores[i] = std::exp(scores[i] - max_val);
      sum += scores[i];
    }
    for (size_t i = 0; i < count; ++i) {
      scores[i] /= sum;
    }
    return;
  }
  if (post_transform == "LOGISTIC") {
    for (size_t i = 0; i < count; ++i) {
      scores[i] = 1.0f / (1.0f + std::exp(-scores[i]));
    }
    return;
  }
  if (post_transform == "SOFTMAX_ZERO") {
    float max_val = *std::max_element(scores, scores + count);
    if (max_val == 0.0f) {
      return;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      scores[i] = std::exp(scores[i] - max_val);
      sum += scores[i];
    }
    for (size_t i = 0; i < count; ++i) {
      scores[i] /= sum;
    }
    return;
  }
  EXT_THROW_INVALID("ApplyPostTransform: unsupported post_transform: ", post_transform);
}

void ApplyPostTransform(std::vector<float> &scores, const std::string &post_transform) {
  ApplyPostTransform(scores.data(), scores.size(), post_transform);
}

int64_t TraverseClassicTree(const ClassicNodeMap &node_map, int64_t tree_id, const double *x_row,
                            int64_t feature_count) {
  int64_t cur_node_id = 0;
  for (;;) {
    auto it = node_map.find({tree_id, cur_node_id});
    EXT_ENFORCE_INVALID(it != node_map.end(), "TraverseClassicTree: node not found in tree.");
    const ClassicTreeNode &node = it->second;
    if (node.mode == TreeNodeMode::kLeaf) {
      return cur_node_id;
    }
    EXT_ENFORCE_INVALID(node.feature_id >= 0 && node.feature_id < feature_count,
                        "TraverseClassicTree: feature_id out of range.");
    const double feature_value = x_row[node.feature_id];
    bool go_true;
    if (std::isnan(feature_value)) {
      go_true = node.missing_tracks_true;
    } else {
      go_true = ApplyTreeNodeMode(node.mode, feature_value, node.threshold);
    }
    cur_node_id = go_true ? node.true_node_id : node.false_node_id;
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
