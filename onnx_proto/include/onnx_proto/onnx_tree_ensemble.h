// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"

#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

/// Returns an error for invalid opset-5 TreeEnsemble topology, or an empty string.
inline std::string ValidateTreeEnsembleTopology(std::span<const int64_t> roots, size_t node_count,
                                                size_t leaf_count,
                                                std::span<const int64_t> true_ids,
                                                std::span<const int64_t> false_ids,
                                                std::span<const int64_t> true_leafs,
                                                std::span<const int64_t> false_leafs) {
  for (const auto &[name, size] : {std::pair{"nodes_truenodeids", true_ids.size()},
                                   {"nodes_falsenodeids", false_ids.size()},
                                   {"nodes_trueleafs", true_leafs.size()},
                                   {"nodes_falseleafs", false_leafs.size()}}) {
    if (size != node_count) {
      return std::string("TreeEnsemble: attribute '") + name + "' must have length " +
             std::to_string(node_count) + ", got " + std::to_string(size) + ".";
    }
  }
  for (size_t i = 0; i < node_count; ++i) {
    for (bool is_true : {true, false}) {
      const int64_t leaf = is_true ? true_leafs[i] : false_leafs[i];
      const int64_t child = is_true ? true_ids[i] : false_ids[i];
      const std::string branch = is_true ? "true" : "false";
      if (leaf != 0 && leaf != 1) {
        return "TreeEnsemble: nodes_" + branch + "leafs must contain 0 or 1 at index " +
               std::to_string(i) + ".";
      }
      if (child < 0 || static_cast<uint64_t>(child) >= (leaf ? leaf_count : node_count)) {
        return "TreeEnsemble: " + std::string(leaf ? "leaf" : "internal node") + " index " +
               std::to_string(child) + " out of range in nodes_" + branch + "nodeids at index " +
               std::to_string(i) + ".";
      }
    }
  }
  // Tree-specific visit stamps avoid clearing the entire node array for each root.
  std::vector<size_t> visited(node_count, roots.size());
  std::vector<int64_t> pending;
  for (size_t tree = 0; tree < roots.size(); ++tree) {
    const int64_t root = roots[tree];
    if (root < 0 || static_cast<uint64_t>(root) >= node_count) {
      return "TreeEnsemble: internal node index " + std::to_string(root) +
             " out of range in tree_roots at index " + std::to_string(tree) + ".";
    }
    pending.push_back(root);
    while (!pending.empty()) {
      const size_t node = static_cast<size_t>(pending.back());
      pending.pop_back();
      if (visited[node] == tree) {
        return "TreeEnsemble contains a cycle or shared internal node at index " +
               std::to_string(node) + " in tree " + std::to_string(tree) + ".";
      }
      visited[node] = tree;
      if (!false_leafs[node]) {
        pending.push_back(false_ids[node]);
      }
      if (!true_leafs[node]) {
        pending.push_back(true_ids[node]);
      }
    }
  }
  return {};
}

/// Validates attribute lengths and topology without reading tensor payloads.
/// Accepts either a NodeProto attribute lookup or an inference-context lookup.
template <typename GetAttribute>
std::string ValidateTreeEnsembleAttributes(GetAttribute get_attribute) {
  // Function attribute references cannot be validated until they are bound.
  for (const char *name :
       {"tree_roots", "nodes_featureids", "nodes_splits", "nodes_modes", "nodes_truenodeids",
        "nodes_falsenodeids", "nodes_trueleafs", "nodes_falseleafs",
        "nodes_missing_value_tracks_true", "nodes_hitrates", "leaf_targetids", "leaf_weights"}) {
    const auto *attr = get_attribute(name);
    if (attr && !attr->ref_attr_name().empty()) {
      return {};
    }
  }
  const auto *features = get_attribute("nodes_featureids");
  const auto *targets = get_attribute("leaf_targetids");
  if (!features || !targets) {
    return "TreeEnsemble: attributes 'nodes_featureids' and 'leaf_targetids' are required.";
  }
  const size_t node_count = features->ints().size();
  const size_t leaf_count = targets->ints().size();
  for (const char *name : {"tree_roots", "nodes_featureids", "nodes_truenodeids",
                           "nodes_falsenodeids", "nodes_trueleafs", "nodes_falseleafs",
                           "nodes_missing_value_tracks_true", "leaf_targetids"}) {
    const auto *attr = get_attribute(name);
    const std::string key(name);
    if (!attr && key == "nodes_missing_value_tracks_true") {
      continue;
    }
    if (!attr || attr->type() != AttributeProto::INTS) {
      return "TreeEnsemble: attribute '" + key + "' must be INTS.";
    }
    if (key != "tree_roots" && key != "leaf_targetids" && attr->ints().size() != node_count) {
      return "TreeEnsemble: attribute '" + key + "' must have length " +
             std::to_string(node_count) + ", got " + std::to_string(attr->ints_size()) + ".";
    }
  }
  for (const char *name : {"nodes_splits", "nodes_modes", "nodes_hitrates", "leaf_weights"}) {
    const auto *attr = get_attribute(name);
    const std::string key(name);
    if (!attr && key == "nodes_hitrates") {
      continue;
    }
    if (!attr || attr->type() != AttributeProto::TENSOR || !attr->has_t()) {
      return "TreeEnsemble: attribute '" + key + "' must be a tensor.";
    }
    if (attr->t().dims_size() != 1) {
      return "TreeEnsemble: attribute '" + key + "' must be 1D.";
    }
    const size_t expected = key == "leaf_weights" ? leaf_count : node_count;
    if (attr->t().dims(0) < 0 || static_cast<uint64_t>(attr->t().dims(0)) != expected) {
      return "TreeEnsemble: attribute '" + key + "' must have length " + std::to_string(expected) +
             ", got " + std::to_string(attr->t().dims(0)) + ".";
    }
  }
  return ValidateTreeEnsembleTopology(
      get_attribute("tree_roots")->ints(), node_count, leaf_count,
      get_attribute("nodes_truenodeids")->ints(), get_attribute("nodes_falsenodeids")->ints(),
      get_attribute("nodes_trueleafs")->ints(), get_attribute("nodes_falseleafs")->ints());
}

} // namespace ONNX_LIGHT_NAMESPACE
