// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_svm_common.h"
#include "onnx_extensions/kernels/kernels/traditionalml/kernel_tree_ensemble_common.h"

#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

TreeEnsembleRegressor::TreeEnsembleRegressor(
    const KernelContext &ctx, const std::vector<int64_t> &nodes_treeids,
    const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
    const std::vector<float> &nodes_values, const ParamStrings &nodes_modes,
    const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &target_treeids,
    const std::vector<int64_t> &target_nodeids, const std::vector<int64_t> &target_ids,
    const std::vector<float> &target_weights)
    : KernelBase(ctx), node_map_(BuildClassicNodeMap(nodes_treeids, nodes_nodeids, nodes_featureids,
                                                     nodes_values, nodes_modes, nodes_truenodeids,
                                                     nodes_falsenodeids, nodes_missing)) {
  // Build the leaf map: (tree_id, node_id) -> list of (target_id, weight).
  const size_t n_leaves = target_treeids.size();
  EXT_ENFORCE_INVALID(target_nodeids.size() == n_leaves && target_ids.size() == n_leaves &&
                          target_weights.size() == n_leaves,
                      "kernel::TreeEnsembleRegressor: target_* arrays must have the same length.");
  leaf_map_.reserve(n_leaves);
  for (size_t i = 0; i < n_leaves; ++i) {
    leaf_map_[{target_treeids[i], target_nodeids[i]}].push_back({target_ids[i], target_weights[i]});
  }

  // Collect the set of distinct tree ids (in traversal order).
  std::unordered_set<int64_t> seen;
  tree_ids_.reserve(nodes_treeids.size());
  for (int64_t tid : nodes_treeids) {
    if (seen.insert(tid).second) {
      tree_ids_.push_back(tid);
    }
  }
}

template <typename T>
Tensor TreeEnsembleRegressor::operator()(const Tensor &x, int64_t n_targets,
                                         const std::string &aggregate_function,
                                         const std::string &post_transform,
                                         const std::vector<float> &base_values,
                                         RuntimeContext *rt) const {
  (void)rt;
  EXT_ENFORCE_INVALID(n_targets >= 1, "kernel::TreeEnsembleRegressor: n_targets must be >= 1.");
  EXT_ENFORCE_INVALID(post_transform == "NONE",
                      "kernel::TreeEnsembleRegressor: only post_transform 'NONE' is supported.");

  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);

  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  std::vector<float> output_flat(static_cast<size_t>(sample_count * n_targets), 0.0f);

  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values.data() + n * feature_count;

    // Per-sample accumulator.
    std::vector<float> accum(static_cast<size_t>(n_targets), 0.0f);
    std::vector<int64_t> counts(static_cast<size_t>(n_targets), 0);

    for (int64_t tree_id : tree_ids_) {
      const int64_t leaf_node_id = TraverseClassicTree(node_map_, tree_id, x_row, feature_count);
      auto lit = leaf_map_.find({tree_id, leaf_node_id});
      if (lit != leaf_map_.end()) {
        for (const LeafEntry &entry : lit->second) {
          AggregateTreeLeafWeight(accum, entry.target_id, entry.weight, aggregate_function, counts);
        }
      }
    }

    FinalizeAggregation(accum, counts, aggregate_function);

    // Apply base_values and post_transform.
    if (!base_values.empty()) {
      EXT_ENFORCE_INVALID(static_cast<int64_t>(base_values.size()) == n_targets,
                          "kernel::TreeEnsembleRegressor: base_values size must equal n_targets.");
      for (int64_t t = 0; t < n_targets; ++t) {
        accum[static_cast<size_t>(t)] += base_values[static_cast<size_t>(t)];
      }
    }
    ApplyPostTransform(accum, post_transform);

    for (int64_t t = 0; t < n_targets; ++t) {
      output_flat[static_cast<size_t>(n * n_targets + t)] = accum[static_cast<size_t>(t)];
    }
  }

  return Tensor::FromFloat("", {sample_count, n_targets}, output_flat, ctx_.allocator);
}

#define ONNX_LIGHT_INSTANTIATE_TREE_REGRESSOR(T)                                                   \
  template Tensor TreeEnsembleRegressor::operator()<T>(                                            \
      const Tensor &, int64_t, const std::string &, const std::string &,                           \
      const std::vector<float> &, RuntimeContext *) const

ONNX_LIGHT_INSTANTIATE_TREE_REGRESSOR(float);
ONNX_LIGHT_INSTANTIATE_TREE_REGRESSOR(double);
ONNX_LIGHT_INSTANTIATE_TREE_REGRESSOR(int64_t);
ONNX_LIGHT_INSTANTIATE_TREE_REGRESSOR(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_TREE_REGRESSOR

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
