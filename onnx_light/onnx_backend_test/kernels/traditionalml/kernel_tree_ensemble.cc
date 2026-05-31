// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_backend_test/kernels/traditionalml/kernel_svm_common.h"
#include "onnx_backend_test/kernels/traditionalml/kernel_tree_ensemble_common.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

/// TreeEnsemble v5 aggregate function codes.
constexpr int64_t kAggAverage = 0;
constexpr int64_t kAggSum = 1;
constexpr int64_t kAggMin = 2;
constexpr int64_t kAggMax = 3;

/// TreeEnsemble v5 post_transform codes.
constexpr int64_t kPostNone = 0;
constexpr int64_t kPostSoftmax = 1;

/// Traverses a single tree (v5 encoding) for a single input sample and
/// accumulates leaf contributions into ``accum``.
///
/// The v5 encoding uses a flat array of interior nodes indexed by
/// ``tree_roots[tree_idx]``. At each node:
///   - compare x[feature_id] against nodes_splits[node_idx] using
///     nodes_modes[node_idx].
///   - if go_true: next = nodes_truenodeids[node_idx] and it is a leaf iff
///     nodes_trueleafs[node_idx] == 1.
///   - if go_false: next = nodes_falsenodeids[node_idx] and it is a leaf iff
///     nodes_falseleafs[node_idx] == 1.
///   - When a branch is a leaf, the index is used to look up
///     leaf_targetids[leaf_idx] and leaf_weights[leaf_idx].
template <typename T>
void TraverseTreeV5(
    int64_t root_node_idx, const std::vector<int64_t> &nodes_featureids,
    const std::vector<T> &nodes_splits, const std::vector<uint8_t> &nodes_modes,
    const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_trueleafs, const std::vector<int64_t> &nodes_falseleafs,
    const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &leaf_targetids,
    const std::vector<T> &leaf_weights, const T *x_row, int64_t feature_count, int64_t n_targets,
    int64_t aggregate_function, std::vector<T> &accum, std::vector<int64_t> &counts) {
  int64_t node_idx = root_node_idx;
  for (;;) {
    EXT_ENFORCE_INVALID(node_idx >= 0 && node_idx < static_cast<int64_t>(nodes_featureids.size()),
                        "kernel::TreeEnsemble: node index out of range.");
    const auto mode = static_cast<TreeNodeModeV5>(nodes_modes[static_cast<size_t>(node_idx)]);
    const int64_t feat = nodes_featureids[static_cast<size_t>(node_idx)];
    EXT_ENFORCE_INVALID(feat >= 0 && feat < feature_count,
                        "kernel::TreeEnsemble: feature_id out of range.");
    const double feature_value = static_cast<double>(x_row[feat]);
    const double threshold = static_cast<double>(nodes_splits[static_cast<size_t>(node_idx)]);
    bool go_true;
    if (std::isnan(feature_value)) {
      go_true = !nodes_missing.empty() && nodes_missing[static_cast<size_t>(node_idx)] != 0;
    } else {
      go_true = ApplyTreeNodeModeV5(mode, feature_value, threshold);
    }

    int64_t next_idx;
    bool is_leaf;
    if (go_true) {
      next_idx = nodes_truenodeids[static_cast<size_t>(node_idx)];
      is_leaf = nodes_trueleafs[static_cast<size_t>(node_idx)] != 0;
    } else {
      next_idx = nodes_falsenodeids[static_cast<size_t>(node_idx)];
      is_leaf = nodes_falseleafs[static_cast<size_t>(node_idx)] != 0;
    }

    if (is_leaf) {
      EXT_ENFORCE_INVALID(next_idx >= 0 && next_idx < static_cast<int64_t>(leaf_targetids.size()),
                          "kernel::TreeEnsemble: leaf index out of range.");
      const int64_t target_id = leaf_targetids[static_cast<size_t>(next_idx)];
      EXT_ENFORCE_INVALID(target_id >= 0 && target_id < n_targets,
                          "kernel::TreeEnsemble: leaf_targetid out of range.");
      const T weight = leaf_weights[static_cast<size_t>(next_idx)];
      if (aggregate_function == kAggSum || aggregate_function == kAggAverage) {
        accum[static_cast<size_t>(target_id)] += weight;
        counts[static_cast<size_t>(target_id)]++;
      } else if (aggregate_function == kAggMin) {
        if (counts[static_cast<size_t>(target_id)] == 0) {
          accum[static_cast<size_t>(target_id)] = weight;
        } else {
          accum[static_cast<size_t>(target_id)] =
              std::min(accum[static_cast<size_t>(target_id)], weight);
        }
        counts[static_cast<size_t>(target_id)]++;
      } else if (aggregate_function == kAggMax) {
        if (counts[static_cast<size_t>(target_id)] == 0) {
          accum[static_cast<size_t>(target_id)] = weight;
        } else {
          accum[static_cast<size_t>(target_id)] =
              std::max(accum[static_cast<size_t>(target_id)], weight);
        }
        counts[static_cast<size_t>(target_id)]++;
      }
      return;
    }
    node_idx = next_idx;
  }
}

} // namespace

template <typename T>
Tensor TreeEnsemble::operator()(
    const Tensor &x, const std::vector<int64_t> &tree_roots,
    const std::vector<int64_t> &nodes_featureids, const std::vector<T> &nodes_splits,
    const std::vector<uint8_t> &nodes_modes, const std::vector<int64_t> &nodes_truenodeids,
    const std::vector<int64_t> &nodes_falsenodeids, const std::vector<int64_t> &nodes_trueleafs,
    const std::vector<int64_t> &nodes_falseleafs, const std::vector<int64_t> &nodes_missing,
    const std::vector<int64_t> &leaf_targetids, const std::vector<T> &leaf_weights,
    int64_t n_targets, int64_t aggregate_function, int64_t post_transform) const {
  EXT_ENFORCE_INVALID(n_targets >= 1, "kernel::TreeEnsemble: n_targets must be >= 1.");
  EXT_ENFORCE_INVALID(post_transform == kPostNone || post_transform == kPostSoftmax,
                      "kernel::TreeEnsemble: only post_transform 0 (NONE) or 1 (SOFTMAX) "
                      "are supported.");

  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);

  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "kernel::TreeEnsemble: input data_type does not match template T.");
  const T *px = x.As<T>();

  std::vector<T> output_flat(static_cast<size_t>(sample_count * n_targets), T{0});

  for (int64_t n = 0; n < sample_count; ++n) {
    const T *x_row = px + n * feature_count;
    std::vector<T> accum(static_cast<size_t>(n_targets), T{0});
    std::vector<int64_t> counts(static_cast<size_t>(n_targets), 0);

    for (int64_t root_idx : tree_roots) {
      TraverseTreeV5<T>(root_idx, nodes_featureids, nodes_splits, nodes_modes, nodes_truenodeids,
                        nodes_falsenodeids, nodes_trueleafs, nodes_falseleafs, nodes_missing,
                        leaf_targetids, leaf_weights, x_row, feature_count, n_targets,
                        aggregate_function, accum, counts);
    }

    // Finalize AVERAGE.
    if (aggregate_function == kAggAverage) {
      for (int64_t t = 0; t < n_targets; ++t) {
        if (counts[static_cast<size_t>(t)] > 0) {
          accum[static_cast<size_t>(t)] /= static_cast<T>(counts[static_cast<size_t>(t)]);
        }
      }
    }

    // Apply post_transform.
    if (post_transform == kPostSoftmax) {
      T max_val = accum[0];
      for (int64_t t = 1; t < n_targets; ++t) {
        if (accum[static_cast<size_t>(t)] > max_val) {
          max_val = accum[static_cast<size_t>(t)];
        }
      }
      T sum = T{0};
      for (int64_t t = 0; t < n_targets; ++t) {
        accum[static_cast<size_t>(t)] = static_cast<T>(std::exp(
            static_cast<double>(accum[static_cast<size_t>(t)]) - static_cast<double>(max_val)));
        sum += accum[static_cast<size_t>(t)];
      }
      for (int64_t t = 0; t < n_targets; ++t) {
        accum[static_cast<size_t>(t)] /= sum;
      }
    }

    for (int64_t t = 0; t < n_targets; ++t) {
      output_flat[static_cast<size_t>(n * n_targets + t)] = accum[static_cast<size_t>(t)];
    }
  }

  return Tensor::From<T>("", {sample_count, n_targets}, output_flat);
}

#define ONNX_LIGHT_INSTANTIATE_TREE_ENSEMBLE(T)                                                    \
  template Tensor TreeEnsemble::operator()<T>(                                                     \
      const Tensor &, const std::vector<int64_t> &, const std::vector<int64_t> &,                  \
      const std::vector<T> &, const std::vector<uint8_t> &, const std::vector<int64_t> &,          \
      const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &,    \
      const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<T> &, int64_t, \
      int64_t, int64_t) const

ONNX_LIGHT_INSTANTIATE_TREE_ENSEMBLE(float);
ONNX_LIGHT_INSTANTIATE_TREE_ENSEMBLE(double);

#undef ONNX_LIGHT_INSTANTIATE_TREE_ENSEMBLE

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
