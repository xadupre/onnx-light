// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_svm_common.h"
#include "onnx_extensions/kernels/kernels/traditionalml/kernel_tree_ensemble_common.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cmath>
#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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
    const std::vector<double> &nodes_splits, const std::vector<uint8_t> &nodes_modes,
    const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_trueleafs, const std::vector<int64_t> &nodes_falseleafs,
    const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &leaf_targetids,
    const std::vector<double> &leaf_weights,
    const std::vector<std::vector<double>> &node_member_sets, const T *x_row, int64_t feature_count,
    int64_t n_targets, int64_t aggregate_function, std::vector<T> &accum,
    std::vector<int64_t> &counts) {
  int64_t node_idx = root_node_idx;
  for (;;) {
    EXT_ENFORCE_INVALID(node_idx >= 0 && node_idx < static_cast<int64_t>(nodes_featureids.size()),
                        "kernel::TreeEnsemble: node index out of range.");
    const auto mode = static_cast<TreeNodeModeV5>(nodes_modes[static_cast<size_t>(node_idx)]);
    const int64_t feat = nodes_featureids[static_cast<size_t>(node_idx)];
    EXT_ENFORCE_INVALID(feat >= 0 && feat < feature_count,
                        "kernel::TreeEnsemble: feature_id out of range.");
    const double feature_value = static_cast<double>(x_row[feat]);
    const double threshold = nodes_splits[static_cast<size_t>(node_idx)];
    bool go_true;
    if (std::isnan(feature_value)) {
      go_true = !nodes_missing.empty() && nodes_missing[static_cast<size_t>(node_idx)] != 0;
    } else if (mode == TreeNodeModeV5::kBranchMember) {
      const std::vector<double> &members = node_member_sets[static_cast<size_t>(node_idx)];
      go_true = false;
      for (const double &m : members) {
        if (m == feature_value) {
          go_true = true;
          break;
        }
      }
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
      const T weight = static_cast<T>(leaf_weights[static_cast<size_t>(next_idx)]);
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

TreeEnsemble::TreeEnsemble(
    const KernelContext &ctx, const std::vector<int64_t> &tree_roots,
    const std::vector<int64_t> &nodes_featureids, const std::vector<double> &nodes_splits,
    const std::vector<uint8_t> &nodes_modes, const std::vector<int64_t> &nodes_truenodeids,
    const std::vector<int64_t> &nodes_falsenodeids, const std::vector<int64_t> &nodes_trueleafs,
    const std::vector<int64_t> &nodes_falseleafs, const std::vector<int64_t> &nodes_missing,
    const std::vector<int64_t> &leaf_targetids, const std::vector<double> &leaf_weights,
    const std::vector<double> &membership_values)
    : KernelBase(ctx), tree_roots_(tree_roots), nodes_featureids_(nodes_featureids),
      nodes_splits_(nodes_splits), nodes_modes_(nodes_modes), nodes_truenodeids_(nodes_truenodeids),
      nodes_falsenodeids_(nodes_falsenodeids), nodes_trueleafs_(nodes_trueleafs),
      nodes_falseleafs_(nodes_falseleafs), nodes_missing_(nodes_missing),
      leaf_targetids_(leaf_targetids), leaf_weights_(leaf_weights) {
  // Precompute per-node membership sets for BRANCH_MEMBER (mode 6) nodes by
  // walking ``membership_values`` in nodes_modes order, where each set is
  // delimited by a NaN sentinel.
  const size_t n_nodes = nodes_modes_.size();
  node_member_sets_.resize(n_nodes);
  size_t mv_cursor = 0;
  for (size_t i = 0; i < n_nodes; ++i) {
    if (static_cast<TreeNodeModeV5>(nodes_modes_[i]) != TreeNodeModeV5::kBranchMember) {
      continue;
    }
    std::vector<double> &members = node_member_sets_[i];
    while (mv_cursor < membership_values.size()) {
      const double value = membership_values[mv_cursor++];
      if (std::isnan(value)) {
        break;
      }
      members.push_back(value);
    }
  }
}

template <typename T>
Tensor TreeEnsemble::operator()(const Tensor &x, int64_t n_targets, int64_t aggregate_function,
                                int64_t post_transform, RuntimeContext *rt) const {
  (void)rt;
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

    for (int64_t root_idx : tree_roots_) {
      TraverseTreeV5<T>(root_idx, nodes_featureids_, nodes_splits_, nodes_modes_,
                        nodes_truenodeids_, nodes_falsenodeids_, nodes_trueleafs_,
                        nodes_falseleafs_, nodes_missing_, leaf_targetids_, leaf_weights_,
                        node_member_sets_, x_row, feature_count, n_targets, aggregate_function,
                        accum, counts);
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

  return Tensor::From<T>("", {sample_count, n_targets}, output_flat, ctx_.allocator);
}

#define ONNX_LIGHT_INSTANTIATE_TREE_ENSEMBLE(T)                                                    \
  template Tensor TreeEnsemble::operator()<T>(const Tensor &, int64_t, int64_t, int64_t,           \
                                              RuntimeContext *) const

ONNX_LIGHT_INSTANTIATE_TREE_ENSEMBLE(float);
ONNX_LIGHT_INSTANTIATE_TREE_ENSEMBLE(double);

#undef ONNX_LIGHT_INSTANTIATE_TREE_ENSEMBLE

void TreeEnsemble::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const std::vector<int64_t> tree_roots = GetAttributeIntsOrDefault(node, "tree_roots", {});
  const std::vector<int64_t> nodes_featureids =
      GetAttributeIntsOrDefault(node, "nodes_featureids", {});
  const std::vector<int64_t> nodes_truenodeids =
      GetAttributeIntsOrDefault(node, "nodes_truenodeids", {});
  const std::vector<int64_t> nodes_falsenodeids =
      GetAttributeIntsOrDefault(node, "nodes_falsenodeids", {});
  const std::vector<int64_t> nodes_trueleafs =
      GetAttributeIntsOrDefault(node, "nodes_trueleafs", {});
  const std::vector<int64_t> nodes_falseleafs =
      GetAttributeIntsOrDefault(node, "nodes_falseleafs", {});
  const std::vector<int64_t> nodes_missing =
      GetAttributeIntsOrDefault(node, "nodes_missing_value_tracks_true", {});
  const std::vector<int64_t> leaf_targetids = GetAttributeIntsOrDefault(node, "leaf_targetids", {});
  const int64_t n_targets = GetAttributeIntOrDefault(node, "n_targets", 1);
  const int64_t aggregate_function = GetAttributeIntOrDefault(node, "aggregate_function", 1);
  const int64_t post_transform = GetAttributeIntOrDefault(node, "post_transform", 0);
  const Tensor nodes_splits = GetRequiredAttributeTensor(node, "nodes_splits");
  const Tensor leaf_weights = GetRequiredAttributeTensor(node, "leaf_weights");
  const Tensor nodes_modes_t = GetRequiredAttributeTensor(node, "nodes_modes");
  const Tensor membership_values =
      GetAttributeTensorOrEmpty(node, "membership_values", x.data_type);
  EXT_ENFORCE_INVALID(!(nodes_modes_t.data_type != static_cast<int32_t>(DataType::UINT8)),
                      "RunNode: TreeEnsemble attribute 'nodes_modes' must be a UINT8 tensor.");
  EXT_ENFORCE_INVALID(
      !(nodes_splits.data_type != x.data_type || leaf_weights.data_type != x.data_type),
      "RunNode: TreeEnsemble attributes 'nodes_splits' and 'leaf_weights' must "
      "have the same element type as input 'X'.");
  EXT_ENFORCE_INVALID(
      !(membership_values.element_count() > 0 && membership_values.data_type != x.data_type),
      "RunNode: TreeEnsemble attribute 'membership_values' must have the same "
      "element type as input 'X'.");
  const std::span<const uint8_t> nodes_modes_span = TensorSpan<uint8_t>(nodes_modes_t);
  // Materialize the tensor-typed attributes as ``double`` so the kernel
  // owns its tree data (independent of the input element type).
  const auto tensor_to_double = [](const Tensor &t) -> std::vector<double> {
    if (t.element_count() == 0) {
      return {};
    }
    switch (t.data_type) {
    case static_cast<int32_t>(DataType::FLOAT): {
      const std::span<const float> s = TensorSpan<float>(t);
      return std::vector<double>(s.begin(), s.end());
    }
    case static_cast<int32_t>(DataType::DOUBLE): {
      const std::span<const double> s = TensorSpan<double>(t);
      return std::vector<double>(s.begin(), s.end());
    }
    default:
      EXT_THROW_INVALID("RunNode: TreeEnsemble input 'X' must be FLOAT or DOUBLE.");
    }
  };
  const std::vector<double> nodes_splits_d = tensor_to_double(nodes_splits);
  const std::vector<double> leaf_weights_d = tensor_to_double(leaf_weights);
  const std::vector<double> membership_d = tensor_to_double(membership_values);
  const std::vector<uint8_t> nodes_modes_vec(nodes_modes_span.begin(), nodes_modes_span.end());
  onnx_kernels::kernel::TreeEnsemble tree_ens(
      rt.kernel_ctx(), tree_roots, nodes_featureids, nodes_splits_d, nodes_modes_vec,
      nodes_truenodeids, nodes_falsenodeids, nodes_trueleafs, nodes_falseleafs, nodes_missing,
      leaf_targetids, leaf_weights_d, membership_d);
  Tensor y;
  switch (x.data_type) {
  case static_cast<int32_t>(DataType::FLOAT):
    y = tree_ens.operator()<float>(x, n_targets, aggregate_function, post_transform);
    break;
  case static_cast<int32_t>(DataType::DOUBLE):
    y = tree_ens.operator()<double>(x, n_targets, aggregate_function, post_transform);
    break;
  default:
    EXT_THROW_INVALID("RunNode: TreeEnsemble input 'X' must be FLOAT or DOUBLE.");
  }
  SetOutput(node, 0, std::move(y), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
