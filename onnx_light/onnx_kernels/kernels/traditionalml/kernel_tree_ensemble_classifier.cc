// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_kernels/kernels/traditionalml/kernel_svm_common.h"
#include "onnx_kernels/kernels/traditionalml/kernel_tree_ensemble_common.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

/// Runs the tree ensemble on the input and returns raw class scores [N, E].
/// Uses class_treeids/class_nodeids/class_ids/class_weights (classic encoding).
std::vector<float>
ComputeClassifierScores(const ClassicNodeMap &node_map, const ClassicLeafMap &leaf_map,
                        const std::vector<int64_t> &tree_ids, const double *x_values,
                        int64_t sample_count, int64_t feature_count, int64_t n_classes,
                        const std::vector<float> &base_values, const std::string &post_transform) {
  std::vector<float> scores(static_cast<size_t>(sample_count * n_classes), 0.0f);

  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values + n * feature_count;
    std::vector<float> accum(static_cast<size_t>(n_classes), 0.0f);

    for (int64_t tree_id : tree_ids) {
      const int64_t leaf_node_id = TraverseClassicTree(node_map, tree_id, x_row, feature_count);
      auto lit = leaf_map.find({tree_id, leaf_node_id});
      if (lit != leaf_map.end()) {
        for (const LeafEntry &entry : lit->second) {
          EXT_ENFORCE_INVALID(entry.target_id >= 0 && entry.target_id < n_classes,
                              "ComputeClassifierScores: class_id out of range.");
          accum[static_cast<size_t>(entry.target_id)] += entry.weight;
        }
      }
    }

    if (!base_values.empty()) {
      EXT_ENFORCE_INVALID(
          static_cast<int64_t>(base_values.size()) == n_classes,
          "ComputeClassifierScores: base_values size must equal number of classes.");
      for (int64_t c = 0; c < n_classes; ++c) {
        accum[static_cast<size_t>(c)] += base_values[static_cast<size_t>(c)];
      }
    }

    ApplyPostTransform(accum, post_transform);

    for (int64_t c = 0; c < n_classes; ++c) {
      scores[static_cast<size_t>(n * n_classes + c)] = accum[static_cast<size_t>(c)];
    }
  }
  return scores;
}

/// Finds the argmax across a row of scores.
int64_t ArgMaxRow(const float *scores, int64_t count) {
  int64_t best = 0;
  for (int64_t i = 1; i < count; ++i) {
    if (scores[i] > scores[best]) {
      best = i;
    }
  }
  return best;
}

/// Collects distinct tree ids in encounter order.
std::vector<int64_t> DistinctTreeIds(const std::vector<int64_t> &nodes_treeids) {
  std::vector<int64_t> result;
  std::unordered_set<int64_t> seen;
  result.reserve(nodes_treeids.size());
  for (int64_t tid : nodes_treeids) {
    if (seen.insert(tid).second) {
      result.push_back(tid);
    }
  }
  return result;
}

} // namespace

template <typename T>
std::pair<Tensor, Tensor> TreeEnsembleClassifier::operator()(
    const Tensor &x, const std::vector<int64_t> &nodes_treeids,
    const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
    const std::vector<float> &nodes_values, const std::vector<std::string> &nodes_modes,
    const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &class_treeids,
    const std::vector<int64_t> &class_nodeids, const std::vector<int64_t> &class_ids,
    const std::vector<float> &class_weights, const std::vector<int64_t> &classlabels_int64s,
    const std::vector<float> &base_values, const std::string &post_transform) const {
  EXT_ENFORCE_INVALID(!classlabels_int64s.empty(),
                      "kernel::TreeEnsembleClassifier: classlabels_int64s must not be empty.");

  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);

  const int64_t n_classes = static_cast<int64_t>(classlabels_int64s.size());
  const ClassicNodeMap node_map =
      BuildClassicNodeMap(nodes_treeids, nodes_nodeids, nodes_featureids, nodes_values, nodes_modes,
                          nodes_truenodeids, nodes_falsenodeids, nodes_missing);

  const size_t n_leaves = class_treeids.size();
  EXT_ENFORCE_INVALID(class_nodeids.size() == n_leaves && class_ids.size() == n_leaves &&
                          class_weights.size() == n_leaves,
                      "kernel::TreeEnsembleClassifier: class_* arrays must have the same length.");

  ClassicLeafMap leaf_map;
  leaf_map.reserve(n_leaves);
  for (size_t i = 0; i < n_leaves; ++i) {
    leaf_map[{class_treeids[i], class_nodeids[i]}].push_back({class_ids[i], class_weights[i]});
  }

  const std::vector<int64_t> tree_ids = DistinctTreeIds(nodes_treeids);
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);

  const std::vector<float> scores =
      ComputeClassifierScores(node_map, leaf_map, tree_ids, x_values.data(), sample_count,
                              feature_count, n_classes, base_values, post_transform);

  std::vector<int64_t> labels(static_cast<size_t>(sample_count));
  for (int64_t n = 0; n < sample_count; ++n) {
    const int64_t idx = ArgMaxRow(scores.data() + n * n_classes, n_classes);
    labels[static_cast<size_t>(n)] = classlabels_int64s[static_cast<size_t>(idx)];
  }

  Tensor y = Tensor::FromInt64("", {sample_count}, labels);
  Tensor z = Tensor::FromFloat("", {sample_count, n_classes}, scores);
  return std::make_pair(std::move(y), std::move(z));
}

template <typename T>
std::pair<Tensor, Tensor> TreeEnsembleClassifier::operator()(
    const Tensor &x, const std::vector<int64_t> &nodes_treeids,
    const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
    const std::vector<float> &nodes_values, const std::vector<std::string> &nodes_modes,
    const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &class_treeids,
    const std::vector<int64_t> &class_nodeids, const std::vector<int64_t> &class_ids,
    const std::vector<float> &class_weights, const std::vector<std::string> &classlabels_strings,
    const std::vector<float> &base_values, const std::string &post_transform) const {
  EXT_ENFORCE_INVALID(!classlabels_strings.empty(),
                      "kernel::TreeEnsembleClassifier: classlabels_strings must not be empty.");

  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);

  const int64_t n_classes = static_cast<int64_t>(classlabels_strings.size());
  const ClassicNodeMap node_map =
      BuildClassicNodeMap(nodes_treeids, nodes_nodeids, nodes_featureids, nodes_values, nodes_modes,
                          nodes_truenodeids, nodes_falsenodeids, nodes_missing);

  const size_t n_leaves = class_treeids.size();
  EXT_ENFORCE_INVALID(class_nodeids.size() == n_leaves && class_ids.size() == n_leaves &&
                          class_weights.size() == n_leaves,
                      "kernel::TreeEnsembleClassifier: class_* arrays must have the same length.");

  ClassicLeafMap leaf_map;
  leaf_map.reserve(n_leaves);
  for (size_t i = 0; i < n_leaves; ++i) {
    leaf_map[{class_treeids[i], class_nodeids[i]}].push_back({class_ids[i], class_weights[i]});
  }

  const std::vector<int64_t> tree_ids = DistinctTreeIds(nodes_treeids);
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);

  const std::vector<float> scores =
      ComputeClassifierScores(node_map, leaf_map, tree_ids, x_values.data(), sample_count,
                              feature_count, n_classes, base_values, post_transform);

  std::vector<std::string> labels(static_cast<size_t>(sample_count));
  for (int64_t n = 0; n < sample_count; ++n) {
    const int64_t idx = ArgMaxRow(scores.data() + n * n_classes, n_classes);
    labels[static_cast<size_t>(n)] = classlabels_strings[static_cast<size_t>(idx)];
  }

  Tensor y = Tensor::FromStrings("", {sample_count}, labels);
  Tensor z = Tensor::FromFloat("", {sample_count, n_classes}, scores);
  return std::make_pair(std::move(y), std::move(z));
}

#define ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(T)                                                  \
  template std::pair<Tensor, Tensor> TreeEnsembleClassifier::operator()<T>(                        \
      const Tensor &, const std::vector<int64_t> &, const std::vector<int64_t> &,                  \
      const std::vector<int64_t> &, const std::vector<float> &, const std::vector<std::string> &,  \
      const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &,    \
      const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &,    \
      const std::vector<float> &, const std::vector<int64_t> &, const std::vector<float> &,        \
      const std::string &) const;                                                                  \
  template std::pair<Tensor, Tensor> TreeEnsembleClassifier::operator()<T>(                        \
      const Tensor &, const std::vector<int64_t> &, const std::vector<int64_t> &,                  \
      const std::vector<int64_t> &, const std::vector<float> &, const std::vector<std::string> &,  \
      const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &,    \
      const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &,    \
      const std::vector<float> &, const std::vector<std::string> &, const std::vector<float> &,    \
      const std::string &) const

ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(float);
ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(double);
ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(int64_t);
ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
