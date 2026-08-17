// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_svm_common.h"
#include "onnx_extensions/kernels/kernels/traditionalml/kernel_tree_ensemble_common.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

/// Runs the tree ensemble on the input and writes raw class scores [N, E] into
/// the caller-provided ``scores`` buffer (allocator-backed output storage).
/// The buffer must be zero-initialized and hold ``sample_count * n_classes``
/// floats. Uses class_treeids/class_nodeids/class_ids/class_weights (classic
/// encoding).
void ComputeClassifierScores(const ClassicNodeMap &node_map, const ClassicLeafMap &leaf_map,
                             const std::vector<int64_t> &tree_ids, const double *x_values,
                             int64_t sample_count, int64_t feature_count, int64_t n_classes,
                             const std::vector<float> &base_values,
                             const std::string &post_transform, float *scores) {
  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values + n * feature_count;
    float *row = scores + n * n_classes;
    // Allocator-backed output storage is not guaranteed to be zeroed, so the
    // per-sample score row is cleared before scores are accumulated into it.
    std::fill(row, row + n_classes, 0.0f);

    for (int64_t tree_id : tree_ids) {
      const int64_t leaf_node_id = TraverseClassicTree(node_map, tree_id, x_row, feature_count);
      auto lit = leaf_map.find({tree_id, leaf_node_id});
      if (lit != leaf_map.end()) {
        for (const LeafEntry &entry : lit->second) {
          EXT_ENFORCE_INVALID(entry.target_id >= 0 && entry.target_id < n_classes,
                              "ComputeClassifierScores: class_id out of range.");
          row[static_cast<size_t>(entry.target_id)] += entry.weight;
        }
      }
    }

    if (!base_values.empty()) {
      EXT_ENFORCE_INVALID(
          static_cast<int64_t>(base_values.size()) == n_classes,
          "ComputeClassifierScores: base_values size must equal number of classes.");
      for (int64_t c = 0; c < n_classes; ++c) {
        row[static_cast<size_t>(c)] += base_values[static_cast<size_t>(c)];
      }
    }

    ApplyPostTransform(row, static_cast<size_t>(n_classes), post_transform);
  }
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

TreeEnsembleClassifier::TreeEnsembleClassifier(
    const KernelContext &ctx, const std::vector<int64_t> &nodes_treeids,
    const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
    const std::vector<float> &nodes_values, const ParamStrings &nodes_modes,
    const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &class_treeids,
    const std::vector<int64_t> &class_nodeids, const std::vector<int64_t> &class_ids,
    const std::vector<float> &class_weights)
    : KernelBase(ctx), node_map_(BuildClassicNodeMap(nodes_treeids, nodes_nodeids, nodes_featureids,
                                                     nodes_values, nodes_modes, nodes_truenodeids,
                                                     nodes_falsenodeids, nodes_missing)),
      tree_ids_(DistinctTreeIds(nodes_treeids)) {
  const size_t n_leaves = class_treeids.size();
  EXT_ENFORCE_INVALID(class_nodeids.size() == n_leaves && class_ids.size() == n_leaves &&
                          class_weights.size() == n_leaves,
                      "kernel::TreeEnsembleClassifier: class_* arrays must have the same length.");
  leaf_map_.reserve(n_leaves);
  for (size_t i = 0; i < n_leaves; ++i) {
    leaf_map_[{class_treeids[i], class_nodeids[i]}].push_back({class_ids[i], class_weights[i]});
  }
}

template <typename T>
std::pair<Tensor, Tensor>
TreeEnsembleClassifier::operator()(const Tensor &x, const std::vector<int64_t> &classlabels_int64s,
                                   const std::vector<float> &base_values,
                                   const std::string &post_transform, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(!classlabels_int64s.empty(),
                      "kernel::TreeEnsembleClassifier: classlabels_int64s must not be empty.");

  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);

  const int64_t n_classes = static_cast<int64_t>(classlabels_int64s.size());
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);

  const size_t score_n_bytes = static_cast<size_t>(sample_count * n_classes) * sizeof(float);
  Tensor z = rt ? rt->MakeOutputTensor(1, DataType::FLOAT, {sample_count, n_classes}, score_n_bytes)
                : MakeOutputTensor(DataType::FLOAT, {sample_count, n_classes}, score_n_bytes,
                                   ctx_.allocator);
  float *scores = z.AsFloat();
  ComputeClassifierScores(node_map_, leaf_map_, tree_ids_, x_values.data(), sample_count,
                          feature_count, n_classes, base_values, post_transform, scores);

  std::vector<int64_t> labels(static_cast<size_t>(sample_count));
  for (int64_t n = 0; n < sample_count; ++n) {
    const int64_t idx = ArgMaxRow(scores + n * n_classes, n_classes);
    labels[static_cast<size_t>(n)] = classlabels_int64s[static_cast<size_t>(idx)];
  }

  Tensor y = rt ? rt->MakeOutputTensor(0, DataType::INT64, {sample_count},
                                       static_cast<size_t>(sample_count) * sizeof(int64_t))
                : Tensor::FromInt64("", {sample_count}, labels, ctx_.allocator);
  if (rt != nullptr) {
    std::copy(labels.begin(), labels.end(), y.AsInt64());
  }
  return std::make_pair(std::move(y), std::move(z));
}

template <typename T>
std::pair<Tensor, Tensor>
TreeEnsembleClassifier::operator()(const Tensor &x, const ParamStrings &classlabels_strings,
                                   const std::vector<float> &base_values,
                                   const std::string &post_transform, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(!classlabels_strings.empty(),
                      "kernel::TreeEnsembleClassifier: classlabels_strings must not be empty.");

  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);

  const int64_t n_classes = static_cast<int64_t>(classlabels_strings.size());
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);

  const size_t score_n_bytes = static_cast<size_t>(sample_count * n_classes) * sizeof(float);
  Tensor z = rt ? rt->MakeOutputTensor(1, DataType::FLOAT, {sample_count, n_classes}, score_n_bytes)
                : MakeOutputTensor(DataType::FLOAT, {sample_count, n_classes}, score_n_bytes,
                                   ctx_.allocator);
  float *scores = z.AsFloat();
  ComputeClassifierScores(node_map_, leaf_map_, tree_ids_, x_values.data(), sample_count,
                          feature_count, n_classes, base_values, post_transform, scores);

  std::vector<std::string> labels(static_cast<size_t>(sample_count));
  for (int64_t n = 0; n < sample_count; ++n) {
    const int64_t idx = ArgMaxRow(scores + n * n_classes, n_classes);
    labels[static_cast<size_t>(n)] = classlabels_strings[static_cast<size_t>(idx)];
  }

  Tensor y = rt ? rt->MakeOutputTensor(0, DataType::STRING, {sample_count}, 0)
                : Tensor::FromStrings("", {sample_count}, labels);
  if (rt != nullptr) {
    y.string_data = std::move(labels);
  }
  return std::make_pair(std::move(y), std::move(z));
}

#define ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(T)                                                  \
  template std::pair<Tensor, Tensor> TreeEnsembleClassifier::operator()<T>(                        \
      const Tensor &, const std::vector<int64_t> &, const std::vector<float> &,                    \
      const std::string &, RuntimeContext *) const;                                                \
  template std::pair<Tensor, Tensor> TreeEnsembleClassifier::operator()<T>(                        \
      const Tensor &, const ParamStrings &, const std::vector<float> &, const std::string &,       \
      RuntimeContext *) const

ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(float);
ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(double);
ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(int64_t);
ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_TREE_CLASSIFIER

void TreeEnsembleClassifier::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 2);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const std::vector<int64_t> nodes_treeids = GetAttributeIntsOrDefault(node, "nodes_treeids", {});
  const std::vector<int64_t> nodes_nodeids = GetAttributeIntsOrDefault(node, "nodes_nodeids", {});
  const std::vector<int64_t> nodes_featureids =
      GetAttributeIntsOrDefault(node, "nodes_featureids", {});
  const std::vector<float> nodes_values = GetAttributeFloatsOrDefault(node, "nodes_values", {});
  const ParamStrings nodes_modes = GetAttributeStringsOrDefault(node, "nodes_modes", {});
  const std::vector<int64_t> nodes_truenodeids =
      GetAttributeIntsOrDefault(node, "nodes_truenodeids", {});
  const std::vector<int64_t> nodes_falsenodeids =
      GetAttributeIntsOrDefault(node, "nodes_falsenodeids", {});
  const std::vector<int64_t> nodes_missing =
      GetAttributeIntsOrDefault(node, "nodes_missing_value_tracks_true", {});
  const std::vector<int64_t> class_treeids = GetAttributeIntsOrDefault(node, "class_treeids", {});
  const std::vector<int64_t> class_nodeids = GetAttributeIntsOrDefault(node, "class_nodeids", {});
  const std::vector<int64_t> class_ids = GetAttributeIntsOrDefault(node, "class_ids", {});
  const std::vector<float> class_weights = GetAttributeFloatsOrDefault(node, "class_weights", {});
  const std::vector<int64_t> classlabels_int64s =
      GetAttributeIntsOrDefault(node, "classlabels_int64s", {});
  const ParamStrings classlabels_strings =
      GetAttributeStringsOrDefault(node, "classlabels_strings", {});
  const std::vector<float> base_values = GetAttributeFloatsOrDefault(node, "base_values", {});
  const std::string post_transform = GetAttributeStringOrDefault(node, "post_transform", "NONE");
  const bool use_strings = !classlabels_strings.empty();
  const bool has_ints = !classlabels_int64s.empty();
  EXT_ENFORCE_INVALID(use_strings != has_ints,
                      "RunNode: TreeEnsembleClassifier requires exactly one of "
                      "'classlabels_int64s' or 'classlabels_strings' to be set.");
  onnx_kernels::kernel::TreeEnsembleClassifier cls(
      rt.kernel_ctx(), nodes_treeids, nodes_nodeids, nodes_featureids, nodes_values, nodes_modes,
      nodes_truenodeids, nodes_falsenodeids, nodes_missing, class_treeids, class_nodeids, class_ids,
      class_weights);
  std::pair<Tensor, Tensor> yz =
      DispatchTreeEnsembleClassicByDataType(x, "TreeEnsembleClassifier", [&](auto *tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        (void)tag;
        return use_strings ? cls.template operator()<T>(x, classlabels_strings, base_values,
                                                        post_transform, &rt)
                           : cls.template operator()<T>(x, classlabels_int64s, base_values,
                                                        post_transform, &rt);
      });
  SetOutput(node, 0, std::move(yz.first), rt);
  SetOutput(node, 1, std::move(yz.second), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
