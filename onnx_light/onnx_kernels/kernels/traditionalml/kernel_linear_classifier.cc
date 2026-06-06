// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_backend_test/kernels/traditionalml/kernel_svm_common.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Computes raw decision scores ``S`` of shape ``[sample_count, class_count]``.
// ``coefficients`` is flat row-major of shape ``[class_count, feature_count]``.
// ``intercepts`` is either empty or of length ``class_count``.
std::vector<float> ComputeLinearScores(const std::vector<double> &x_values, int64_t sample_count,
                                       int64_t feature_count,
                                       const std::vector<float> &coefficients,
                                       const std::vector<float> &intercepts, int64_t class_count) {
  EXT_ENFORCE_INVALID(class_count >= 1, "kernel::LinearClassifier requires at least one class.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(coefficients.size()) == class_count * feature_count,
                      "kernel::LinearClassifier coefficients size must be "
                      "class_count * feature_count.");
  EXT_ENFORCE_INVALID(intercepts.empty() || static_cast<int64_t>(intercepts.size()) == class_count,
                      "kernel::LinearClassifier intercepts size must be 0 or equal to "
                      "class_count.");
  std::vector<float> scores(static_cast<size_t>(sample_count * class_count), 0.0f);
  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values.data() + n * feature_count;
    for (int64_t c = 0; c < class_count; ++c) {
      const float *w = coefficients.data() + c * feature_count;
      double value = 0.0;
      for (int64_t j = 0; j < feature_count; ++j) {
        value += x_row[j] * static_cast<double>(w[j]);
      }
      if (!intercepts.empty()) {
        value += static_cast<double>(intercepts[static_cast<size_t>(c)]);
      }
      scores[static_cast<size_t>(n * class_count + c)] = static_cast<float>(value);
    }
  }
  return scores;
}

// Per the ONNX spec, when ``intercepts`` has length 1 but the operator declares
// two class labels, the operator behaves as a binary classifier. In that case
// the single raw score ``z`` is expanded to a pair ``[-z, z]``.
struct LinearClassifierExpansion {
  std::vector<float> scores; // shape [N, E_out]
  int64_t score_class_count; // E_out
};

LinearClassifierExpansion ExpandBinaryScores(std::vector<float> raw_scores, int64_t sample_count,
                                             int64_t raw_class_count, int64_t label_count) {
  if (raw_class_count == 1 && label_count == 2) {
    std::vector<float> expanded(static_cast<size_t>(sample_count * 2));
    for (int64_t n = 0; n < sample_count; ++n) {
      const float z = raw_scores[static_cast<size_t>(n)];
      expanded[static_cast<size_t>(n * 2)] = -z;
      expanded[static_cast<size_t>(n * 2 + 1)] = z;
    }
    return {std::move(expanded), 2};
  }
  return {std::move(raw_scores), raw_class_count};
}

int64_t ArgMax(const float *scores, int64_t count) {
  int64_t best = 0;
  for (int64_t i = 1; i < count; ++i) {
    if (scores[i] > scores[best]) {
      best = i;
    }
  }
  return best;
}

} // namespace

template <typename T>
std::pair<Tensor, Tensor> LinearClassifier::operator()(const Tensor &x,
                                                       const std::vector<float> &coefficients,
                                                       const std::vector<float> &intercepts,
                                                       const std::vector<int64_t> &class_labels,
                                                       const std::string &post_transform) const {
  EXT_ENFORCE_INVALID(post_transform == "NONE",
                      "kernel::LinearClassifier only supports post_transform == 'NONE'.");
  EXT_ENFORCE_INVALID(!class_labels.empty(),
                      "kernel::LinearClassifier requires non-empty class labels.");
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  const int64_t raw_class_count =
      feature_count == 0 ? 0 : static_cast<int64_t>(coefficients.size()) / feature_count;
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  std::vector<float> raw_scores = ComputeLinearScores(x_values, sample_count, feature_count,
                                                      coefficients, intercepts, raw_class_count);
  LinearClassifierExpansion expansion =
      ExpandBinaryScores(std::move(raw_scores), sample_count, raw_class_count,
                         static_cast<int64_t>(class_labels.size()));
  EXT_ENFORCE_INVALID(static_cast<int64_t>(class_labels.size()) == expansion.score_class_count,
                      "kernel::LinearClassifier class_labels size must match the number of "
                      "score columns.");

  std::vector<int64_t> labels(static_cast<size_t>(sample_count));
  for (int64_t n = 0; n < sample_count; ++n) {
    const int64_t idx = ArgMax(expansion.scores.data() + n * expansion.score_class_count,
                               expansion.score_class_count);
    labels[static_cast<size_t>(n)] = class_labels[static_cast<size_t>(idx)];
  }
  Tensor y = Tensor::FromInt64("", {sample_count}, labels);
  Tensor z = Tensor::FromFloat("", {sample_count, expansion.score_class_count}, expansion.scores);
  return std::make_pair(std::move(y), std::move(z));
}

template <typename T>
std::pair<Tensor, Tensor> LinearClassifier::operator()(const Tensor &x,
                                                       const std::vector<float> &coefficients,
                                                       const std::vector<float> &intercepts,
                                                       const std::vector<std::string> &class_labels,
                                                       const std::string &post_transform) const {
  EXT_ENFORCE_INVALID(post_transform == "NONE",
                      "kernel::LinearClassifier only supports post_transform == 'NONE'.");
  EXT_ENFORCE_INVALID(!class_labels.empty(),
                      "kernel::LinearClassifier requires non-empty class labels.");
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  const int64_t raw_class_count =
      feature_count == 0 ? 0 : static_cast<int64_t>(coefficients.size()) / feature_count;
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  std::vector<float> raw_scores = ComputeLinearScores(x_values, sample_count, feature_count,
                                                      coefficients, intercepts, raw_class_count);
  LinearClassifierExpansion expansion =
      ExpandBinaryScores(std::move(raw_scores), sample_count, raw_class_count,
                         static_cast<int64_t>(class_labels.size()));
  EXT_ENFORCE_INVALID(static_cast<int64_t>(class_labels.size()) == expansion.score_class_count,
                      "kernel::LinearClassifier class_labels size must match the number of "
                      "score columns.");

  std::vector<std::string> labels(static_cast<size_t>(sample_count));
  for (int64_t n = 0; n < sample_count; ++n) {
    const int64_t idx = ArgMax(expansion.scores.data() + n * expansion.score_class_count,
                               expansion.score_class_count);
    labels[static_cast<size_t>(n)] = class_labels[static_cast<size_t>(idx)];
  }
  Tensor y = Tensor::FromStrings("", {sample_count}, labels);
  Tensor z = Tensor::FromFloat("", {sample_count, expansion.score_class_count}, expansion.scores);
  return std::make_pair(std::move(y), std::move(z));
}

#define ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(T)                                                \
  template std::pair<Tensor, Tensor> LinearClassifier::operator()<T>(                              \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<int64_t> &, const std::string &) const;                                    \
  template std::pair<Tensor, Tensor> LinearClassifier::operator()<T>(                              \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<std::string> &, const std::string &) const

ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(float);
ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(double);
ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(int64_t);
ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
