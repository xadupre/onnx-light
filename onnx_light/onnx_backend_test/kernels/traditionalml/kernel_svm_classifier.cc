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

std::vector<float>
ComputeBinaryDecisionScores(const std::vector<double> &x_values, int64_t sample_count,
                            int64_t feature_count, const std::vector<float> &support_vectors,
                            const std::vector<float> &coefficients, const std::vector<float> &rho,
                            const std::vector<int64_t> &vectors_per_class, const char *kernel_type,
                            float gamma, float coef0, float degree) {
  EXT_ENFORCE_INVALID(vectors_per_class.size() == 2,
                      "kernel::SVMClassifier currently supports binary classifiers only.");
  const int64_t expected_supports = vectors_per_class[0] + vectors_per_class[1];
  EXT_ENFORCE_INVALID(expected_supports >= 0,
                      "kernel::SVMClassifier vectors_per_class must be non-negative.");
  EXT_ENFORCE_INVALID(
      static_cast<int64_t>(support_vectors.size()) == expected_supports * feature_count,
      "kernel::SVMClassifier support_vectors size must be vectors_per_class_sum * feature_count.");
  EXT_ENFORCE_INVALID(
      static_cast<int64_t>(coefficients.size()) == expected_supports,
      "kernel::SVMClassifier coefficients size must match number of support vectors.");
  EXT_ENFORCE_INVALID(!rho.empty(), "kernel::SVMClassifier rho must be non-empty.");

  std::vector<float> scores(static_cast<size_t>(sample_count), 0.0f);
  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values.data() + n * feature_count;
    double decision = 0.0;
    for (int64_t i = 0; i < expected_supports; ++i) {
      const float *sv = support_vectors.data() + i * feature_count;
      const double k =
          ComputeSvmKernel(kernel_type, x_row, sv, feature_count, gamma, coef0, degree);
      decision += static_cast<double>(coefficients[static_cast<size_t>(i)]) * k;
    }
    decision -= static_cast<double>(rho[0]);
    scores[static_cast<size_t>(n)] = static_cast<float>(decision);
  }
  return scores;
}

} // namespace

template <typename T>
std::pair<Tensor, Tensor>
SVMClassifier::operator()(const Tensor &x, const std::vector<float> &support_vectors,
                          const std::vector<float> &coefficients, const std::vector<float> &rho,
                          const std::vector<int64_t> &vectors_per_class,
                          const std::vector<int64_t> &class_labels, const char *kernel_type,
                          float gamma, float coef0, float degree) const {
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  EXT_ENFORCE_INVALID(class_labels.size() == 2,
                      "kernel::SVMClassifier requires exactly two int64 class labels.");
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  const std::vector<float> scores = ComputeBinaryDecisionScores(
      x_values, sample_count, feature_count, support_vectors, coefficients, rho, vectors_per_class,
      kernel_type, gamma, coef0, degree);
  std::vector<int64_t> labels(static_cast<size_t>(sample_count));
  for (int64_t i = 0; i < sample_count; ++i) {
    labels[static_cast<size_t>(i)] =
        scores[static_cast<size_t>(i)] > 0.0f ? class_labels[1] : class_labels[0];
  }
  Tensor y = Tensor::FromInt64("", {sample_count}, labels);
  Tensor z = Tensor::FromFloat("", {sample_count, 1}, scores);
  return std::make_pair(std::move(y), std::move(z));
}

template <typename T>
std::pair<Tensor, Tensor>
SVMClassifier::operator()(const Tensor &x, const std::vector<float> &support_vectors,
                          const std::vector<float> &coefficients, const std::vector<float> &rho,
                          const std::vector<int64_t> &vectors_per_class,
                          const std::vector<std::string> &class_labels, const char *kernel_type,
                          float gamma, float coef0, float degree) const {
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  EXT_ENFORCE_INVALID(class_labels.size() == 2,
                      "kernel::SVMClassifier requires exactly two string class labels.");
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  const std::vector<float> scores = ComputeBinaryDecisionScores(
      x_values, sample_count, feature_count, support_vectors, coefficients, rho, vectors_per_class,
      kernel_type, gamma, coef0, degree);
  std::vector<std::string> labels(static_cast<size_t>(sample_count));
  for (int64_t i = 0; i < sample_count; ++i) {
    labels[static_cast<size_t>(i)] =
        scores[static_cast<size_t>(i)] > 0.0f ? class_labels[1] : class_labels[0];
  }
  Tensor y = Tensor::FromStrings("", {sample_count}, labels);
  Tensor z = Tensor::FromFloat("", {sample_count, 1}, scores);
  return std::make_pair(std::move(y), std::move(z));
}

#define ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(T)                                                   \
  template std::pair<Tensor, Tensor> SVMClassifier::operator()<T>(                                 \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<float> &, const std::vector<int64_t> &, const std::vector<int64_t> &,      \
      const char *, float, float, float) const;                                                    \
  template std::pair<Tensor, Tensor> SVMClassifier::operator()<T>(                                 \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<float> &, const std::vector<int64_t> &, const std::vector<std::string> &,  \
      const char *, float, float, float) const

ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(float);
ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(double);
ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(int64_t);
ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
