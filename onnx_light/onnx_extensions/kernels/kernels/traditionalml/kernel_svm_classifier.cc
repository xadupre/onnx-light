// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_svm_common.h"

#include "onnx_core/runtime/simple_tensor.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

Tensor ComputeBinaryDecisionScores(const std::vector<double> &x_values, int64_t sample_count,
                                   int64_t feature_count, const std::vector<float> &support_vectors,
                                   const std::vector<float> &coefficients,
                                   const std::vector<float> &rho,
                                   const std::vector<int64_t> &vectors_per_class,
                                   const char *kernel_type, float gamma, float coef0, float degree,
                                   RawBufferAllocator *allocator) {
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

  // The per-sample decision scores are held in an allocator-backed scratch
  // buffer so no working memory is allocated outside the runtime allocator;
  // it falls back to inline storage when ``allocator`` is null.
  const size_t scores_n_bytes = static_cast<size_t>(sample_count) * sizeof(float);
  Tensor scores_buf = MakeOutputTensor(DataType::FLOAT, {sample_count}, scores_n_bytes, allocator);
  float *scores = scores_buf.AsFloat();
  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values.data() + n * feature_count;
    double decision = 0.0;
    for (int64_t i = 0; i < expected_supports; ++i) {
      const float *sv = support_vectors.data() + i * feature_count;
      const double k =
          ComputeSvmKernel(kernel_type, x_row, sv, feature_count, gamma, coef0, degree);
      decision += static_cast<double>(coefficients[static_cast<size_t>(i)]) * k;
    }
    decision += static_cast<double>(rho[0]);
    scores[static_cast<size_t>(n)] = static_cast<float>(decision);
  }
  return scores_buf;
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
  const Tensor scores_buf = ComputeBinaryDecisionScores(
      x_values, sample_count, feature_count, support_vectors, coefficients, rho, vectors_per_class,
      kernel_type, gamma, coef0, degree, ctx_.allocator);
  const float *scores = scores_buf.AsFloat();
  std::vector<int64_t> labels(static_cast<size_t>(sample_count));
  std::vector<float> expanded_scores(static_cast<size_t>(sample_count * 2));
  for (int64_t i = 0; i < sample_count; ++i) {
    const float s = scores[static_cast<size_t>(i)];
    labels[static_cast<size_t>(i)] = s > 0.0f ? class_labels[0] : class_labels[1];
    expanded_scores[static_cast<size_t>(i * 2)] = -s;
    expanded_scores[static_cast<size_t>(i * 2 + 1)] = s;
  }
  Tensor y = Tensor::FromInt64("", {sample_count}, labels, ctx_.allocator);
  Tensor z = Tensor::FromFloat("", {sample_count, 2}, expanded_scores, ctx_.allocator);
  return std::make_pair(std::move(y), std::move(z));
}

template <typename T>
std::pair<Tensor, Tensor>
SVMClassifier::operator()(const Tensor &x, const std::vector<float> &support_vectors,
                          const std::vector<float> &coefficients, const std::vector<float> &rho,
                          const std::vector<int64_t> &vectors_per_class,
                          const ParamStrings &class_labels, const char *kernel_type, float gamma,
                          float coef0, float degree) const {
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  EXT_ENFORCE_INVALID(class_labels.size() == 2,
                      "kernel::SVMClassifier requires exactly two string class labels.");
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  const Tensor scores_buf = ComputeBinaryDecisionScores(
      x_values, sample_count, feature_count, support_vectors, coefficients, rho, vectors_per_class,
      kernel_type, gamma, coef0, degree, ctx_.allocator);
  const float *scores = scores_buf.AsFloat();
  std::vector<std::string> labels(static_cast<size_t>(sample_count));
  std::vector<float> expanded_scores(static_cast<size_t>(sample_count * 2));
  for (int64_t i = 0; i < sample_count; ++i) {
    const float s = scores[static_cast<size_t>(i)];
    labels[static_cast<size_t>(i)] = s > 0.0f ? class_labels[0] : class_labels[1];
    expanded_scores[static_cast<size_t>(i * 2)] = -s;
    expanded_scores[static_cast<size_t>(i * 2 + 1)] = s;
  }
  Tensor y = Tensor::FromStrings("", {sample_count}, labels);
  Tensor z = Tensor::FromFloat("", {sample_count, 2}, expanded_scores, ctx_.allocator);
  return std::make_pair(std::move(y), std::move(z));
}

#define ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(T)                                                   \
  template std::pair<Tensor, Tensor> SVMClassifier::operator()<T>(                                 \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<float> &, const std::vector<int64_t> &, const std::vector<int64_t> &,      \
      const char *, float, float, float) const;                                                    \
  template std::pair<Tensor, Tensor> SVMClassifier::operator()<T>(                                 \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<float> &, const std::vector<int64_t> &, const ParamStrings &,              \
      const char *, float, float, float) const

ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(float);
ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(double);
ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(int64_t);
ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER

void SVMClassifier::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 2);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const SVMCommonAttrs a = ParseSVMCommonAttrs(node, "SVMClassifier");
  const std::vector<int64_t> vectors_per_class =
      GetAttributeIntsOrDefault(node, "vectors_per_class", {});
  const std::vector<int64_t> classlabels_ints =
      GetAttributeIntsOrDefault(node, "classlabels_ints", {});
  const ParamStrings classlabels_strings =
      GetAttributeStringsOrDefault(node, "classlabels_strings", {});
  const bool use_strings = !classlabels_strings.empty();
  const bool has_ints = !classlabels_ints.empty();
  EXT_ENFORCE_INVALID(use_strings != has_ints,
                      "RunNode: SVMClassifier requires exactly one of 'classlabels_ints' or "
                      "'classlabels_strings' to be set.");
  onnx_kernels::kernel::SVMClassifier svm(rt.kernel_ctx());
  std::pair<Tensor, Tensor> yz = DispatchSVMByDataType(x, "SVMClassifier", [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    (void)tag;
    return use_strings
               ? svm.template operator()<T>(x, a.support_vectors, a.coefficients, a.rho,
                                            vectors_per_class, classlabels_strings,
                                            a.kernel_type.c_str(), a.gamma, a.coef0, a.degree)
               : svm.template operator()<T>(x, a.support_vectors, a.coefficients, a.rho,
                                            vectors_per_class, classlabels_ints,
                                            a.kernel_type.c_str(), a.gamma, a.coef0, a.degree);
  });
  SetOutput(node, 0, std::move(yz.first), rt);
  SetOutput(node, 1, std::move(yz.second), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
