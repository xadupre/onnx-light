// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_svm_common.h"

#include "onnx_core/runtime/memory/simple_tensor.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

float SigmoidProbability(float score, float prob_a, float prob_b) {
  const float value = score * prob_a + prob_b;
  return value >= 0.0f ? std::exp(-value) / (1.0f + std::exp(-value))
                       : 1.0f / (1.0f + std::exp(value));
}

void CoupleBinaryProbability(float pairwise_probability, float *probabilities) {
  const float bounded = std::min(std::max(pairwise_probability, 1.0e-7f), 1.0f - 1.0e-7f);
  const float r[4] = {0.0f, bounded, 1.0f - bounded, 0.0f};
  float q[4] = {};
  float qp[2] = {};
  probabilities[0] = 0.5f;
  probabilities[1] = 0.5f;

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < i; ++j) {
      q[i * 2 + i] += r[j * 2 + i] * r[j * 2 + i];
      q[i * 2 + j] = q[j * 2 + i];
    }
    for (size_t j = i + 1; j < 2; ++j) {
      q[i * 2 + i] += r[j * 2 + i] * r[j * 2 + i];
      q[i * 2 + j] = -r[j * 2 + i] * r[i * 2 + j];
    }
  }

  for (size_t iteration = 0; iteration < 100; ++iteration) {
    float pqp = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
      qp[i] = 0.0f;
      for (size_t j = 0; j < 2; ++j) {
        qp[i] += q[i * 2 + j] * probabilities[j];
      }
      pqp += probabilities[i] * qp[i];
    }
    const float max_error = std::max(std::fabs(qp[0] - pqp), std::fabs(qp[1] - pqp));
    if (max_error < 0.0025f) {
      break;
    }
    for (size_t i = 0; i < 2; ++i) {
      const float diff = (-qp[i] + pqp) / q[i * 2 + i];
      probabilities[i] += diff;
      pqp = (pqp + diff * (diff * q[i * 2 + i] + 2.0f * qp[i])) / ((1.0f + diff) * (1.0f + diff));
      for (size_t j = 0; j < 2; ++j) {
        qp[j] = (qp[j] + diff * q[i * 2 + j]) / (1.0f + diff);
        probabilities[j] /= 1.0f + diff;
      }
    }
  }
}

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
                          float gamma, float coef0, float degree, const std::vector<float> &prob_a,
                          const std::vector<float> &prob_b, RuntimeContext *rt) const {
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  EXT_ENFORCE_INVALID(class_labels.size() == 2,
                      "kernel::SVMClassifier requires exactly two int64 class labels.");
  EXT_ENFORCE_INVALID(prob_a.size() == prob_b.size() && (prob_a.empty() || prob_a.size() == 1),
                      "kernel::SVMClassifier binary probability attributes must both contain one "
                      "value or both be empty.");
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  const Tensor scores_buf = ComputeBinaryDecisionScores(
      x_values, sample_count, feature_count, support_vectors, coefficients, rho, vectors_per_class,
      kernel_type, gamma, coef0, degree, rt ? rt->execution_allocator() : ctx_.allocator);
  const float *scores = scores_buf.AsFloat();
  std::vector<int64_t> labels(static_cast<size_t>(sample_count));
  const onnx_kernels::Shape score_shape = {sample_count, 2};
  const size_t score_n_bytes = static_cast<size_t>(sample_count * 2) * sizeof(float);
  Tensor z = rt ? rt->MakeOutputTensor(1, DataType::FLOAT, score_shape, score_n_bytes)
                : MakeOutputTensor(DataType::FLOAT, score_shape, score_n_bytes, ctx_.allocator);
  float *expanded_scores = z.AsFloat();
  for (int64_t i = 0; i < sample_count; ++i) {
    const float s = scores[static_cast<size_t>(i)];
    labels[static_cast<size_t>(i)] = s > 0.0f ? class_labels[0] : class_labels[1];
    if (prob_a.empty()) {
      expanded_scores[i * 2] = -s;
      expanded_scores[i * 2 + 1] = s;
    } else {
      CoupleBinaryProbability(SigmoidProbability(s, prob_a[0], prob_b[0]), expanded_scores + i * 2);
    }
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
SVMClassifier::operator()(const Tensor &x, const std::vector<float> &support_vectors,
                          const std::vector<float> &coefficients, const std::vector<float> &rho,
                          const std::vector<int64_t> &vectors_per_class,
                          const ParamStrings &class_labels, const char *kernel_type, float gamma,
                          float coef0, float degree, const std::vector<float> &prob_a,
                          const std::vector<float> &prob_b, RuntimeContext *rt) const {
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  EXT_ENFORCE_INVALID(class_labels.size() == 2,
                      "kernel::SVMClassifier requires exactly two string class labels.");
  EXT_ENFORCE_INVALID(prob_a.size() == prob_b.size() && (prob_a.empty() || prob_a.size() == 1),
                      "kernel::SVMClassifier binary probability attributes must both contain one "
                      "value or both be empty.");
  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  const Tensor scores_buf = ComputeBinaryDecisionScores(
      x_values, sample_count, feature_count, support_vectors, coefficients, rho, vectors_per_class,
      kernel_type, gamma, coef0, degree, rt ? rt->execution_allocator() : ctx_.allocator);
  const float *scores = scores_buf.AsFloat();
  std::vector<std::string> labels(static_cast<size_t>(sample_count));
  const onnx_kernels::Shape score_shape = {sample_count, 2};
  const size_t score_n_bytes = static_cast<size_t>(sample_count * 2) * sizeof(float);
  Tensor z = rt ? rt->MakeOutputTensor(1, DataType::FLOAT, score_shape, score_n_bytes)
                : MakeOutputTensor(DataType::FLOAT, score_shape, score_n_bytes, ctx_.allocator);
  float *expanded_scores = z.AsFloat();
  for (int64_t i = 0; i < sample_count; ++i) {
    const float s = scores[static_cast<size_t>(i)];
    labels[static_cast<size_t>(i)] = s > 0.0f ? class_labels[0] : class_labels[1];
    if (prob_a.empty()) {
      expanded_scores[i * 2] = -s;
      expanded_scores[i * 2 + 1] = s;
    } else {
      CoupleBinaryProbability(SigmoidProbability(s, prob_a[0], prob_b[0]), expanded_scores + i * 2);
    }
  }
  Tensor y = rt ? rt->MakeOutputTensor(0, DataType::STRING, {sample_count}, 0)
                : Tensor::FromStrings("", {sample_count}, labels);
  if (rt != nullptr) {
    y.string_data = std::move(labels);
  }
  return std::make_pair(std::move(y), std::move(z));
}

#define ONNX_LIGHT_INSTANTIATE_SVM_CLASSIFIER(T)                                                   \
  template std::pair<Tensor, Tensor> SVMClassifier::operator()<T>(                                 \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<float> &, const std::vector<int64_t> &, const std::vector<int64_t> &,      \
      const char *, float, float, float, const std::vector<float> &, const std::vector<float> &,   \
      RuntimeContext *) const;                                                                     \
  template std::pair<Tensor, Tensor> SVMClassifier::operator()<T>(                                 \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<float> &, const std::vector<int64_t> &, const ParamStrings &,              \
      const char *, float, float, float, const std::vector<float> &, const std::vector<float> &,   \
      RuntimeContext *) const

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
  const std::vector<float> prob_a = GetAttributeFloatsOrDefault(node, "prob_a", {});
  const std::vector<float> prob_b = GetAttributeFloatsOrDefault(node, "prob_b", {});
  const bool use_strings = !classlabels_strings.empty();
  const bool has_ints = !classlabels_ints.empty();
  EXT_ENFORCE_INVALID(use_strings != has_ints,
                      "RunNode: SVMClassifier requires exactly one of 'classlabels_ints' or "
                      "'classlabels_strings' to be set.");
  onnx_kernels::kernel::SVMClassifier svm(rt.kernel_ctx());
  std::pair<Tensor, Tensor> yz = DispatchSVMByDataType(x, "SVMClassifier", [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    (void)tag;
    return use_strings ? svm.template operator()<T>(x, a.support_vectors, a.coefficients, a.rho,
                                                    vectors_per_class, classlabels_strings,
                                                    a.kernel_type.c_str(), a.gamma, a.coef0,
                                                    a.degree, prob_a, prob_b, &rt)
                       : svm.template operator()<T>(x, a.support_vectors, a.coefficients, a.rho,
                                                    vectors_per_class, classlabels_ints,
                                                    a.kernel_type.c_str(), a.gamma, a.coef0,
                                                    a.degree, prob_a, prob_b, &rt);
  });
  SetOutput(node, 0, std::move(yz.first), rt);
  SetOutput(node, 1, std::move(yz.second), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
