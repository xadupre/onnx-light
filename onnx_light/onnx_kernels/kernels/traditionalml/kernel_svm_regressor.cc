// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_backend_test/kernels/traditionalml/kernel_svm_common.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

template <typename T>
Tensor SVMRegressor::operator()(const Tensor &x, const std::vector<float> &support_vectors,
                                const std::vector<float> &coefficients,
                                const std::vector<float> &rho, const char *kernel_type, float gamma,
                                float coef0, float degree) const {
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  const int64_t support_count =
      feature_count == 0 ? 0 : static_cast<int64_t>(support_vectors.size()) / feature_count;
  EXT_ENFORCE_INVALID(
      static_cast<int64_t>(support_vectors.size()) == support_count * feature_count,
      "kernel::SVMRegressor support_vectors size must be divisible by feature_count.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(coefficients.size()) == support_count,
                      "kernel::SVMRegressor coefficients size must match support vector count.");
  EXT_ENFORCE_INVALID(!rho.empty(), "kernel::SVMRegressor rho must be non-empty.");

  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  std::vector<float> predictions(static_cast<size_t>(sample_count), 0.0f);
  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values.data() + n * feature_count;
    double value = 0.0;
    for (int64_t i = 0; i < support_count; ++i) {
      const float *sv = support_vectors.data() + i * feature_count;
      const double k =
          ComputeSvmKernel(kernel_type, x_row, sv, feature_count, gamma, coef0, degree);
      value += static_cast<double>(coefficients[static_cast<size_t>(i)]) * k;
    }
    value -= static_cast<double>(rho[0]);
    predictions[static_cast<size_t>(n)] = static_cast<float>(value);
  }
  return Tensor::FromFloat("", {sample_count, 1}, predictions);
}

#define ONNX_LIGHT_INSTANTIATE_SVM_REGRESSOR(T)                                                    \
  template Tensor SVMRegressor::operator()<T>(                                                     \
      const Tensor &, const std::vector<float> &, const std::vector<float> &,                      \
      const std::vector<float> &, const char *, float, float, float) const

ONNX_LIGHT_INSTANTIATE_SVM_REGRESSOR(float);
ONNX_LIGHT_INSTANTIATE_SVM_REGRESSOR(double);
ONNX_LIGHT_INSTANTIATE_SVM_REGRESSOR(int64_t);
ONNX_LIGHT_INSTANTIATE_SVM_REGRESSOR(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_SVM_REGRESSOR

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
