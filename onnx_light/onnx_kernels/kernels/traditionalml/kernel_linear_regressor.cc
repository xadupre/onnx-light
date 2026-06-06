// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_kernels/kernels/traditionalml/kernel_svm_common.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

template <typename T>
Tensor LinearRegressor::operator()(const Tensor &x, const std::vector<float> &coefficients,
                                   const std::vector<float> &intercepts, int64_t targets,
                                   const std::string &post_transform) const {
  EXT_ENFORCE_INVALID(targets >= 1, "kernel::LinearRegressor 'targets' must be >= 1.");
  EXT_ENFORCE_INVALID(post_transform == "NONE",
                      "kernel::LinearRegressor only supports post_transform == 'NONE'.");

  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  EXT_ENFORCE_INVALID(static_cast<int64_t>(coefficients.size()) == targets * feature_count,
                      "kernel::LinearRegressor coefficients size must be targets * feature_count.");
  EXT_ENFORCE_INVALID(intercepts.empty() || static_cast<int64_t>(intercepts.size()) == targets,
                      "kernel::LinearRegressor intercepts size must be 0 or equal to targets.");

  const std::vector<double> x_values = ToDoubleRowMajor<T>(x, sample_count, feature_count);
  std::vector<float> predictions(static_cast<size_t>(sample_count * targets), 0.0f);
  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values.data() + n * feature_count;
    for (int64_t t = 0; t < targets; ++t) {
      const float *w = coefficients.data() + t * feature_count;
      double value = 0.0;
      for (int64_t j = 0; j < feature_count; ++j) {
        value += x_row[j] * static_cast<double>(w[j]);
      }
      if (!intercepts.empty()) {
        value += static_cast<double>(intercepts[static_cast<size_t>(t)]);
      }
      predictions[static_cast<size_t>(n * targets + t)] = static_cast<float>(value);
    }
  }
  return Tensor::FromFloat("", {sample_count, targets}, predictions);
}

#define ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(T)                                                 \
  template Tensor LinearRegressor::operator()<T>(const Tensor &, const std::vector<float> &,       \
                                                 const std::vector<float> &, int64_t,              \
                                                 const std::string &) const

ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(float);
ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(double);
ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(int64_t);
ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
