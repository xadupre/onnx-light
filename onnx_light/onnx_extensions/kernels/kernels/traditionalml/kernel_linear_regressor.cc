// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_svm_common.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

template <typename T>
Tensor LinearRegressor::operator()(const Tensor &x, const ParamFloats &coefficients,
                                   const ParamFloats &intercepts, int64_t targets,
                                   const std::string &post_transform, RuntimeContext *rt) const {
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
  return Tensor::FromFloat("", {sample_count, targets}, predictions, ctx_.allocator);
}

#define ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(T)                                                 \
  template Tensor LinearRegressor::operator()<T>(const Tensor &, const ParamFloats &,              \
                                                 const ParamFloats &, int64_t,                     \
                                                 const std::string &, RuntimeContext *) const

ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(float);
ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(double);
ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(int64_t);
ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_LINEAR_REGRESSOR

void LinearRegressor::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const ParamFloats coefficients = GetAttributeFloatsOrDefault(node, "coefficients", {});
  const ParamFloats intercepts = GetAttributeFloatsOrDefault(node, "intercepts", {});
  const int64_t targets = GetAttributeIntOrDefault(node, "targets", 1);
  const std::string post_transform = GetAttributeStringOrDefault(node, "post_transform", "NONE");
  onnx_kernels::kernel::LinearRegressor reg(rt.kernel_ctx());
  Tensor y = DispatchSVMByDataType(x, "LinearRegressor", [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    (void)tag;
    return reg.template operator()<T>(x, coefficients, intercepts, targets, post_transform);
  });
  SetOutput(node, 0, std::move(y), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
