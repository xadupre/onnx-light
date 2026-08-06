// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_svm_common.h"

#include <cmath>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

double ComputeSvmKernel(const char *kernel_type, const double *x, const float *sv,
                        int64_t feature_count, float gamma, float coef0, float degree) {
  std::string kind = kernel_type == nullptr ? "LINEAR" : std::string(kernel_type);
  double dot = 0.0;
  for (int64_t j = 0; j < feature_count; ++j) {
    dot += x[j] * static_cast<double>(sv[j]);
  }
  if (kind == "LINEAR") {
    return dot;
  }
  if (kind == "POLY") {
    return std::pow(static_cast<double>(gamma) * dot + static_cast<double>(coef0),
                    static_cast<double>(degree));
  }
  if (kind == "RBF") {
    double sq_norm = 0.0;
    for (int64_t j = 0; j < feature_count; ++j) {
      const double d = x[j] - static_cast<double>(sv[j]);
      sq_norm += d * d;
    }
    return std::exp(-static_cast<double>(gamma) * sq_norm);
  }
  if (kind == "SIGMOID") {
    return std::tanh(static_cast<double>(gamma) * dot + static_cast<double>(coef0));
  }
  EXT_THROW_INVALID("Unsupported SVM kernel_type: ", kind);
}

void ValidateFeatureMatrixShape(const Tensor &x, int64_t &sample_count, int64_t &feature_count) {
  EXT_ENFORCE_INVALID(!x.shape.empty(), "SVM kernels expect input rank 1 or 2.");
  EXT_ENFORCE_INVALID(x.shape.size() == 1 || x.shape.size() == 2,
                      "SVM kernels expect input rank 1 or 2.");
  if (x.shape.size() == 1) {
    sample_count = 1;
    feature_count = x.shape[0];
  } else {
    sample_count = x.shape[0];
    feature_count = x.shape[1];
  }
  EXT_ENFORCE_INVALID(sample_count >= 0 && feature_count >= 0,
                      "SVM kernels expect non-negative input dimensions.");
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
