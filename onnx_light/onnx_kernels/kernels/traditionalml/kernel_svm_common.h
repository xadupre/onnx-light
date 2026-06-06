// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/simple_tensor.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

inline double ComputeSvmKernel(const char *kernel_type, const double *x, const float *sv,
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
  throw std::invalid_argument("Unsupported SVM kernel_type: " + kind);
}

template <typename T>
inline std::vector<double> ToDoubleRowMajor(const Tensor &x, int64_t sample_count,
                                            int64_t feature_count) {
  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "SVM kernel input data_type does not match the requested T.");
  const T *px = x.As<T>();
  std::vector<double> values(static_cast<size_t>(sample_count * feature_count));
  for (int64_t i = 0; i < sample_count * feature_count; ++i) {
    values[static_cast<size_t>(i)] = static_cast<double>(px[i]);
  }
  return values;
}

inline void ValidateFeatureMatrixShape(const Tensor &x, int64_t &sample_count,
                                       int64_t &feature_count) {
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

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
