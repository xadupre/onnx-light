// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/simple_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

/// Evaluates the SVM kernel function for a single (sample, support-vector) pair.
/// Supported ``kernel_type`` values: "LINEAR", "POLY", "RBF", "SIGMOID".
double ComputeSvmKernel(const char *kernel_type, const double *x, const float *sv,
                        int64_t feature_count, float gamma, float coef0, float degree);

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

// Same as :cpp:func:`ToDoubleRowMajor` but writes the converted values into an
// allocator-backed ``double`` :cpp:class:`Tensor` (obtained from
// :cpp:func:`MakeOutputTensor`) instead of a temporary ``std::vector<double>``.
template <typename T>
inline Tensor ToDoubleRowMajorTensor(const Tensor &x, int64_t sample_count, int64_t feature_count,
                                     RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "SVM kernel input data_type does not match the requested T.");
  const T *px = x.As<T>();
  const int64_t count = sample_count * feature_count;
  Tensor values =
      MakeOutputTensor(TensorElementType<double>::value, {sample_count, feature_count},
                       PackedByteSize(TensorElementType<double>::value, count), allocator);
  double *pv = values.As<double>();
  for (int64_t i = 0; i < count; ++i) {
    pv[i] = static_cast<double>(px[i]);
  }
  return values;
}

/// Validates the shape of the feature-matrix input ``x`` and populates
/// ``sample_count`` and ``feature_count``. Accepts rank-1 (single sample)
/// and rank-2 (batch) tensors.
void ValidateFeatureMatrixShape(const Tensor &x, int64_t &sample_count, int64_t &feature_count);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
