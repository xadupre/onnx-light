// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::softmax_detail {

template <typename T>
std::pair<T, T> ComputeMaxAndLogSum(const T *values, int64_t count, int64_t stride) {
  T max_value = -std::numeric_limits<T>::infinity();
  int64_t max_index = 0;
  for (int64_t index = 0; index < count; ++index) {
    const T value = values[index * stride];
    if (value > max_value) {
      max_value = value;
      max_index = index;
    }
  }

  T sum = 0;
  for (int64_t index = 0; index < count; ++index) {
    if (index == max_index) {
      continue;
    }
    sum += std::exp(values[index * stride] - max_value);
  }
  const T log_sum =
      std::isfinite(max_value) ? std::log1p(sum) : std::numeric_limits<T>::quiet_NaN();
  return {max_value, log_sum};
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::softmax_detail
