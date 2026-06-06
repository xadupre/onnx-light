// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor Softplus::operator()(const Tensor &x) const {
  Tensor y("", DataType::FLOAT, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(float)));
  (*this)(x, y);
  return y;
}

void Softplus::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT,
                      "kernel::Softplus only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::Softplus preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Softplus preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::Softplus preallocated output buffer has unexpected size in bytes.");

  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const size_t idx = static_cast<size_t>(i);
    // Numerically stable softplus: log1p(exp(-|x|)) + max(x, 0).
    const float xv = px[idx];
    const float abs_x = std::fabs(xv);
    py[idx] = std::log1p(std::exp(-abs_x)) + std::fmax(xv, 0.0f);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
