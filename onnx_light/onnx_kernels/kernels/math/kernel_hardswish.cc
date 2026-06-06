// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <algorithm>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// HardSwish parameters are fixed by the ONNX spec: alpha = 1/6, beta = 0.5.
constexpr float kHardSwishAlpha = 1.0f / 6.0f;
constexpr float kHardSwishBeta = 0.5f;

} // namespace

Tensor HardSwish::operator()(const Tensor &x) const {
  Tensor y("", DataType::FLOAT, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(float)));
  (*this)(x, y);
  return y;
}

void HardSwish::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT,
                      "kernel::HardSwish only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::HardSwish preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::HardSwish preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::HardSwish preallocated output buffer has unexpected size in bytes.");

  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const size_t idx = static_cast<size_t>(i);
    const float v = px[idx];
    const float hs = std::max(0.0f, std::min(1.0f, kHardSwishAlpha * v + kHardSwishBeta));
    py[idx] = v * hs;
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
