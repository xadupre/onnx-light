// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/logical/include_logical_kernels.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor IsInf::operator()(const Tensor &x, int64_t detect_positive, int64_t detect_negative) const {
  Tensor y("", DataType::BOOL, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count())));
  (*this)(x, detect_positive, detect_negative, y);
  return y;
}

void IsInf::operator()(const Tensor &x, int64_t detect_positive, int64_t detect_negative,
                       Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT, "kernel::IsInf only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::BOOL,
                      "kernel::IsInf preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::IsInf preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::IsInf preallocated output buffer has unexpected size in bytes.");
  const bool report_pos = detect_positive != 0;
  const bool report_neg = detect_negative != 0;
  const float *px = x.AsFloat();
  uint8_t *py = output.data.data();
  for (int64_t i = 0; i < n; ++i) {
    const float v = px[i];
    uint8_t r = 0;
    if (std::isinf(v)) {
      if (v > 0.0f) {
        r = report_pos ? 1u : 0u;
      } else {
        r = report_neg ? 1u : 0u;
      }
    }
    py[static_cast<size_t>(i)] = r;
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
