// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor IsNaN::operator()(const Tensor &x) const {
  Tensor y("", DataType::BOOL, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count())));
  (*this)(x, y);
  return y;
}

void IsNaN::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT, "kernel::IsNaN only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::BOOL,
                      "kernel::IsNaN preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::IsNaN preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::IsNaN preallocated output buffer has unexpected size in bytes.");
  const float *px = x.AsFloat();
  uint8_t *py = output.data.data();
  for (int64_t i = 0; i < n; ++i) {
    py[static_cast<size_t>(i)] = std::isnan(px[i]) ? 1u : 0u;
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
