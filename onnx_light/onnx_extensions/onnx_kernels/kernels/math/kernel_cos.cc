// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/onnx_kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor Cos::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * sizeof(float);
  Tensor y = MakeOutputTensor(DataType::FLOAT, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, y);
  return y;
}

void Cos::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT, "kernel::Cos only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::Cos preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Cos preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::Cos preallocated output buffer has unexpected size in bytes.");
  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    py[static_cast<size_t>(i)] = std::cos(px[i]);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
