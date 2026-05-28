// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor Atanh::operator()(const Tensor &x) const {
  Tensor y("", TensorProto::DataType::FLOAT, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(float)));
  (*this)(x, y);
  return y;
}

void Atanh::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == TensorProto::DataType::FLOAT,
                      "kernel::Atanh only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == TensorProto::DataType::FLOAT,
                      "kernel::Atanh preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Atanh preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::Atanh preallocated output buffer has unexpected size in bytes.");
  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    py[static_cast<size_t>(i)] = std::atanh(px[i]);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
