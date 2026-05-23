// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/kernel_abs.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor Abs(const Tensor &x) {
  if (x.data_type != TensorProto::DataType::FLOAT) {
    throw std::invalid_argument("kernel::Abs only supports FLOAT tensors.");
  }
  const int64_t n = x.element_count();
  const float *px = x.AsFloat();
  std::vector<float> y(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    y[static_cast<size_t>(i)] = std::fabs(px[i]);
  }
  return Tensor::FromFloat("", x.shape, y);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
