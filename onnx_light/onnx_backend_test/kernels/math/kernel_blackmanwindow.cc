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

Tensor BlackmanWindow::operator()(const Tensor &size, bool periodic) const {
  if (size.data_type != TensorProto::DataType::INT32) {
    throw std::invalid_argument("kernel::BlackmanWindow expects an INT32 size tensor.");
  }
  if (size.element_count() != 1 || !size.shape.empty()) {
    throw std::invalid_argument("kernel::BlackmanWindow expects a scalar size tensor.");
  }
  const int32_t n = size.AsInt32()[0];
  if (n < 0) {
    throw std::invalid_argument("kernel::BlackmanWindow size must be non-negative.");
  }
  Tensor y("", TensorProto::DataType::FLOAT, {n},
           std::vector<uint8_t>(static_cast<size_t>(n) * sizeof(float)));
  (*this)(size, periodic, y);
  return y;
}

void BlackmanWindow::operator()(const Tensor &size, bool periodic, Tensor &output) const {
  if (size.data_type != TensorProto::DataType::INT32) {
    throw std::invalid_argument("kernel::BlackmanWindow expects an INT32 size tensor.");
  }
  if (size.element_count() != 1 || !size.shape.empty()) {
    throw std::invalid_argument("kernel::BlackmanWindow expects a scalar size tensor.");
  }
  if (output.data_type != TensorProto::DataType::FLOAT) {
    throw std::invalid_argument(
        "kernel::BlackmanWindow preallocated output must be a FLOAT tensor.");
  }

  const int32_t n = size.AsInt32()[0];
  if (n < 0) {
    throw std::invalid_argument("kernel::BlackmanWindow size must be non-negative.");
  }
  if (!periodic && n <= 1) {
    throw std::invalid_argument("kernel::BlackmanWindow symmetric variant requires size > 1.");
  }
  if (output.shape.size() != 1 || output.shape[0] != n) {
    throw std::invalid_argument("kernel::BlackmanWindow preallocated output shape must be {size}.");
  }
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  if (output.data.size() != expected_bytes) {
    throw std::invalid_argument(
        "kernel::BlackmanWindow preallocated output buffer has unexpected size in bytes.");
  }

  constexpr double kPi = 3.14159265358979323846;
  constexpr double a0 = 0.42;
  constexpr double a1 = -0.5;
  constexpr double a2 = 0.08;
  const double divisor = periodic ? static_cast<double>(n) : static_cast<double>(n - 1);

  float *py = output.AsFloat();
  for (int32_t i = 0; i < n; ++i) {
    const double k = static_cast<double>(i) / divisor;
    py[static_cast<size_t>(i)] =
        static_cast<float>(a0 + a1 * std::cos(2.0 * kPi * k) + a2 * std::cos(4.0 * kPi * k));
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
