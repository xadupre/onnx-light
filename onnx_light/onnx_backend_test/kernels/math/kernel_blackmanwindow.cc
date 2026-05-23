// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/kernels_math.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor BlackmanWindow(const Tensor &size, bool periodic) {
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
  if (!periodic && n <= 1) {
    throw std::invalid_argument("kernel::BlackmanWindow symmetric variant requires size > 1.");
  }

  constexpr double kPi = 3.14159265358979323846;
  constexpr double a0 = 0.42;
  constexpr double a1 = -0.5;
  constexpr double a2 = 0.08;
  const double divisor = periodic ? static_cast<double>(n) : static_cast<double>(n - 1);

  std::vector<float> y(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const double k = static_cast<double>(i) / divisor;
    y[static_cast<size_t>(i)] =
        static_cast<float>(a0 + a1 * std::cos(2.0 * kPi * k) + a2 * std::cos(4.0 * kPi * k));
  }
  return Tensor::FromFloat("", {n}, y);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
