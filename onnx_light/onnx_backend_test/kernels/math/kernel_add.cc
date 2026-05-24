// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor Add::operator()(const Tensor &x, const Tensor &y) const {
  if (x.data_type != TensorProto::DataType::FLOAT || y.data_type != TensorProto::DataType::FLOAT) {
    throw std::invalid_argument("kernel::Add only supports FLOAT tensors.");
  }

  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  if (!(nx == ny || nx == 1 || ny == 1)) {
    throw std::invalid_argument(
        "kernel::Add only supports equal-shape tensors or scalar broadcasting.");
  }

  const int64_t n = nx >= ny ? nx : ny;
  const float *px = x.AsFloat();
  const float *py = y.AsFloat();
  std::vector<float> z(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const float a = nx == 1 ? px[0] : px[i];
    const float b = ny == 1 ? py[0] : py[i];
    z[static_cast<size_t>(i)] = a + b;
  }

  const std::vector<int64_t> &shape = nx >= ny ? x.shape : y.shape;
  return Tensor::FromFloat("", shape, z);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
