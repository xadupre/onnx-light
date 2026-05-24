// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"

#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor Xor::operator()(const Tensor &x, const Tensor &y) const {
  if (x.data_type != TensorProto::DataType::BOOL || y.data_type != TensorProto::DataType::BOOL) {
    throw std::invalid_argument("kernel::Xor only supports BOOL tensors.");
  }

  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  if (!(nx == ny || nx == 1 || ny == 1)) {
    throw std::invalid_argument(
        "kernel::Xor only supports equal-shape tensors or scalar broadcasting.");
  }

  const int64_t n = nx >= ny ? nx : ny;
  const uint8_t *px = x.data.data();
  const uint8_t *py = y.data.data();
  std::vector<uint8_t> z(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const uint8_t a = nx == 1 ? px[0] : px[i];
    const uint8_t b = ny == 1 ? py[0] : py[i];
    z[static_cast<size_t>(i)] = ((a != 0) != (b != 0)) ? 1 : 0;
  }

  const std::vector<int64_t> &shape = nx >= ny ? x.shape : y.shape;
  return Tensor("", TensorProto::DataType::BOOL, shape, std::move(z));
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
