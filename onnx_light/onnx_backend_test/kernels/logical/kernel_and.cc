// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"

#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor And::operator()(const Tensor &x, const Tensor &y) const {
  if (x.data_type != TensorProto::DataType::BOOL || y.data_type != TensorProto::DataType::BOOL) {
    throw std::invalid_argument("kernel::And only supports BOOL tensors.");
  }
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  if (!(nx == ny || nx == 1 || ny == 1)) {
    throw std::invalid_argument(
        "kernel::And only supports equal-shape tensors or scalar broadcasting.");
  }
  const std::vector<int64_t> &shape = nx >= ny ? x.shape : y.shape;
  const int64_t n = nx >= ny ? nx : ny;
  Tensor z("", TensorProto::DataType::BOOL, shape, std::vector<uint8_t>(static_cast<size_t>(n)));
  (*this)(x, y, &z);
  return z;
}

void And::operator()(const Tensor &x, const Tensor &y, Tensor *output) const {
  if (x.data_type != TensorProto::DataType::BOOL || y.data_type != TensorProto::DataType::BOOL) {
    throw std::invalid_argument("kernel::And only supports BOOL tensors.");
  }
  if (output == nullptr) {
    throw std::invalid_argument("kernel::And requires a non-null preallocated output tensor.");
  }
  if (output->data_type != TensorProto::DataType::BOOL) {
    throw std::invalid_argument("kernel::And preallocated output must be a BOOL tensor.");
  }
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  if (!(nx == ny || nx == 1 || ny == 1)) {
    throw std::invalid_argument(
        "kernel::And only supports equal-shape tensors or scalar broadcasting.");
  }
  const std::vector<int64_t> &expected_shape = nx >= ny ? x.shape : y.shape;
  if (output->shape != expected_shape) {
    throw std::invalid_argument(
        "kernel::And preallocated output shape must match the broadcasted input shape.");
  }
  const int64_t n = nx >= ny ? nx : ny;
  if (output->data.size() != static_cast<size_t>(n)) {
    throw std::invalid_argument(
        "kernel::And preallocated output buffer has unexpected size in bytes.");
  }
  const uint8_t *px = x.data.data();
  const uint8_t *py = y.data.data();
  uint8_t *pz = output->data.data();
  for (int64_t i = 0; i < n; ++i) {
    const uint8_t a = nx == 1 ? px[0] : px[i];
    const uint8_t b = ny == 1 ? py[0] : py[i];
    pz[static_cast<size_t>(i)] = (a != 0 && b != 0) ? 1 : 0;
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
