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
  const std::vector<int64_t> &shape = nx >= ny ? x.shape : y.shape;
  const int64_t n = nx >= ny ? nx : ny;
  Tensor z("", TensorProto::DataType::FLOAT, shape,
           std::vector<uint8_t>(static_cast<size_t>(n) * sizeof(float)));
  (*this)(x, y, &z);
  return z;
}

void Add::operator()(const Tensor &x, const Tensor &y, Tensor *output) const {
  if (x.data_type != TensorProto::DataType::FLOAT || y.data_type != TensorProto::DataType::FLOAT) {
    throw std::invalid_argument("kernel::Add only supports FLOAT tensors.");
  }
  if (output == nullptr) {
    throw std::invalid_argument("kernel::Add requires a non-null preallocated output tensor.");
  }
  if (output->data_type != TensorProto::DataType::FLOAT) {
    throw std::invalid_argument("kernel::Add preallocated output must be a FLOAT tensor.");
  }
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  if (!(nx == ny || nx == 1 || ny == 1)) {
    throw std::invalid_argument(
        "kernel::Add only supports equal-shape tensors or scalar broadcasting.");
  }
  const std::vector<int64_t> &expected_shape = nx >= ny ? x.shape : y.shape;
  if (output->shape != expected_shape) {
    throw std::invalid_argument(
        "kernel::Add preallocated output shape must match the broadcasted input shape.");
  }
  const int64_t n = nx >= ny ? nx : ny;
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  if (output->data.size() != expected_bytes) {
    throw std::invalid_argument(
        "kernel::Add preallocated output buffer has unexpected size in bytes.");
  }
  const float *px = x.AsFloat();
  const float *py = y.AsFloat();
  float *pz = output->AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const float a = nx == 1 ? px[0] : px[i];
    const float b = ny == 1 ? py[0] : py[i];
    pz[static_cast<size_t>(i)] = a + b;
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
