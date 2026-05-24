// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {
namespace detail {

BroadcastInfo CheckBinaryBroadcast(const char *op_name, const char *dtype_name,
                                   int32_t expected_dtype, const Tensor &x, const Tensor &y) {
  if (x.data_type != expected_dtype || y.data_type != expected_dtype) {
    throw std::invalid_argument(std::string(op_name) + " only supports " + dtype_name +
                                " tensors.");
  }
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  if (!(nx == ny || nx == 1 || ny == 1)) {
    throw std::invalid_argument(std::string(op_name) +
                                " only supports equal-shape tensors or scalar broadcasting.");
  }
  BroadcastInfo bi;
  bi.nx = nx;
  bi.ny = ny;
  bi.element_count = nx >= ny ? nx : ny;
  bi.shape = nx >= ny ? x.shape : y.shape;
  return bi;
}

void CheckPreallocatedOutput(const char *op_name, const char *dtype_name, int32_t expected_dtype,
                             const std::vector<int64_t> &expected_shape, size_t expected_bytes,
                             const Tensor *output) {
  if (output == nullptr) {
    throw std::invalid_argument(std::string(op_name) +
                                " requires a non-null preallocated output tensor.");
  }
  if (output->data_type != expected_dtype) {
    throw std::invalid_argument(std::string(op_name) + " preallocated output must be a " +
                                dtype_name + " tensor.");
  }
  if (output->shape != expected_shape) {
    throw std::invalid_argument(std::string(op_name) +
                                " preallocated output shape must match the broadcasted "
                                "input shape.");
  }
  if (output->data.size() != expected_bytes) {
    throw std::invalid_argument(std::string(op_name) +
                                " preallocated output buffer has unexpected size in bytes.");
  }
}

} // namespace detail
} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
