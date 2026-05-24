// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"

#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor Constant::operator()(const Tensor &value) const {
  Tensor out("", value.data_type, value.shape, std::vector<uint8_t>(value.data.size()));
  (*this)(value, out);
  return out;
}

void Constant::operator()(const Tensor &value, Tensor &output) const {
  if (output.data_type != value.data_type) {
    throw std::invalid_argument(
        "kernel::Constant preallocated output must have the same data type as the value.");
  }
  if (output.shape != value.shape) {
    throw std::invalid_argument(
        "kernel::Constant preallocated output shape must match the value shape.");
  }
  if (output.data.size() != value.data.size()) {
    throw std::invalid_argument(
        "kernel::Constant preallocated output buffer has unexpected size in bytes.");
  }
  if (!value.data.empty()) {
    std::memcpy(output.data.data(), value.data.data(), value.data.size());
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
