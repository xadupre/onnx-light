// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor Constant::operator()(const Tensor &value) const {
  Tensor out("", value.data_type, value.shape, std::vector<uint8_t>(value.data.size()));
  (*this)(value, out);
  return out;
}

void Constant::operator()(const Tensor &value, Tensor &output) const {
  EXT_ENFORCE_INVALID(
      output.data_type == value.data_type,
      "kernel::Constant preallocated output must have the same data type as the value.");
  EXT_ENFORCE_INVALID(output.shape == value.shape,
                      "kernel::Constant preallocated output shape must match the value shape.");
  EXT_ENFORCE_INVALID(output.data.size() == value.data.size(),
                      "kernel::Constant preallocated output buffer has unexpected size in bytes.");
  if (!value.data.empty()) {
    std::memcpy(output.data.data(), value.data.data(), value.data.size());
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
