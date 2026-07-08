// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include "onnx_kernels/runtime_context.h"
#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor Constant::operator()(RuntimeContext *rt, const Tensor &value) const {
  const size_t out_n_bytes = value.size_bytes();
  Tensor out =
      MakeOutputTensor(value.data_type, value.shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(value, out);
  return out;
}

void Constant::operator()(const Tensor &value, Tensor &output) const {
  EXT_ENFORCE_INVALID(
      output.data_type == value.data_type,
      "kernel::Constant preallocated output must have the same data type as the value.");
  EXT_ENFORCE_INVALID(output.shape == value.shape,
                      "kernel::Constant preallocated output shape must match the value shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == value.size_bytes(),
                      "kernel::Constant preallocated output buffer has unexpected size in bytes.");
  if (value.size_bytes() > 0) {
    std::memcpy(output.mutable_bytes(), value.bytes(), value.size_bytes());
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
